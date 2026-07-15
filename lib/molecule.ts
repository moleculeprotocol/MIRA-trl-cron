import { MOLECULE_GRAPHQL_ENDPOINT } from "./config.js"

const EXTRACTABLE_CONTENT_TYPES = new Set([
  "application/pdf",
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp",
])

const PROCESS_VERSION = "v3"

const LABS_PER_PAGE = 50

const GET_ALL_PROJECTS_QUERY = `
query GetAllProjects($page: Int, $perPage: Int) {
  labs(page: $page, perPage: $perPage) {
    nodes {
      oclId
      name
    }
    totalCount
    pageInfo {
      hasNextPage
      hasPreviousPage
      currentPage
      totalPages
    }
  }
}
`

interface ProjectNode {
  oclId: string | null
  name: string | null
}

interface PageInfo {
  hasNextPage: boolean
  hasPreviousPage: boolean
  currentPage: number
  totalPages: number
}

interface GetAllProjectsResponse {
  data: {
    labs: {
      nodes: ProjectNode[]
      totalCount: number
      pageInfo: PageInfo
    }
  }
}

export interface ProjectInfo {
  oclId: string
  name: string
}

export interface DataRoomFile {
  did: string
  path: string
  version: number
  contentType: string
  accessLevel: string
  description: string | null
  tags: string[]
  categories: string[]
  downloadUrl: string
}

interface DataRoom {
  id: string
  alias: string
  files: DataRoomFile[]
}

interface ProjectWithDataRoom {
  oclId: string
  shortname: string | null
  dataRoom: DataRoom | null
}

interface GetProjectResponse {
  data: {
    labWithDataRoomAndFiles: ProjectWithDataRoom | null
  }
}

async function hashString(input: string): Promise<string> {
  const encoder = new TextEncoder()
  const data = encoder.encode(input)
  const hashBuffer = await crypto.subtle.digest("SHA-256", data)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("")
}

function filterPublicFiles(files: DataRoomFile[]): DataRoomFile[] {
  return files.filter((file) => file.accessLevel === "PUBLIC")
}

function filterExtractableFiles(files: DataRoomFile[]): DataRoomFile[] {
  return files.filter((file) => EXTRACTABLE_CONTENT_TYPES.has(file.contentType))
}

export async function getAllProjects(): Promise<ProjectInfo[]> {
  const apiKey = process.env.MOLECULE_API_KEY

  if (!apiKey) {
    throw new Error("MOLECULE_API_KEY environment variable is not set")
  }

  const allNodes: ProjectNode[] = []
  // The `page` argument is 0-indexed; pageInfo.currentPage is 1-indexed.
  let page = 0
  let totalPages = 1
  let expectedTotal: number | undefined

  while (page < totalPages) {
    const response = await fetch(MOLECULE_GRAPHQL_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
      },
      body: JSON.stringify({
        query: GET_ALL_PROJECTS_QUERY,
        variables: { page, perPage: LABS_PER_PAGE },
      }),
    })

    if (!response.ok) {
      throw new Error(
        `Failed to fetch projects: ${response.status} ${response.statusText}`,
      )
    }

    const result = (await response.json()) as GetAllProjectsResponse

    const nodes = result.data?.labs?.nodes ?? []
    allNodes.push(...nodes)

    const pageInfo = result.data?.labs?.pageInfo
    if (pageInfo) {
      totalPages = pageInfo.totalPages
      expectedTotal ??= result.data?.labs?.totalCount
      console.log(
        `Fetched page ${pageInfo.currentPage} of ${pageInfo.totalPages} (${nodes.length} labs)`,
      )
    }

    page++
  }

  console.log(
    `Total labs fetched: ${allNodes.length}${expectedTotal != null ? ` (API totalCount: ${expectedTotal})` : ""}`,
  )

  // Skip labs that have no oclId.
  return allNodes
    .filter(
      (node): node is ProjectNode & { oclId: string } => node.oclId != null,
    )
    .map((node) => ({
      oclId: node.oclId.toLowerCase(),
      name: node.name ?? node.oclId.toLowerCase(),
    }))
}

const GET_PROJECT_QUERY = `
query GetProject($oclId: String!) {
  labWithDataRoomAndFiles(oclId: $oclId) {
    oclId
    shortname
    dataRoom {
      id
      alias
      files {
        did
        path
        version
        contentType
        accessLevel
        description
        tags
        categories
        downloadUrl
      }
    }
  }
}
`

export async function getProjectDataRoomFiles(
  oclId: string,
): Promise<DataRoomFile[]> {
  const apiKey = process.env.MOLECULE_API_KEY

  if (!apiKey) {
    throw new Error("MOLECULE_API_KEY environment variable is not set")
  }

  const response = await fetch(MOLECULE_GRAPHQL_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
    },
    body: JSON.stringify({
      query: GET_PROJECT_QUERY,
      variables: { oclId },
    }),
  })

  if (!response.ok) {
    throw new Error(
      `Failed to fetch project data: ${response.status} ${response.statusText}`,
    )
  }

  const result = (await response.json()) as GetProjectResponse

  const project = result.data?.labWithDataRoomAndFiles

  if (!project) {
    console.log(`No lab found for oclId: ${oclId}`)
    return []
  }

  return project.dataRoom?.files ?? []
}

export function getPublicExtractableFiles(
  files: DataRoomFile[],
): DataRoomFile[] {
  return filterExtractableFiles(filterPublicFiles(files))
}

export async function getDataRoomHash(oclId: string): Promise<string> {
  const files = await getProjectDataRoomFiles(oclId)
  const publicFiles = filterPublicFiles(files)

  const fileHashes = await Promise.all(
    publicFiles.map((file) =>
      hashString(`${file.did}:${file.path}:${file.version}`),
    ),
  )

  const hash = await hashString(fileHashes.sort().join(""))
  return `${PROCESS_VERSION}:${hash}`
}
