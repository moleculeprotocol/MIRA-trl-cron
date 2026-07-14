import { IPNFT_CONTRACT_ADDRESS, MOLECULE_GRAPHQL_ENDPOINT } from "./config.js"

const EXTRACTABLE_CONTENT_TYPES = new Set([
  "application/pdf",
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp",
])

const PROCESS_VERSION = "v3"

const GET_ALL_PROJECTS_QUERY = `
query GetAllProjects($page: Int, $perPage: Int) {
  labs(page: $page, perPage: $perPage) {
    nodes {
      ipnftId
      ipnft {
        symbol
      }
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
  ipnftId: string | null
  ipnft: {
    symbol: string
  } | null
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
  tokenId: string
  symbol: string
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
  ipnftUid: string
  ipnftSymbol: string
  dataRoom: DataRoom | null
}

interface GetProjectResponse {
  data: {
    projectWithDataRoomAndFilesV2: ProjectWithDataRoom | null
  }
}

function buildIpnftUid(ipnftId: string): string {
  return `${IPNFT_CONTRACT_ADDRESS}_${ipnftId}`
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
  // The `labs` query is 0-indexed (defaults to 0); perPage max is 100.
  let currentPage = 0
  let hasNextPage = true

  while (hasNextPage) {
    const response = await fetch(MOLECULE_GRAPHQL_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
      },
      body: JSON.stringify({
        query: GET_ALL_PROJECTS_QUERY,
        variables: { page: currentPage, perPage: 100 },
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
    hasNextPage = pageInfo?.hasNextPage ?? false
    currentPage++

    if (pageInfo) {
      console.log(
        `Fetched page ${pageInfo.currentPage} of ${pageInfo.totalPages} (${nodes.length} labs)`,
      )
    }
  }

  console.log(`Total labs fetched: ${allNodes.length}`)

  // Only labs linked to a legacy IPNFT can flow through the IPNFT-keyed
  // pipeline (Sanity doc id + ipnftUid). Skip labs without a linked IPNFT.
  return allNodes
    .filter(
      (
        node,
      ): node is ProjectNode & { ipnftId: string; ipnft: { symbol: string } } =>
        node.ipnftId != null && node.ipnft != null,
    )
    .map((node) => ({
      tokenId: node.ipnftId,
      symbol: node.ipnft.symbol,
    }))
}

const GET_PROJECT_QUERY = `
query GetProject($ipnftUid: ID!) {
  projectWithDataRoomAndFilesV2(ipnftUid: $ipnftUid) {
    ipnftUid
    ipnftSymbol
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
  ipnftId: string,
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
      variables: { ipnftUid: buildIpnftUid(ipnftId) },
    }),
  })

  if (!response.ok) {
    throw new Error(
      `Failed to fetch project data: ${response.status} ${response.statusText}`,
    )
  }

  const result = (await response.json()) as GetProjectResponse

  const project = result.data?.projectWithDataRoomAndFilesV2

  if (!project) {
    console.log(`No project found for ipnftId: ${ipnftId}`)
    return []
  }

  return project.dataRoom?.files ?? []
}

export function getPublicExtractableFiles(
  files: DataRoomFile[],
): DataRoomFile[] {
  return filterExtractableFiles(filterPublicFiles(files))
}

export async function getDataRoomHash(ipnftId: string): Promise<string> {
  const files = await getProjectDataRoomFiles(ipnftId)
  const publicFiles = filterPublicFiles(files)

  const fileHashes = await Promise.all(
    publicFiles.map((file) =>
      hashString(`${file.did}:${file.path}:${file.version}`),
    ),
  )

  const hash = await hashString(fileHashes.sort().join(""))
  return `${PROCESS_VERSION}:${hash}`
}
