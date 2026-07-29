import type { Dirent } from "node:fs"
import * as fs from "node:fs/promises"
import * as path from "node:path"
import type { DataRoomFile } from "./molecule.js"

/**
 * Base directory for manually-provided project files. Each project's files live
 * in a subfolder named after its (lowercased) OCL id, e.g.
 * `input/files/0x0101.../My Document.pdf`.
 */
const INPUT_BASE_DIR = path.join(process.cwd(), "input", "files")

/**
 * File extensions we can extract, mapped to the MIME type the extraction
 * pipeline expects. Mirrors the extractable content types in `molecule.ts`.
 */
const EXTRACTABLE_EXTENSIONS = new Map<string, string>([
  [".pdf", "application/pdf"],
  [".png", "image/png"],
  [".jpg", "image/jpeg"],
  [".jpeg", "image/jpeg"],
  [".gif", "image/gif"],
  [".webp", "image/webp"],
])

/**
 * Scans `input/files/<oclId>/` for additional project files to include in the
 * extraction and scoring process, returning them as synthetic `DataRoomFile`
 * objects (with `localPath` set) so they flow through the same pipeline as
 * files fetched from the Molecule API.
 *
 * Returns an empty array when no folder exists for the given OCL id.
 */
export async function getLocalInputFiles(
  oclId: string,
): Promise<DataRoomFile[]> {
  const projectDir = path.join(INPUT_BASE_DIR, oclId)

  let entries: Dirent[]
  try {
    entries = await fs.readdir(projectDir, { withFileTypes: true })
  } catch {
    // No local input folder for this project — nothing to add.
    return []
  }

  const files: DataRoomFile[] = []

  for (const entry of entries) {
    if (!entry.isFile()) continue
    // Skip hidden/system files like .DS_Store.
    if (entry.name.startsWith(".")) continue

    const ext = path.extname(entry.name).toLowerCase()
    const contentType = EXTRACTABLE_EXTENSIONS.get(ext)
    if (!contentType) {
      console.log(
        `Skipping unsupported local input file: ${entry.name} (in ${oclId})`,
      )
      continue
    }

    const fullPath = path.join(projectDir, entry.name)
    const stats = await fs.stat(fullPath)

    files.push({
      // Stable, unique identifier for cache keying.
      did: `local:${oclId}:${entry.name}`,
      path: `input/files/${oclId}/${entry.name}`,
      // Use the file's mtime so edits invalidate the extraction cache and the
      // dataroom change-detection hash.
      version: Math.floor(stats.mtimeMs),
      contentType,
      accessLevel: "PUBLIC",
      description: null,
      tags: [],
      categories: ["local-input"],
      downloadUrl: "",
      localPath: fullPath,
    })
  }

  return files
}
