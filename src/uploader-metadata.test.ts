import assert from "node:assert/strict"
import { test } from "node:test"
import {
  type ExtractedDocument,
  formatExtractedDocumentsAsText,
  formatUploaderMetadata,
} from "../lib/llm.js"
import type { DataRoomFile } from "../lib/molecule.js"

const file = (overrides: Partial<DataRoomFile> = {}): DataRoomFile => ({
  did: "did:x",
  path: "study.pdf",
  version: 1,
  contentType: "application/pdf",
  accessLevel: "PUBLIC",
  description: null,
  tags: [],
  categories: [],
  downloadUrl: "https://example.com/study.pdf",
  createdAt: "2025-01-01T00:00:00Z",
  ...overrides,
})

const doc: ExtractedDocument = {
  document_type: "experiment_report",
  relevance: "high",
  summary: "Mouse study.",
  content_blocks: [],
  key_findings: [],
}

test("formatUploaderMetadata returns null without metadata", () => {
  assert.equal(formatUploaderMetadata(file()), null)
})

test("formatUploaderMetadata ignores synthetic local-input category", () => {
  assert.equal(
    formatUploaderMetadata(file({ categories: ["local-input"] })),
    null,
  )
})

test("formatUploaderMetadata labels metadata as unverified", () => {
  const line = formatUploaderMetadata(
    file({
      categories: ["science"],
      tags: ["In vivo data", "Validation"],
      description: 'Shows "tumor"\nregression',
    }),
  )
  assert.equal(
    line,
    "Uploader metadata (self-reported, unverified hint — not evidence): category=science; tags=In vivo data, Validation; description=\"Shows 'tumor' regression\"",
  )
})

test("formatUploaderMetadata truncates long descriptions", () => {
  const line = formatUploaderMetadata(file({ description: "a".repeat(900) }))
  assert.ok(line?.endsWith('a…"'))
  assert.ok((line?.length ?? 0) < 600)
})

test("formatExtractedDocumentsAsText includes metadata for matching files", () => {
  const text = formatExtractedDocumentsAsText(new Map([["study.pdf", doc]]), [
    file({ categories: ["science"], tags: ["In vivo data"] }),
  ])
  assert.match(text, /Uploader metadata .*tags=In vivo data/)
})

test("formatExtractedDocumentsAsText works without files", () => {
  const text = formatExtractedDocumentsAsText(new Map([["study.pdf", doc]]))
  assert.doesNotMatch(text, /Uploader metadata/)
})
