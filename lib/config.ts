/** Model used for document content extraction (cheaper/faster). */
export const EXTRACTION_MODEL = "claude-sonnet-4-6"

/** Model used for questionnaire, TRL classification, scoring, and todo generation. */
export const REASONING_MODEL = "claude-opus-4-6"

/**
 * LLM sampling temperatures per task.
 */
export const TEMPERATURE = {
  extraction: 0,
  questionnaire: 0.2,
  classification: 0,
  scoring: 0,
  todos: 0.4,
  scheduleStatus: 0,
} as const

export const EXTRACTION_VERSION = 2

export const MOLECULE_GRAPHQL_ENDPOINT =
  "https://production.graphql.api.molecule.xyz/graphql"

// =============================================================================
// Sanity CMS
// =============================================================================

export const SANITY_PROJECT_ID = "pd4i2bbg"

export const SANITY_DATASETS = {
  staging: "staging-dataset",
  production: "production-dataset",
} as const

// =============================================================================
// Test Projects (local development only)
// =============================================================================

/**
 * Hardcoded list of oclIds for local development.
 * Set USE_ALL_PROJECTS=true to fetch real projects from the Molecule API instead.
 */
export const TEST_PROJECTS: string[] = [
  "0x01010000000000000000001f96a3babca74d62c672e2e7180a061e296bb5581b",
]
