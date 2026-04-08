/** Model used for document content extraction (cheaper/faster). */
export const EXTRACTION_MODEL = "claude-sonnet-4-5-20250929"

/** Model used for questionnaire, TRL classification, scoring, and todo generation. */
export const REASONING_MODEL = "claude-opus-4-5"

export const MOLECULE_GRAPHQL_ENDPOINT =
  "https://production.graphql.api.molecule.xyz/graphql"

/** The IP-NFT smart contract address used to construct project identifiers. */
export const IPNFT_CONTRACT_ADDRESS =
  "0xcaD88677CA87a7815728C72D74B4ff4982d54Fc1"

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

export interface TestProject {
  documentId: string
  symbol: string
}

/**
 * Hardcoded project list for local development.
 * Set USE_ALL_PROJECTS=true to fetch real projects from the Molecule API instead.
 */
export const TEST_PROJECTS: TestProject[] = [
  { documentId: "2", symbol: "VITA-FAST" },
  { documentId: "28", symbol: "VITARNA" },
  { documentId: "122", symbol: "AB4AD" },
  { documentId: "150", symbol: "VITAFOXO" },
  {
    documentId:
      "108494037067113761580099112583860151730516105403483528465874625006707409835912",
    symbol: "FIDEL",
  },
  { documentId: "29", symbol: "ApoptoSENS" },
  { documentId: "220", symbol: "DONNA" },
  { documentId: "129", symbol: "CLAW" },
  { documentId: "139", symbol: "HEMPY" },
]
