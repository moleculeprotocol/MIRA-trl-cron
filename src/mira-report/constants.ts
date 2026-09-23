/**
 * Names that the report depends on in the Langfuse project. They were checked
 * against real data on 2026-09-23 (see README, "MIRA chat report"). If one is
 * renamed in Langfuse, change it here: there are deliberately no fallbacks.
 */

/** Root observation of every MIRA request. One observation = one user message. */
export const ROOT_OBSERVATION_NAME = "handle-chat-message"

/** Environment that holds the LLM-as-a-judge runs (used for judge cost only). */
export const JUDGE_ENVIRONMENT = "langfuse-llm-as-a-judge"

/** Score names as configured for the three evaluators in the Langfuse UI. */
export const SCORE_NAMES = {
  topic: "Classify Input Topic",
  answered: "answered",
  outOfScope: "Detect Out-of-Scope Request",
} as const

export const TOPIC_CATEGORIES = [
  "protocol_docs",
  "lab_discovery",
  "market_data",
  "project_updates",
  "how_to_buy",
  "desci_ecosystem",
  "off_topic",
  "other",
] as const

/** Topics that are left out of the "on-topic" answer rate. */
export const ANSWER_RATE_EXCLUDED_TOPICS: readonly string[] = [
  "off_topic",
  "other",
]

export const OFF_TOPIC = "off_topic"

/** Tool name that the MIRA system prompt describes as a "last resort". */
export const WEB_SEARCH_TOOL = "search-web"
