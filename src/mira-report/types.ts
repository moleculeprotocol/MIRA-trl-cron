/**
 * Shapes of the data fetched from Langfuse. This is also the format of the
 * `--dump` file, so `--from-dump` can re-run aggregation without the network.
 * Bump RAW_DATA_VERSION when the shape changes.
 */

export const RAW_DATA_VERSION = 1

export interface Window {
  /** Inclusive, ISO 8601 UTC. */
  from: string
  /** Exclusive, ISO 8601 UTC. */
  to: string
}

/** A root `handle-chat-message` observation, i.e. one user message. */
export interface RawMessage {
  id: string
  traceId: string
  sessionId: string | null
  startTime: string
  level: string
  /** Latest user message as plain text. Null for the previous period (not fetched). */
  input: string | null
}

/** Minimal observation node, used to map tool calls to their root message. */
export interface RawObservationNode {
  id: string
  parentObservationId: string | null
  type: string
  name: string
}

/** Observation-level score, flattened from the v3 scores API. */
export interface RawScore {
  id: string
  name: string
  dataType: string
  /** string for CATEGORICAL, boolean for BOOLEAN, number for NUMERIC. */
  value: string | number | boolean
  comment: string | null
  observationId: string
  traceId: string | null
  timestamp: string
  /** Groups the rows of one judge run (a multi-match categorical run writes several rows). */
  executionId: string | null
}

/** Aggregates read from GET /api/public/v2/metrics. */
export interface RawMetrics {
  messageCount: number
  errorCount: number
  latencyP50Ms: number | null
  latencyP95Ms: number | null
  costUsd: number
  judgeCostUsd: number
  toolCalls: Record<string, number>
}

export interface RawPeriod {
  window: Window
  metrics: RawMetrics
  messages: RawMessage[]
}

export interface RawCurrentPeriod extends RawPeriod {
  observations: RawObservationNode[]
  scores: RawScore[]
}

export interface RawData {
  version: typeof RAW_DATA_VERSION
  fetchedAt: string
  environment: string
  current: RawCurrentPeriod
  previous: RawPeriod
}
