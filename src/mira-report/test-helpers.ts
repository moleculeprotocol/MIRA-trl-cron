/** Builders for synthetic RawData in tests. Text is placeholders only. */
import { ROOT_OBSERVATION_NAME, SCORE_NAMES } from "./constants.js"
import type {
  RawData,
  RawMessage,
  RawMetrics,
  RawObservationNode,
  RawScore,
} from "./types.js"
import { RAW_DATA_VERSION } from "./types.js"

export function message(
  id: string,
  sessionId: string | null,
  startTime: string,
  input: string | null = `<message ${id}>`,
  level = "DEFAULT",
): RawMessage {
  return { id, traceId: `trace-${id}`, sessionId, startTime, level, input }
}

let scoreSeq = 0

export function score(
  observationId: string,
  name: string,
  value: RawScore["value"],
  options: { comment?: string; timestamp?: string; executionId?: string } = {},
): RawScore {
  scoreSeq++
  return {
    id: `score-${scoreSeq}`,
    name,
    dataType:
      name === SCORE_NAMES.topic
        ? "CATEGORICAL"
        : typeof value === "boolean"
          ? "BOOLEAN"
          : "NUMERIC",
    value,
    comment: options.comment ?? null,
    observationId,
    traceId: `trace-${observationId}`,
    timestamp: options.timestamp ?? "2026-09-20T00:00:00.000Z",
    executionId: options.executionId ?? `exec-${observationId}-${name}`,
  }
}

export function metrics(overrides: Partial<RawMetrics> = {}): RawMetrics {
  return {
    messageCount: 0,
    errorCount: 0,
    latencyP50Ms: null,
    latencyP95Ms: null,
    costUsd: 0,
    judgeCostUsd: 0,
    toolCalls: {},
    ...overrides,
  }
}

export function rawData(parts: {
  messages: RawMessage[]
  scores?: RawScore[]
  observations?: RawObservationNode[]
  metrics?: Partial<RawMetrics>
  previousMessages?: RawMessage[]
  previousMetrics?: Partial<RawMetrics>
}): RawData {
  return {
    version: RAW_DATA_VERSION,
    fetchedAt: "2026-09-23T08:00:00.000Z",
    environment: "mira-v1-production",
    current: {
      window: {
        from: "2026-09-09T00:00:00.000Z",
        to: "2026-09-23T00:00:00.000Z",
      },
      metrics: metrics({
        messageCount: parts.messages.length,
        ...parts.metrics,
      }),
      messages: parts.messages,
      observations: parts.observations ?? [],
      scores: parts.scores ?? [],
    },
    previous: {
      window: {
        from: "2026-08-26T00:00:00.000Z",
        to: "2026-09-09T00:00:00.000Z",
      },
      metrics: metrics({
        messageCount: parts.previousMessages?.length ?? 0,
        ...parts.previousMetrics,
      }),
      messages: parts.previousMessages ?? [],
    },
  }
}

export function node(
  id: string,
  parentObservationId: string | null,
  type: string,
  name: string,
): RawObservationNode {
  return { id, parentObservationId, type, name }
}

export const ROOT = ROOT_OBSERVATION_NAME
