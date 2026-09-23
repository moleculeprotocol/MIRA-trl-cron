/**
 * Langfuse Public API client for the report. Endpoints used (checked against
 * https://api.reference.langfuse.com on 2026-09-23):
 *
 * - GET /api/public/v2/metrics       aggregates (counts, latency, cost, tools)
 * - GET /api/public/v2/observations  raw root messages and the observation tree
 * - GET /api/public/v3/scores        observation-level judge scores
 *
 * /api/public/v2/scores is deprecated on Langfuse Cloud (removed 2026-11-16),
 * so scores are read from v3.
 */
import type { LangfuseConfig } from "./config.js"
import {
  JUDGE_ENVIRONMENT,
  ROOT_OBSERVATION_NAME,
  SCORE_NAMES,
} from "./constants.js"
import type {
  RawCurrentPeriod,
  RawData,
  RawMessage,
  RawMetrics,
  RawObservationNode,
  RawPeriod,
  RawScore,
  Window,
} from "./types.js"
import { RAW_DATA_VERSION } from "./types.js"
import type { ReportWindows } from "./window.js"

const MAX_ATTEMPTS = 4
const REQUEST_TIMEOUT_MS = 60_000
const OBSERVATIONS_PAGE_SIZE = 1000
const SCORES_PAGE_SIZE = 100
/** Safety stop against a cursor that never ends. */
const MAX_PAGES = 1000
/**
 * Child observations (tool calls) start after their root. Extending the tree
 * fetch a little past the window end keeps tool calls of messages that started
 * just before the boundary.
 */
const TREE_OVERLAP_MS = 60 * 60 * 1000

type Params = Record<string, string | number | boolean | undefined>

type MetricsFilter = {
  column: string
  operator: string
  value: unknown
  type: string
}

class NonRetryableError extends Error {}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export class LangfuseClient {
  private readonly authHeader: string

  constructor(private readonly config: LangfuseConfig) {
    this.authHeader = `Basic ${Buffer.from(
      `${config.publicKey}:${config.secretKey}`,
    ).toString("base64")}`
  }

  /** GET with retries on 429, 5xx and network errors. Throws on anything else. */
  private async get<T>(path: string, params: Params): Promise<T> {
    const url = new URL(`${this.config.baseUrl}${path}`)
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined) url.searchParams.set(key, String(value))
    }

    let lastError: unknown
    for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
      try {
        const response = await fetch(url, {
          headers: { Authorization: this.authHeader },
          signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
        })
        if (response.ok) return (await response.json()) as T

        const body = await response.text()
        const message = `Langfuse ${path} failed: HTTP ${response.status} ${body.slice(0, 500)}`
        // Client errors (4xx except 429) are final.
        if (response.status !== 429 && response.status < 500) {
          throw new NonRetryableError(message)
        }
        lastError = new Error(message)
        const retryAfter = Number(response.headers.get("retry-after"))
        if (attempt < MAX_ATTEMPTS) {
          await sleep(
            Number.isFinite(retryAfter) && retryAfter > 0
              ? retryAfter * 1000
              : 1000 * 2 ** attempt,
          )
        }
      } catch (error) {
        if (error instanceof NonRetryableError) throw error
        lastError = error
        if (attempt < MAX_ATTEMPTS) await sleep(1000 * 2 ** attempt)
      }
    }
    throw lastError
  }

  // ---------------------------------------------------------------------------
  // Metrics v2
  // ---------------------------------------------------------------------------

  private async metrics(query: {
    view: string
    metrics: { measure: string; aggregation: string }[]
    filters: MetricsFilter[]
    dimensions?: { field: string }[]
    window: Window
  }): Promise<Record<string, unknown>[]> {
    const body = {
      view: query.view,
      dimensions: query.dimensions ?? [],
      metrics: query.metrics,
      filters: query.filters,
      fromTimestamp: query.window.from,
      toTimestamp: query.window.to,
      config: { row_limit: 1000 },
    }
    const result = await this.get<{ data: Record<string, unknown>[] }>(
      "/api/public/v2/metrics",
      { query: JSON.stringify(body) },
    )
    return result.data
  }

  async fetchMetrics(window: Window): Promise<RawMetrics> {
    const env = eq("environment", this.config.environment)
    const root = [
      env,
      eq("name", ROOT_OBSERVATION_NAME),
      {
        column: "isRootObservation",
        operator: "=",
        value: true,
        type: "boolean",
      },
    ]

    const [messages, errors, cost, judgeCost, tools] = await Promise.all([
      this.metrics({
        view: "observations",
        metrics: [
          { measure: "count", aggregation: "count" },
          { measure: "latency", aggregation: "p50" },
          { measure: "latency", aggregation: "p95" },
        ],
        filters: root,
        window,
      }),
      this.metrics({
        view: "observations",
        metrics: [{ measure: "count", aggregation: "count" }],
        filters: [...root, eq("level", "ERROR")],
        window,
      }),
      // Cost sits on the generations. Spans can carry a copy of it.
      this.metrics({
        view: "observations",
        metrics: [{ measure: "totalCost", aggregation: "sum" }],
        filters: [env, eq("type", "GENERATION")],
        window,
      }),
      this.metrics({
        view: "observations",
        metrics: [{ measure: "totalCost", aggregation: "sum" }],
        filters: [
          eq("environment", JUDGE_ENVIRONMENT),
          eq("type", "GENERATION"),
        ],
        window,
      }),
      this.metrics({
        view: "observations",
        dimensions: [{ field: "name" }],
        metrics: [{ measure: "count", aggregation: "count" }],
        filters: [env, eq("type", "TOOL")],
        window,
      }),
    ])

    const toolCalls: Record<string, number> = {}
    for (const row of tools) {
      const count = num(row.count_count) ?? 0
      if (typeof row.name === "string" && count > 0) toolCalls[row.name] = count
    }

    return {
      messageCount: num(messages[0]?.count_count) ?? 0,
      errorCount: num(errors[0]?.count_count) ?? 0,
      latencyP50Ms: num(messages[0]?.p50_latency),
      latencyP95Ms: num(messages[0]?.p95_latency),
      costUsd: num(cost[0]?.sum_totalCost) ?? 0,
      judgeCostUsd: num(judgeCost[0]?.sum_totalCost) ?? 0,
      toolCalls,
    }
  }

  // ---------------------------------------------------------------------------
  // Observations v2 (cursor pagination)
  // ---------------------------------------------------------------------------

  private async listObservations(
    params: Params,
  ): Promise<Record<string, unknown>[]> {
    const rows: Record<string, unknown>[] = []
    let cursor: string | undefined
    for (let page = 0; page < MAX_PAGES; page++) {
      const result = await this.get<{
        data: Record<string, unknown>[]
        meta: { cursor?: string | null }
      }>("/api/public/v2/observations", {
        ...params,
        limit: OBSERVATIONS_PAGE_SIZE,
        cursor,
      })
      rows.push(...result.data)
      if (!result.meta.cursor || result.data.length === 0) return rows
      cursor = result.meta.cursor
    }
    throw new Error(`Observations pagination exceeded ${MAX_PAGES} pages`)
  }

  async fetchMessages(
    window: Window,
    options: { includeInput: boolean },
  ): Promise<RawMessage[]> {
    const rows = await this.listObservations({
      fields: options.includeInput ? "core,basic,io" : "core,basic",
      environment: this.config.environment,
      name: ROOT_OBSERVATION_NAME,
      isRootObservation: true,
      fromStartTime: window.from,
      toStartTime: window.to,
    })
    return rows.map((row) => ({
      id: str(row.id),
      traceId: str(row.traceId),
      sessionId:
        typeof row.sessionId === "string" && row.sessionId !== ""
          ? row.sessionId
          : null,
      startTime: str(row.startTime),
      level: str(row.level),
      input: options.includeInput ? ioText(row.input) : null,
    }))
  }

  async fetchObservationTree(window: Window): Promise<RawObservationNode[]> {
    const to = new Date(new Date(window.to).getTime() + TREE_OVERLAP_MS)
    const rows = await this.listObservations({
      fields: "core,basic",
      environment: this.config.environment,
      fromStartTime: window.from,
      toStartTime: to.toISOString(),
    })
    return rows.map((row) => ({
      id: str(row.id),
      parentObservationId:
        typeof row.parentObservationId === "string"
          ? row.parentObservationId
          : null,
      type: str(row.type),
      name: str(row.name),
    }))
  }

  // ---------------------------------------------------------------------------
  // Scores v3 (cursor pagination)
  // ---------------------------------------------------------------------------

  /**
   * Scores are timestamped when the judge ran, not when the message was sent,
   * and judges can be backfilled later. So this fetches every score from the
   * window start until now; aggregation keeps only the ones whose observation
   * is a message in the window.
   */
  async fetchScores(
    fromTimestamp: string,
    toTimestamp: string,
  ): Promise<RawScore[]> {
    const scores: RawScore[] = []
    let cursor: string | undefined
    for (let page = 0; page < MAX_PAGES; page++) {
      const result = await this.get<{
        data: Record<string, unknown>[]
        meta: { cursor?: string | null }
      }>("/api/public/v3/scores", {
        name: Object.values(SCORE_NAMES).join(","),
        environment: this.config.environment,
        fields: "details,subject",
        fromTimestamp,
        toTimestamp,
        limit: SCORES_PAGE_SIZE,
        cursor,
      })
      for (const row of result.data) {
        const score = toRawScore(row)
        if (score) scores.push(score)
      }
      if (!result.meta.cursor || result.data.length === 0) return scores
      cursor = result.meta.cursor
    }
    throw new Error(`Scores pagination exceeded ${MAX_PAGES} pages`)
  }
}

function eq(column: string, value: string): MetricsFilter {
  return { column, operator: "=", value, type: "string" }
}

/** Metrics v2 returns some numbers as strings (e.g. counts). */
function num(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null
  const n = Number(value)
  return Number.isFinite(n) ? n : null
}

function str(value: unknown): string {
  return typeof value === "string" ? value : String(value ?? "")
}

/** Observations v2 returns input/output as raw strings. Handle JSON strings too. */
function ioText(value: unknown): string | null {
  if (value === null || value === undefined) return null
  if (typeof value !== "string") return JSON.stringify(value)
  if (value.startsWith('"')) {
    try {
      const parsed = JSON.parse(value)
      if (typeof parsed === "string") return parsed
    } catch {
      // Not JSON, keep the raw string.
    }
  }
  return value
}

function toRawScore(row: Record<string, unknown>): RawScore | null {
  const subject = row.subject as
    | { kind?: string; id?: string; traceId?: string }
    | undefined
  // Only observation-level scores. Trace-level scores are deprecated.
  if (subject?.kind !== "observation" || !subject.id) return null
  const metadata = (row.metadata ?? {}) as Record<string, unknown>
  const value = row.value
  if (
    typeof value !== "string" &&
    typeof value !== "number" &&
    typeof value !== "boolean"
  ) {
    return null
  }
  return {
    id: str(row.id),
    name: str(row.name),
    dataType: str(row.dataType),
    value,
    comment: typeof row.comment === "string" ? row.comment : null,
    observationId: subject.id,
    traceId: subject.traceId ?? null,
    timestamp: str(row.timestamp),
    executionId:
      typeof metadata.job_execution_id === "string"
        ? metadata.job_execution_id
        : null,
  }
}

export async function fetchRawData(
  client: LangfuseClient,
  environment: string,
  windows: ReportWindows,
  now: Date,
): Promise<RawData> {
  const { current, previous } = windows

  const [
    currentMetrics,
    previousMetrics,
    currentMessages,
    previousMessages,
    observations,
    scores,
  ] = await Promise.all([
    client.fetchMetrics(current),
    client.fetchMetrics(previous),
    client.fetchMessages(current, { includeInput: true }),
    client.fetchMessages(previous, { includeInput: false }),
    client.fetchObservationTree(current),
    client.fetchScores(current.from, now.toISOString()),
  ])

  const currentPeriod: RawCurrentPeriod = {
    window: current,
    metrics: currentMetrics,
    messages: currentMessages,
    observations,
    scores,
  }
  const previousPeriod: RawPeriod = {
    window: previous,
    metrics: previousMetrics,
    messages: previousMessages,
  }

  return {
    version: RAW_DATA_VERSION,
    fetchedAt: now.toISOString(),
    environment,
    current: currentPeriod,
    previous: previousPeriod,
  }
}
