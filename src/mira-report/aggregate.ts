/**
 * Pure aggregation: RawData -> ReportData. No network, no clock.
 */
import {
  ANSWER_RATE_EXCLUDED_TOPICS,
  OFF_TOPIC,
  ROOT_OBSERVATION_NAME,
  SCORE_NAMES,
  TOPIC_CATEGORIES,
  WEB_SEARCH_TOOL,
} from "./constants.js"
import type {
  RawData,
  RawMessage,
  RawObservationNode,
  RawPeriod,
  RawScore,
  Window,
} from "./types.js"

export interface Delta {
  current: number | null
  previous: number | null
}

export interface Rate {
  numerator: number
  denominator: number
  /** numerator / denominator, or null when the denominator is 0. */
  rate: number | null
}

export interface ReportData {
  environment: string
  window: Window
  previousWindow: Window
  days: number
  headline: {
    conversations: Delta
    messages: Delta
    medianMessagesPerConversation: Delta
    conversationLengths: { one: number; twoToThree: number; fourPlus: number }
    errors: Delta
    errorRate: Delta
    latencyP50Ms: Delta
    latencyP95Ms: Delta
    costUsd: Delta
    judgeCostUsd: number
  }
  topics: {
    /** Messages with at least one topic label. */
    scoredMessages: number
    /** Messages with more than one topic label. */
    multiTopicMessages: number
    categories: { category: string; count: number; share: number | null }[]
    /** Most common topic sets per conversation. */
    conversationTopicSets: { topics: string[]; count: number }[]
  }
  quality: {
    /** answered=true / answered-scored messages, without off_topic and other. */
    answerRateOnTopic: Rate
    /** answered=true / all answered-scored messages. */
    answerRateAll: Rate
  }
  scope: {
    scored: number
    unrelated: number
    capabilityGaps: number
    /** out_of_scope=true but no topic label, so it can't be split. */
    unclassified: number
  }
  tools: {
    calls: { name: string; count: number }[]
    totalCalls: number
    messagesWithTool: Rate
    messagesWithWebSearch: Rate
  }
  coverage: { evaluator: string; scoreName: string; coverage: Rate }[]
  /** Judge reasoning, only for the LLM step. Never rendered verbatim. */
  comments: {
    unanswered: string[]
    capabilityGaps: string[]
  }
  warnings: string[]
}

export interface MessageScores {
  topics: string[] | null
  answered: boolean | null
  answeredComment: string | null
  outOfScope: boolean | null
  outOfScopeComment: string | null
}

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------

export function rate(numerator: number, denominator: number): Rate {
  return {
    numerator,
    denominator,
    rate: denominator === 0 ? null : numerator / denominator,
  }
}

export function median(values: number[]): number | null {
  if (values.length === 0) return null
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 1
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2
}

/** Messages without a sessionId count as a conversation of their own. */
function conversationKey(message: RawMessage): string {
  return message.sessionId ?? `no-session:${message.id}`
}

export function groupBySession(
  messages: RawMessage[],
): Map<string, RawMessage[]> {
  const sessions = new Map<string, RawMessage[]>()
  const sorted = [...messages].sort((a, b) =>
    a.startTime.localeCompare(b.startTime),
  )
  for (const message of sorted) {
    const key = conversationKey(message)
    const list = sessions.get(key)
    if (list) list.push(message)
    else sessions.set(key, [message])
  }
  return sessions
}

function sessionStats(period: RawPeriod) {
  const sizes = [...groupBySession(period.messages).values()].map(
    (list) => list.length,
  )
  return {
    conversations: sizes.length,
    median: median(sizes),
    lengths: {
      one: sizes.filter((n) => n === 1).length,
      twoToThree: sizes.filter((n) => n >= 2 && n <= 3).length,
      fourPlus: sizes.filter((n) => n >= 4).length,
    },
  }
}

function asBoolean(value: RawScore["value"]): boolean | null {
  if (typeof value === "boolean") return value
  if (typeof value === "number")
    return value === 1 ? true : value === 0 ? false : null
  if (value === "true" || value === "True") return true
  if (value === "false" || value === "False") return false
  return null
}

/**
 * A multi-match categorical judge can store several labels either as one row
 * per label or as one delimited value. Both are handled.
 */
export function splitCategories(value: RawScore["value"]): string[] {
  if (typeof value !== "string") return []
  const trimmed = value.trim()
  if (trimmed.startsWith("[")) {
    try {
      const parsed = JSON.parse(trimmed)
      if (Array.isArray(parsed)) {
        return parsed.filter((v): v is string => typeof v === "string")
      }
    } catch {
      // Fall through to delimiter split.
    }
  }
  return trimmed
    .split(/[,;|]/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
}

/**
 * Keeps only the latest judge run per (observation, score name). A re-run
 * (e.g. a new evaluator version) replaces the old result instead of adding to it.
 */
function latestRun(scores: RawScore[]): RawScore[] {
  if (scores.length <= 1) return scores
  const latest = scores.reduce((a, b) => (a.timestamp >= b.timestamp ? a : b))
  if (!latest.executionId) return [latest]
  return scores.filter((s) => s.executionId === latest.executionId)
}

/** Joins the scores to the messages on the observation id. */
export function joinScores(
  messages: RawMessage[],
  scores: RawScore[],
): Map<string, MessageScores> {
  const messageIds = new Set(messages.map((m) => m.id))
  const grouped = new Map<string, RawScore[]>()
  for (const score of scores) {
    if (!messageIds.has(score.observationId)) continue
    const key = `${score.observationId}\u0000${score.name}`
    const list = grouped.get(key)
    if (list) list.push(score)
    else grouped.set(key, [score])
  }

  const result = new Map<string, MessageScores>()
  for (const message of messages) {
    result.set(message.id, {
      topics: null,
      answered: null,
      answeredComment: null,
      outOfScope: null,
      outOfScopeComment: null,
    })
  }

  for (const [key, list] of grouped) {
    const [observationId, name] = key.split("\u0000")
    const entry = result.get(observationId) as MessageScores
    const run = latestRun(list)
    if (name === SCORE_NAMES.topic) {
      const topics = new Set(run.flatMap((s) => splitCategories(s.value)))
      entry.topics = [...topics].sort()
    } else if (name === SCORE_NAMES.answered) {
      entry.answered = asBoolean(run[0].value)
      entry.answeredComment = run[0].comment
    } else if (name === SCORE_NAMES.outOfScope) {
      entry.outOfScope = asBoolean(run[0].value)
      entry.outOfScopeComment = run[0].comment
    }
  }
  return result
}

/** Maps each tool observation to the root message it belongs to. */
export function toolCallsByMessage(
  messages: RawMessage[],
  observations: RawObservationNode[],
): Map<string, string[]> {
  const messageIds = new Set(messages.map((m) => m.id))
  const byId = new Map(observations.map((o) => [o.id, o]))
  const result = new Map<string, string[]>()

  for (const node of observations) {
    if (node.type !== "TOOL") continue
    let parentId = node.parentObservationId
    const seen = new Set<string>()
    while (parentId && !messageIds.has(parentId) && !seen.has(parentId)) {
      seen.add(parentId)
      parentId = byId.get(parentId)?.parentObservationId ?? null
    }
    if (!parentId || !messageIds.has(parentId)) continue
    const list = result.get(parentId)
    if (list) list.push(node.name)
    else result.set(parentId, [node.name])
  }
  return result
}

// -----------------------------------------------------------------------------
// Aggregation
// -----------------------------------------------------------------------------

export function aggregate(raw: RawData, days: number): ReportData {
  const { current, previous } = raw
  const warnings: string[] = []

  const cur = sessionStats(current)
  const prev = sessionStats(previous)

  const messageCount = current.metrics.messageCount
  if (messageCount !== current.messages.length) {
    warnings.push(
      `Metrics API counts ${messageCount} messages, the observations list has ${current.messages.length}. Rates use the list.`,
    )
  }

  if (previous.metrics.messageCount === 0 && messageCount > 0) {
    warnings.push(
      `No ${ROOT_OBSERVATION_NAME} messages in the previous period, so the changes are not meaningful.`,
    )
  }

  const errorRate = (count: number, total: number) =>
    total === 0 ? null : count / total

  // --- Scores ----------------------------------------------------------------
  const joined = joinScores(current.messages, current.scores)
  const entries = [...joined.values()]
  const total = current.messages.length

  const unmatched = current.scores.filter(
    (s) => !joined.has(s.observationId),
  ).length
  if (current.scores.length > 0 && unmatched === current.scores.length) {
    warnings.push(
      "None of the fetched scores belong to a message in the window. Check the score names and the root observation name.",
    )
  }

  const known = new Set<string>(TOPIC_CATEGORIES)
  const unknownTopics = new Set<string>()

  const topicCounts = new Map<string, number>()
  let topicScored = 0
  let multiTopic = 0
  for (const entry of entries) {
    if (!entry.topics || entry.topics.length === 0) continue
    topicScored++
    if (entry.topics.length > 1) multiTopic++
    for (const topic of entry.topics) {
      if (!known.has(topic)) unknownTopics.add(topic)
      topicCounts.set(topic, (topicCounts.get(topic) ?? 0) + 1)
    }
  }
  if (unknownTopics.size > 0) {
    warnings.push(
      `Unknown topic labels from the judge: ${[...unknownTopics].join(", ")}`,
    )
  }

  const categories = [...new Set([...TOPIC_CATEGORIES, ...topicCounts.keys()])]
    .map((category) => {
      const count = topicCounts.get(category) ?? 0
      return {
        category,
        count,
        share: topicScored === 0 ? null : count / topicScored,
      }
    })
    .sort((a, b) => b.count - a.count || a.category.localeCompare(b.category))

  // Topic sets per conversation.
  const setCounts = new Map<string, number>()
  for (const list of groupBySession(current.messages).values()) {
    const topics = new Set<string>()
    for (const message of list) {
      for (const t of joined.get(message.id)?.topics ?? []) topics.add(t)
    }
    if (topics.size === 0) continue
    const key = [...topics].sort().join(" + ")
    setCounts.set(key, (setCounts.get(key) ?? 0) + 1)
  }
  const conversationTopicSets = [...setCounts.entries()]
    .map(([key, count]) => ({ topics: key.split(" + "), count }))
    .sort(
      (a, b) =>
        b.count - a.count || a.topics.join().localeCompare(b.topics.join()),
    )
    .slice(0, 5)

  // Answer rate.
  const answeredScored = entries.filter((e) => e.answered !== null)
  const onTopic = answeredScored.filter(
    (e) =>
      e.topics !== null &&
      e.topics.length > 0 &&
      !e.topics.some((t) => ANSWER_RATE_EXCLUDED_TOPICS.includes(t)),
  )
  const answerRateAll = rate(
    answeredScored.filter((e) => e.answered).length,
    answeredScored.length,
  )
  const answerRateOnTopic = rate(
    onTopic.filter((e) => e.answered).length,
    onTopic.length,
  )

  const unansweredComments = answeredScored
    .filter((e) => e.answered === false && e.answeredComment)
    .map((e) => e.answeredComment as string)

  // Scope.
  const scopeScored = entries.filter((e) => e.outOfScope !== null)
  const outOfScope = scopeScored.filter((e) => e.outOfScope === true)
  const unrelated = outOfScope.filter((e) => e.topics?.includes(OFF_TOPIC))
  const gaps = outOfScope.filter(
    (e) => e.topics && e.topics.length > 0 && !e.topics.includes(OFF_TOPIC),
  )
  const unclassified = outOfScope.filter(
    (e) => !e.topics || e.topics.length === 0,
  )

  // Coverage.
  const coverage = [
    {
      evaluator: "topic",
      scoreName: SCORE_NAMES.topic,
      coverage: rate(topicScored, total),
    },
    {
      evaluator: "answered",
      scoreName: SCORE_NAMES.answered,
      coverage: rate(answeredScored.length, total),
    },
    {
      evaluator: "out_of_scope",
      scoreName: SCORE_NAMES.outOfScope,
      coverage: rate(scopeScored.length, total),
    },
  ]

  // --- Tools -----------------------------------------------------------------
  const calls = Object.entries(current.metrics.toolCalls)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name))
  const byMessage = toolCallsByMessage(current.messages, current.observations)
  const withWebSearch = [...byMessage.values()].filter((names) =>
    names.includes(WEB_SEARCH_TOOL),
  ).length

  if (total > 0 && current.observations.length > 0) {
    const hasRoot = current.observations.some(
      (o) => o.name === ROOT_OBSERVATION_NAME,
    )
    if (!hasRoot) {
      warnings.push(
        "The observation tree has no root messages. Tool usage per message is unreliable.",
      )
    }
  }

  return {
    environment: raw.environment,
    window: current.window,
    previousWindow: previous.window,
    days,
    headline: {
      conversations: {
        current: cur.conversations,
        previous: prev.conversations,
      },
      messages: {
        current: messageCount,
        previous: previous.metrics.messageCount,
      },
      medianMessagesPerConversation: {
        current: cur.median,
        previous: prev.median,
      },
      conversationLengths: cur.lengths,
      errors: {
        current: current.metrics.errorCount,
        previous: previous.metrics.errorCount,
      },
      errorRate: {
        current: errorRate(current.metrics.errorCount, messageCount),
        previous: errorRate(
          previous.metrics.errorCount,
          previous.metrics.messageCount,
        ),
      },
      latencyP50Ms: {
        current: current.metrics.latencyP50Ms,
        previous: previous.metrics.latencyP50Ms,
      },
      latencyP95Ms: {
        current: current.metrics.latencyP95Ms,
        previous: previous.metrics.latencyP95Ms,
      },
      costUsd: {
        current: current.metrics.costUsd,
        previous: previous.metrics.costUsd,
      },
      judgeCostUsd: current.metrics.judgeCostUsd,
    },
    topics: {
      scoredMessages: topicScored,
      multiTopicMessages: multiTopic,
      categories,
      conversationTopicSets,
    },
    quality: { answerRateOnTopic, answerRateAll },
    scope: {
      scored: scopeScored.length,
      unrelated: unrelated.length,
      capabilityGaps: gaps.length,
      unclassified: unclassified.length,
    },
    tools: {
      calls,
      totalCalls: calls.reduce((sum, c) => sum + c.count, 0),
      messagesWithTool: rate(byMessage.size, total),
      messagesWithWebSearch: rate(withWebSearch, total),
    },
    coverage,
    comments: {
      unanswered: unansweredComments,
      capabilityGaps: gaps
        .map((e) => e.outOfScopeComment)
        .filter((c): c is string => Boolean(c)),
    },
    warnings,
  }
}
