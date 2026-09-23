import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { describe, it } from "node:test"
import {
  aggregate,
  joinScores,
  median,
  splitCategories,
  toolCallsByMessage,
} from "./aggregate.js"
import { SCORE_NAMES } from "./constants.js"
import { message, node, ROOT, rawData, score } from "./test-helpers.js"
import type { RawData } from "./types.js"

const T = SCORE_NAMES.topic
const A = SCORE_NAMES.answered
const O = SCORE_NAMES.outOfScope

function scenario(): RawData {
  const messages = [
    message("m1", "s1", "2026-09-10T10:00:00Z"),
    message("m2", "s1", "2026-09-10T10:01:00Z"),
    message("m3", "s1", "2026-09-10T10:02:00Z"),
    message("m4", "s1", "2026-09-10T10:03:00Z"),
    message("m5", "s2", "2026-09-11T10:00:00Z"),
    message("m6", "s3", "2026-09-12T10:00:00Z"),
    message("m7", "s3", "2026-09-12T10:01:00Z", "<message m7>", "ERROR"),
    message("m8", null, "2026-09-13T10:00:00Z"),
  ]
  const scores = [
    score("m1", T, "protocol_docs"),
    // Multi-match stored as one row per label, same judge run.
    score("m2", T, "lab_discovery", { executionId: "e2" }),
    score("m2", T, "market_data", { executionId: "e2" }),
    // Multi-match stored as one delimited value.
    score("m3", T, "how_to_buy, market_data"),
    score("m4", T, "off_topic"),
    score("m5", T, "other"),
    // Re-run: only the latest run counts.
    score("m6", T, "protocol_docs", {
      executionId: "old",
      timestamp: "2026-09-12T11:00:00Z",
    }),
    score("m6", T, "lab_discovery", {
      executionId: "new",
      timestamp: "2026-09-13T11:00:00Z",
    }),
    score("m8", T, "protocol_docs"),

    score("m1", A, true),
    score("m2", A, false, { comment: "unanswered-m2" }),
    score("m3", A, true),
    score("m4", A, false, { comment: "unanswered-m4" }),
    score("m5", A, true),
    score("m6", A, true, {
      executionId: "a-old",
      timestamp: "2026-09-12T11:00:00Z",
    }),
    score("m6", A, false, {
      comment: "unanswered-m6",
      executionId: "a-new",
      timestamp: "2026-09-13T11:00:00Z",
    }),
    score("m7", A, 1), // numeric 0/1 is accepted as boolean

    score("m1", O, false),
    score("m3", O, true, { comment: "gap-wallet" }),
    score("m4", O, true, { comment: "unrelated-weather" }),
    score("m7", O, true, { comment: "unclassified" }),

    // Score for an observation outside the window (e.g. a legacy root).
    score("t-legacy", T, "protocol_docs"),
  ]
  const observations = [
    node("m1", "external", "SPAN", ROOT),
    node("a1", "m1", "SPAN", "ai.streamText"),
    node("tool1", "a1", "TOOL", "docs-searchDocumentation"),
    node("tool2", "a1", "TOOL", "search-web"),
    node("m3", null, "SPAN", ROOT),
    node("tool3", "m3", "TOOL", "molecule-get-ipts"),
    node("tool4", "unknown-parent", "TOOL", "find-lab"),
  ]
  return rawData({
    messages,
    scores,
    observations,
    metrics: {
      errorCount: 1,
      latencyP50Ms: 4000,
      latencyP95Ms: 9000,
      costUsd: 1.5,
      judgeCostUsd: 0.25,
      toolCalls: {
        "docs-searchDocumentation": 1,
        "search-web": 1,
        "molecule-get-ipts": 1,
      },
    },
    previousMessages: [
      message("p1", "ps1", "2026-08-27T10:00:00Z", null),
      message("p2", "ps1", "2026-08-27T10:01:00Z", null),
      message("p3", "ps2", "2026-08-28T10:00:00Z", null),
      message("p4", "ps2", "2026-08-28T10:01:00Z", null),
    ],
    previousMetrics: { latencyP50Ms: 5000, latencyP95Ms: 8000, costUsd: 1 },
  })
}

describe("helpers", () => {
  it("median handles odd, even and empty lists", () => {
    assert.equal(median([3, 1, 2]), 2)
    assert.equal(median([4, 1, 2, 3]), 2.5)
    assert.equal(median([]), null)
  })

  it("splitCategories handles single, delimited and JSON array values", () => {
    assert.deepEqual(splitCategories("market_data"), ["market_data"])
    assert.deepEqual(splitCategories("a, b;c|d"), ["a", "b", "c", "d"])
    assert.deepEqual(splitCategories('["a","b"]'), ["a", "b"])
    assert.deepEqual(splitCategories(true), [])
  })

  it("joinScores keeps only the latest judge run", () => {
    const joined = joinScores(
      scenario().current.messages,
      scenario().current.scores,
    )
    assert.deepEqual(joined.get("m6")?.topics, ["lab_discovery"])
    assert.equal(joined.get("m6")?.answered, false)
    assert.deepEqual(joined.get("m2")?.topics, ["lab_discovery", "market_data"])
    assert.equal(joined.get("m7")?.answered, true)
    assert.equal(joined.has("t-legacy"), false)
  })

  it("toolCallsByMessage walks up to the root message", () => {
    const raw = scenario()
    const map = toolCallsByMessage(
      raw.current.messages,
      raw.current.observations,
    )
    assert.deepEqual(map.get("m1"), ["docs-searchDocumentation", "search-web"])
    assert.deepEqual(map.get("m3"), ["molecule-get-ipts"])
    assert.equal(map.size, 2)
  })
})

describe("aggregate (synthetic)", () => {
  const report = aggregate(scenario(), 14)

  it("computes headline numbers with changes", () => {
    const h = report.headline
    assert.deepEqual(h.conversations, { current: 4, previous: 2 })
    assert.deepEqual(h.messages, { current: 8, previous: 4 })
    assert.deepEqual(h.medianMessagesPerConversation, {
      current: 1.5,
      previous: 2,
    })
    assert.deepEqual(h.conversationLengths, {
      one: 2,
      twoToThree: 1,
      fourPlus: 1,
    })
    assert.deepEqual(h.errors, { current: 1, previous: 0 })
    assert.equal(h.errorRate.current, 1 / 8)
    assert.deepEqual(h.latencyP50Ms, { current: 4000, previous: 5000 })
    assert.deepEqual(h.costUsd, { current: 1.5, previous: 1 })
    assert.equal(h.judgeCostUsd, 0.25)
  })

  it("counts topics with multi-match", () => {
    assert.equal(report.topics.scoredMessages, 7)
    assert.equal(report.topics.multiTopicMessages, 2)
    const byName = Object.fromEntries(
      report.topics.categories.map((c) => [c.category, c.count]),
    )
    assert.deepEqual(byName, {
      protocol_docs: 2,
      lab_discovery: 2,
      market_data: 2,
      how_to_buy: 1,
      off_topic: 1,
      other: 1,
      project_updates: 0,
      desci_ecosystem: 0,
    })
    assert.equal(report.topics.categories[0].share, 2 / 7)
    assert.deepEqual(report.topics.conversationTopicSets[0].count, 1)
  })

  it("computes answer rates, excluding off_topic and other", () => {
    assert.deepEqual(report.quality.answerRateAll, {
      numerator: 4,
      denominator: 7,
      rate: 4 / 7,
    })
    assert.deepEqual(report.quality.answerRateOnTopic, {
      numerator: 2,
      denominator: 4,
      rate: 0.5,
    })
    assert.deepEqual(report.comments.unanswered, [
      "unanswered-m2",
      "unanswered-m4",
      "unanswered-m6",
    ])
  })

  it("splits out-of-scope into unrelated and capability gaps", () => {
    assert.deepEqual(report.scope, {
      scored: 4,
      unrelated: 1,
      capabilityGaps: 1,
      unclassified: 1,
    })
    assert.deepEqual(report.comments.capabilityGaps, ["gap-wallet"])
  })

  it("reports tools and evaluator coverage", () => {
    assert.equal(report.tools.totalCalls, 3)
    assert.equal(report.tools.calls[0].name, "docs-searchDocumentation")
    assert.deepEqual(report.tools.messagesWithTool, {
      numerator: 2,
      denominator: 8,
      rate: 0.25,
    })
    assert.deepEqual(report.tools.messagesWithWebSearch, {
      numerator: 1,
      denominator: 8,
      rate: 0.125,
    })
    assert.deepEqual(
      report.coverage.map((c) => [
        c.evaluator,
        c.coverage.numerator,
        c.coverage.denominator,
      ]),
      [
        ["topic", 7, 8],
        ["answered", 7, 8],
        ["out_of_scope", 4, 8],
      ],
    )
    assert.deepEqual(report.warnings, [])
  })

  it("warns when the metrics and list counts differ or there is no previous data", () => {
    const raw = scenario()
    raw.current.metrics.messageCount = 9
    raw.previous.metrics.messageCount = 0
    const r = aggregate(raw, 14)
    assert.equal(r.warnings.length, 2)
  })

  it("handles an empty window", () => {
    const r = aggregate(rawData({ messages: [] }), 14)
    assert.equal(r.headline.conversations.current, 0)
    assert.equal(r.headline.medianMessagesPerConversation.current, null)
    assert.equal(r.quality.answerRateAll.rate, null)
    assert.equal(r.headline.errorRate.current, null)
  })
})

describe("aggregate (anonymised dump, 2026-09-09 – 2026-09-22)", () => {
  // Real Langfuse data with all text replaced by placeholders. The expected
  // numbers were spot-checked against the Langfuse UI and metrics API.
  const raw = JSON.parse(
    readFileSync(
      new URL("./fixtures/raw-2026-09-09.json", import.meta.url),
      "utf8",
    ),
  ) as RawData
  const report = aggregate(raw, 14)

  it("matches the verified counts", () => {
    assert.equal(report.headline.messages.current, 26)
    assert.equal(report.headline.conversations.current, 10)
    assert.equal(report.topics.scoredMessages, 26)
    assert.deepEqual(report.quality.answerRateOnTopic, {
      numerator: 12,
      denominator: 17,
      rate: 12 / 17,
    })
    assert.equal(report.tools.totalCalls, 16)
    assert.equal(report.tools.messagesWithTool.numerator, 10)
    // Legacy roots in the previous period aren't handle-chat-message roots.
    assert.equal(report.headline.messages.previous, 0)
    assert.equal(report.warnings.length, 1)
  })
})
