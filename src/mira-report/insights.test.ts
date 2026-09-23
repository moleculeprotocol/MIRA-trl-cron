import assert from "node:assert/strict"
import { describe, it } from "node:test"
import { aggregate } from "./aggregate.js"
import {
  buildInsightsInput,
  type InsightsOutput,
  isVerbatim,
  redactPii,
  sanitizeInsights,
} from "./insights.js"
import { message, rawData } from "./test-helpers.js"

describe("buildInsightsInput", () => {
  it("includes every session when under budget", () => {
    const raw = rawData({
      messages: [
        message("a", "s1", "2026-09-10T00:00:00Z", "<q1>"),
        message("b", "s1", "2026-09-10T00:01:00Z", "<q2>"),
        message("c", "s2", "2026-09-11T00:00:00Z", "<q3>"),
      ],
    })
    const input = buildInsightsInput(raw.current.messages, aggregate(raw, 14))
    assert.equal(input.sampledSessions, 2)
    assert.equal(input.totalSessions, 2)
    assert.match(input.prompt, /Session 1:\n- <q1>\n- <q2>/)
    assert.doesNotMatch(input.prompt, /random sample/)
  })

  it("samples sessions deterministically when over budget", () => {
    const messages = Array.from({ length: 200 }, (_, i) =>
      message(
        `m${i}`,
        `s${i}`,
        `2026-09-10T00:00:${String(i % 60).padStart(2, "0")}Z`,
        "x".repeat(400),
      ),
    )
    const raw = rawData({ messages })
    const report = aggregate(raw, 14)
    const a = buildInsightsInput(messages, report, 10_000)
    const b = buildInsightsInput(messages, report, 10_000)
    assert.ok(a.sampledSessions < 200)
    assert.ok(a.sampledSessions > 0)
    assert.equal(a.totalSessions, 200)
    assert.match(
      a.prompt,
      new RegExp(`random sample of ${a.sampledSessions} of 200 sessions`),
    )
    assert.equal(a.prompt, b.prompt)
    assert.ok(a.prompt.length < 10_000 + 3_000) // budget + fixed instructions
  })

  it("truncates long messages", () => {
    const raw = rawData({
      messages: [message("a", "s1", "2026-09-10T00:00:00Z", "y".repeat(5000))],
    })
    const input = buildInsightsInput(raw.current.messages, aggregate(raw, 14))
    assert.ok(!input.prompt.includes("y".repeat(501)))
  })
})

describe("privacy filters", () => {
  it("redacts personal data patterns", () => {
    assert.equal(
      redactPii("mail me at jane.doe@example.com"),
      "mail me at [email]",
    )
    assert.equal(
      redactPii("my wallet 0x1234567890abcdef1234567890abcdef12345678"),
      "my wallet [address]",
    )
    assert.equal(redactPii("I am vitalik.eth"), "I am [ens]")
    assert.equal(redactPii("How do I buy IPTs?"), "How do I buy IPTs?")
  })

  it("detects verbatim copies", () => {
    const users = [
      "How can I buy the VITA token on Base with my wallet today?",
      "hi",
    ]
    assert.ok(isVerbatim("how can I buy the VITA token on base", users))
    assert.ok(isVerbatim("Hi!", users))
    assert.ok(
      !isVerbatim("What is the process to purchase a lab token?", users),
    )
  })

  it("drops unsafe example questions and caps list lengths", () => {
    const output: InsightsOutput = {
      themes: Array.from({ length: 7 }, (_, i) => ({
        title: `Theme ${i}`,
        description: "Contact admin@example.com",
        share: "~10%",
      })),
      unanswered: [],
      capabilityGaps: [{ capability: "needs wallet access", approxCount: 2 }],
      exampleQuestions: [
        "What is the process to purchase a lab token?",
        "Can you check the balance of 0xabcdef1234567890abcdef1234567890abcdef12?",
        "how can I buy the VITA token on base",
      ],
    }
    const result = sanitizeInsights(output, [
      "How can I buy the VITA token on Base with my wallet today?",
    ])
    assert.equal(result.output.themes.length, 5)
    assert.equal(result.output.themes[0].description, "Contact [email]")
    assert.deepEqual(result.output.exampleQuestions, [
      "What is the process to purchase a lab token?",
    ])
    assert.equal(result.droppedExamples, 2)
  })
})
