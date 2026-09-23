import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { describe, it } from "node:test"
import { aggregate } from "./aggregate.js"
import type { Insights } from "./insights.js"
import { renderConsole } from "./render-console.js"
import { MAX_BLOCKS, MAX_SECTION_TEXT, renderSlack } from "./render-slack.js"
import type { RawData } from "./types.js"

const raw = JSON.parse(
  readFileSync(
    new URL("./fixtures/raw-2026-09-09.json", import.meta.url),
    "utf8",
  ),
) as RawData
const report = aggregate(raw, 14)

const insights: Insights = {
  themes: [
    {
      title: "Protocol basics",
      description: "How Molecule works",
      share: "~40%",
    },
    {
      title: "Lab discovery",
      description: "Finding <labs> & tokens",
      share: "~30%",
    },
  ],
  unanswered: [{ summary: "Details for a named lab", approxCount: 4 }],
  capabilityGaps: [
    { capability: "Launch a lab on the user's behalf", approxCount: 1 },
  ],
  exampleQuestions: ["How do I start a lab?"],
  model: "test-model",
  sampledSessions: 10,
  totalSessions: 10,
  droppedExamples: 0,
}

function allText(value: unknown): string {
  return JSON.stringify(value)
}

describe("renderConsole", () => {
  const text = renderConsole(report, insights, { llmSkipped: false })

  it("renders every section", () => {
    for (const heading of [
      "# MIRA chat report: 2026-09-09 – 2026-09-22 (14 days, UTC)",
      "## Headline numbers",
      "## Topics",
      "## Answer quality",
      "## Scope",
      "## Themes",
      "## Tool usage",
    ]) {
      assert.ok(text.includes(heading), `missing ${heading}`)
    }
  })

  it("has no footer", () => {
    assert.doesNotMatch(
      text,
      /Footer|Evaluator coverage|Production traffic only/,
    )
  })

  it("shows the key numbers", () => {
    assert.match(text, /Conversations\s+10\s+\+10 \(new\)/)
    assert.match(text, /Messages\s+26/)
    assert.match(text, /Answer rate \(on-topic\)\s+71% \(12\/17\)/)
    assert.match(text, /can add up to more than 100%/)
    assert.match(text, /docs-searchDocumentation\s+8/)
    assert.match(text, /Based on all 10 sessions\. Model: test-model\./)
  })

  it("never prints user messages", () => {
    assert.doesNotMatch(text, /<user message/)
    assert.doesNotMatch(text, /<judge comment/)
  })

  it("says when the LLM step was skipped", () => {
    assert.match(
      renderConsole(report, null, { llmSkipped: true }),
      /skipped with --no-llm/,
    )
  })
})

describe("renderSlack", () => {
  it("fits into one message within Block Kit limits", () => {
    const payload = renderSlack(report, insights, { llmSkipped: false })
    assert.equal(payload.thread, null)
    assert.ok(payload.main.blocks.length <= MAX_BLOCKS)
    assert.equal(payload.main.blocks[0].type, "header")
    assert.match(
      payload.main.text,
      /MIRA chat report 2026-09-09 – 2026-09-22 · 10 conversations · 26 messages/,
    )
    for (const block of payload.main.blocks) {
      const text = (block.text as { text?: string } | undefined)?.text
      if (text) assert.ok(text.length <= MAX_SECTION_TEXT)
    }
  })

  it("escapes LLM text and never includes raw user messages or judge comments", () => {
    const payload = renderSlack(report, insights, { llmSkipped: false })
    const text = allText(payload)
    assert.ok(text.includes("Finding &lt;labs&gt; &amp; tokens"))
    assert.doesNotMatch(text, /<user message/)
    assert.doesNotMatch(text, /<judge comment/)
  })

  it("moves details into a thread when a section overflows", () => {
    const long: Insights = {
      ...insights,
      themes: Array.from({ length: 5 }, (_, i) => ({
        title: `Theme ${i}`,
        description: "z".repeat(900),
        share: "~20%",
      })),
    }
    const payload = renderSlack(report, long, { llmSkipped: false })
    assert.ok(payload.thread)
    assert.ok(payload.thread.blocks.length > 0)
    assert.ok(allText(payload.thread).includes("Theme 0"))
    assert.ok(!allText(payload.main).includes("Theme 0"))
    assert.ok(allText(payload.main).includes("Answer quality"))
  })
})
