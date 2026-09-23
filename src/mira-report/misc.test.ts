import assert from "node:assert/strict"
import { describe, it } from "node:test"
import {
  ConfigError,
  loadInsightsConfig,
  loadLangfuseConfig,
  loadSlackConfig,
} from "./config.js"
import { change, dateRange } from "./format.js"
import { computeWindows } from "./window.js"

describe("computeWindows", () => {
  it("defaults to the last N full UTC days", () => {
    const w = computeWindows({
      now: new Date("2026-09-23T08:15:00Z"),
      days: 14,
    })
    assert.deepEqual(w.current, {
      from: "2026-09-09T00:00:00.000Z",
      to: "2026-09-23T00:00:00.000Z",
    })
    assert.deepEqual(w.previous, {
      from: "2026-08-26T00:00:00.000Z",
      to: "2026-09-09T00:00:00.000Z",
    })
    assert.equal(w.days, 14)
  })

  it("accepts --from/--to overrides", () => {
    const w = computeWindows({
      now: new Date(),
      days: 14,
      from: "2026-09-01",
      to: "2026-09-08",
    })
    assert.equal(w.current.from, "2026-09-01T00:00:00.000Z")
    assert.equal(w.previous.from, "2026-08-25T00:00:00.000Z")
    assert.equal(w.days, 7)
  })

  it("uses --days back from --to", () => {
    const w = computeWindows({ now: new Date(), days: 7, to: "2026-09-08" })
    assert.equal(w.current.from, "2026-09-01T00:00:00.000Z")
  })

  it("rejects invalid or inverted windows", () => {
    assert.throws(() =>
      computeWindows({ now: new Date(), days: 14, from: "nope" }),
    )
    assert.throws(() =>
      computeWindows({
        now: new Date(),
        days: 14,
        from: "2026-09-10",
        to: "2026-09-01",
      }),
    )
  })
})

describe("format", () => {
  it("formats changes", () => {
    assert.equal(change({ current: 13, previous: 10 }), "+3 (+30%)")
    assert.equal(change({ current: 5, previous: 10 }), "-5 (-50%)")
    assert.equal(change({ current: 5, previous: 0 }), "+5 (new)")
    assert.equal(change({ current: 0, previous: 0 }), "±0")
    assert.equal(change({ current: null, previous: 3 }), "no comparison")
    assert.equal(
      change({ current: 4500, previous: 4000 }, "ms"),
      "+0.5s (+13%)",
    )
    assert.equal(change({ current: 0.5, previous: 0.4 }, "ratio"), "+10.0 pp")
  })

  it("shows the exclusive window end as the last included day", () => {
    assert.equal(
      dateRange("2026-09-09T00:00:00.000Z", "2026-09-23T00:00:00.000Z"),
      "2026-09-09 – 2026-09-22",
    )
  })
})

describe("config", () => {
  it("fails loudly and lists every missing variable", () => {
    assert.throws(
      () => loadLangfuseConfig({ LANGFUSE_PUBLIC_KEY: "pk" }),
      (error: unknown) =>
        error instanceof ConfigError &&
        error.message.includes("LANGFUSE_SECRET_KEY") &&
        error.message.includes("LANGFUSE_BASE_URL") &&
        error.message.includes("LANGFUSE_ENVIRONMENT"),
    )
    assert.throws(
      () => loadInsightsConfig({ ANTHROPIC_API_KEY: "k", INSIGHTS_MODEL: " " }),
      ConfigError,
    )
    assert.throws(() => loadSlackConfig({ SLACK_BOT_TOKEN: "x" }), ConfigError)
  })

  it("loads a complete config", () => {
    const config = loadLangfuseConfig({
      LANGFUSE_PUBLIC_KEY: "pk",
      LANGFUSE_SECRET_KEY: "sk",
      LANGFUSE_BASE_URL: "https://cloud.langfuse.com/",
      LANGFUSE_ENVIRONMENT: "mira-v1-production",
    })
    assert.equal(config.baseUrl, "https://cloud.langfuse.com")
    assert.equal(config.environment, "mira-v1-production")
  })
})
