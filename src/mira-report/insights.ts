/**
 * Section E: one LLM call that groups the window's user messages and judge
 * comments into themes. Input building and output sanitising are pure.
 */
import { createAnthropic } from "@ai-sdk/anthropic"
import { generateText, Output } from "ai"
import { z } from "zod"
import { groupBySession, type ReportData } from "./aggregate.js"
import type { InsightsConfig } from "./config.js"
import type { RawMessage } from "./types.js"

/** Hard cap on the characters of user content sent to the model. */
export const MAX_INPUT_CHARS = 60_000
/** Single messages are cut to this length. */
const MAX_MESSAGE_CHARS = 500
/** Share of the budget that judge comments may use. */
const COMMENT_BUDGET_SHARE = 0.25

export const insightsSchema = z.object({
  themes: z
    .array(
      z.object({
        title: z.string().describe("Short theme name, 2-5 words"),
        description: z.string().describe("One line describing the theme"),
        share: z.string().describe('Rough share of conversations, e.g. "~30%"'),
      }),
    )
    .describe("3-5 themes across the conversations"),
  unanswered: z
    .array(
      z.object({
        summary: z
          .string()
          .describe("What users wanted but did not get, one line"),
        approxCount: z.number().int().describe("Rough number of messages"),
      }),
    )
    .describe("Grouped notable unanswered questions"),
  capabilityGaps: z
    .array(
      z.object({
        capability: z
          .string()
          .describe('The missing capability, e.g. "needs wallet access"'),
        approxCount: z.number().int().describe("Rough number of messages"),
      }),
    )
    .describe("Grouped capabilities MIRA lacks"),
  exampleQuestions: z
    .array(z.string())
    .describe(
      "3-5 paraphrased, representative user questions. Never verbatim, no personal data.",
    ),
})

export type InsightsOutput = z.infer<typeof insightsSchema>

export interface Insights extends InsightsOutput {
  model: string
  sampledSessions: number
  totalSessions: number
  /** Example questions dropped by the verbatim/PII filter. */
  droppedExamples: number
}

export interface InsightsInput {
  prompt: string
  sampledSessions: number
  totalSessions: number
}

function truncate(text: string, max: number): string {
  const clean = text.replace(/\s+/g, " ").trim()
  return clean.length <= max ? clean : `${clean.slice(0, max - 1)}…`
}

/** Deterministic PRNG so a re-run on the same dump samples the same sessions. */
function seededRandom(seed: string): () => number {
  let h = 1779033703 ^ seed.length
  for (let i = 0; i < seed.length; i++) {
    h = Math.imul(h ^ seed.charCodeAt(i), 3432918353)
    h = (h << 13) | (h >>> 19)
  }
  return () => {
    h = Math.imul(h ^ (h >>> 16), 2246822507)
    h = Math.imul(h ^ (h >>> 13), 3266489909)
    h ^= h >>> 16
    return (h >>> 0) / 4294967296
  }
}

function shuffle<T>(items: T[], random: () => number): T[] {
  const copy = [...items]
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1))
    ;[copy[i], copy[j]] = [copy[j], copy[i]]
  }
  return copy
}

function takeWithinBudget(lines: string[], budget: number): string[] {
  const taken: string[] = []
  let used = 0
  for (const line of lines) {
    if (used + line.length + 1 > budget) break
    taken.push(line)
    used += line.length + 1
  }
  return taken
}

const INSTRUCTIONS = `You analyse two weeks of chat logs of MIRA, the AI assistant on Molecule's screener web app. Molecule is a decentralized science (DeSci) platform where research labs raise funding through IP-NFTs and tradeable lab tokens (IPTs). Users ask MIRA about the protocol, labs, token market data and how to buy tokens.

You get:
- <conversations>: user messages grouped by chat session (MIRA's replies are left out)
- <unanswered>: an LLM judge's reasoning for replies that did not give the user what they asked for
- <capability_gaps>: an LLM judge's reasoning for requests that need a capability MIRA does not have

Return:
- themes: 3-5 themes across the conversations, each with a one-line description and a rough share of conversations (e.g. "~30%").
- unanswered: group the unanswered reasoning into a few recurring needs, with a rough count each. Empty if there is none.
- capabilityGaps: group the capability-gap reasoning into missing capabilities (e.g. "needs wallet access", "wants to trade"), with a rough count each. Empty if there is none.
- exampleQuestions: 3-5 representative questions, PARAPHRASED in your own words.

Privacy rules (strict):
- Never quote users verbatim. Always rephrase.
- Never include wallet addresses, transaction hashes, emails, names of private persons, handles or any other personal data. Public lab and token names are fine.
- Write in English, even if users wrote in another language.`

export function buildInsightsInput(
  messages: RawMessage[],
  report: ReportData,
  maxChars: number = MAX_INPUT_CHARS,
): InsightsInput {
  const sessions = [...groupBySession(messages).values()]
  const commentBudget = Math.floor(maxChars * COMMENT_BUDGET_SHARE)

  const unanswered = takeWithinBudget(
    report.comments.unanswered.map(
      (c) => `- ${truncate(c, MAX_MESSAGE_CHARS)}`,
    ),
    commentBudget / 2,
  )
  const gaps = takeWithinBudget(
    report.comments.capabilityGaps.map(
      (c) => `- ${truncate(c, MAX_MESSAGE_CHARS)}`,
    ),
    commentBudget / 2,
  )

  const commentChars = [...unanswered, ...gaps].join("\n").length
  const conversationBudget = maxChars - commentChars

  const blocks = sessions.map((list, i) => {
    const lines = list
      .filter((m) => m.input && m.input.trim().length > 0)
      .map((m) => `- ${truncate(m.input as string, MAX_MESSAGE_CHARS)}`)
    return lines.length === 0 ? "" : `Session ${i + 1}:\n${lines.join("\n")}`
  })
  const nonEmpty = blocks.filter((b) => b.length > 0)

  const fullSize = nonEmpty.join("\n\n").length
  const chosen =
    fullSize <= conversationBudget
      ? nonEmpty
      : takeWithinBudget(
          shuffle(nonEmpty, seededRandom(report.window.from)),
          conversationBudget,
        )

  const sampleNote =
    chosen.length < nonEmpty.length
      ? `Note: this is a random sample of ${chosen.length} of ${nonEmpty.length} sessions. Estimate shares relative to the sample.\n\n`
      : ""

  const prompt = `${INSTRUCTIONS}

${sampleNote}<conversations>
${chosen.join("\n\n")}
</conversations>

<unanswered>
${unanswered.join("\n")}
</unanswered>

<capability_gaps>
${gaps.join("\n")}
</capability_gaps>`

  return {
    prompt,
    sampledSessions: chosen.length,
    totalSessions: nonEmpty.length,
  }
}

// -----------------------------------------------------------------------------
// Output sanitising
// -----------------------------------------------------------------------------

const PII_PATTERNS: [RegExp, string][] = [
  [/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi, "[email]"],
  [/\b0x[a-fA-F0-9]{8,}\b/g, "[address]"],
  [/\b[a-z0-9-]+\.eth\b/gi, "[ens]"],
  [/\b[1-9A-HJ-NP-Za-km-z]{32,44}\b/g, "[address]"],
  [/https?:\/\/\S+/gi, "[link]"],
  [/\+?\d[\d\s().-]{8,}\d/g, "[number]"],
]

export function redactPii(text: string): string {
  return PII_PATTERNS.reduce(
    (acc, [pattern, replacement]) => acc.replace(pattern, replacement),
    text,
  )
}

function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim()
}

/** Word n-grams, used to detect copied phrases. */
function ngrams(words: string[], n: number): Set<string> {
  const result = new Set<string>()
  for (let i = 0; i + n <= words.length; i++) {
    result.add(words.slice(i, i + n).join(" "))
  }
  return result
}

const VERBATIM_NGRAM = 6

/**
 * True if the candidate equals a user message, or shares a run of
 * VERBATIM_NGRAM words with one. Short messages ("What is Molecule?") are
 * compared for equality only, since any paraphrase would look similar.
 */
export function isVerbatim(candidate: string, userMessages: string[]): boolean {
  const c = normalize(candidate)
  if (c.length === 0) return false
  const cGrams = ngrams(c.split(" "), VERBATIM_NGRAM)
  for (const message of userMessages) {
    const m = normalize(message)
    if (m.length === 0) continue
    if (m === c) return true
    const words = m.split(" ")
    if (words.length < VERBATIM_NGRAM) continue
    for (const gram of ngrams(words, VERBATIM_NGRAM)) {
      if (cGrams.has(gram)) return true
    }
  }
  return false
}

export function sanitizeInsights(
  output: InsightsOutput,
  userMessages: string[],
): { output: InsightsOutput; droppedExamples: number } {
  const examples = output.exampleQuestions
    .map((q) => q.trim())
    .filter((q) => q.length > 0)
  const kept = examples.filter(
    (q) => !isVerbatim(q, userMessages) && redactPii(q) === q,
  )

  return {
    droppedExamples: examples.length - kept.length,
    output: {
      themes: output.themes.slice(0, 5).map((t) => ({
        title: redactPii(t.title),
        description: redactPii(t.description),
        share: t.share,
      })),
      unanswered: output.unanswered.slice(0, 6).map((u) => ({
        summary: redactPii(u.summary),
        approxCount: u.approxCount,
      })),
      capabilityGaps: output.capabilityGaps.slice(0, 6).map((g) => ({
        capability: redactPii(g.capability),
        approxCount: g.approxCount,
      })),
      exampleQuestions: kept.slice(0, 5),
    },
  }
}

// -----------------------------------------------------------------------------
// LLM call
// -----------------------------------------------------------------------------

export async function generateInsights(
  config: InsightsConfig,
  messages: RawMessage[],
  report: ReportData,
): Promise<Insights | null> {
  const input = buildInsightsInput(messages, report)
  if (input.totalSessions === 0) return null

  const anthropic = createAnthropic({ apiKey: config.anthropicApiKey })
  const { output } = await generateText({
    model: anthropic(config.model),
    temperature: 0.2,
    prompt: input.prompt,
    output: Output.object({ schema: insightsSchema }),
  })

  const userMessages = messages
    .map((m) => m.input)
    .filter((m): m is string => Boolean(m))
  const sanitized = sanitizeInsights(output, userMessages)

  return {
    ...sanitized.output,
    model: config.model,
    sampledSessions: input.sampledSessions,
    totalSessions: input.totalSessions,
    droppedExamples: sanitized.droppedExamples,
  }
}
