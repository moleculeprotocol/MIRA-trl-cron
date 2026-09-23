/**
 * Pure: ReportData + Insights -> Slack Block Kit payload(s).
 * No raw user messages are ever rendered here, only aggregates and the
 * sanitised LLM output.
 */
import type { ReportData } from "./aggregate.js"
import {
  change,
  dateRange,
  pct,
  plain,
  rateText,
  seconds,
  topicLabel,
  usd,
} from "./format.js"
import type { Insights } from "./insights.js"
import { renderInsightsLines } from "./render-console.js"

export const MAX_BLOCKS = 50
export const MAX_SECTION_TEXT = 3000
const MAX_HEADER_TEXT = 150
const MAX_FIELD_TEXT = 2000
/** Context blocks allow at most 10 elements. */
const MAX_CONTEXT_ELEMENTS = 10

export type Block = Record<string, unknown>

export interface SlackMessage {
  text: string
  blocks: Block[]
}

export interface SlackPayload {
  main: SlackMessage
  /** Posted as a reply to `main` when everything doesn't fit in one message. */
  thread: SlackMessage | null
}

export function escapeMrkdwn(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
}

function truncate(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`
}

/** Records whether any section text had to be cut to fit. */
interface Overflow {
  hit: boolean
}

function sectionBlock(text: string, overflow?: Overflow): Block {
  if (overflow && text.length > MAX_SECTION_TEXT) overflow.hit = true
  return {
    type: "section",
    text: { type: "mrkdwn", text: truncate(text, MAX_SECTION_TEXT) },
  }
}

function fieldsBlock(fields: string[]): Block {
  return {
    type: "section",
    fields: fields.slice(0, 10).map((text) => ({
      type: "mrkdwn",
      text: truncate(text, MAX_FIELD_TEXT),
    })),
  }
}

function contextBlock(lines: string[]): Block {
  return {
    type: "context",
    elements: lines.slice(0, MAX_CONTEXT_ELEMENTS).map((text) => ({
      type: "mrkdwn",
      text: truncate(text, MAX_SECTION_TEXT),
    })),
  }
}

const divider: Block = { type: "divider" }

function bullets(lines: string[]): string {
  return lines.map((l) => `• ${escapeMrkdwn(l)}`).join("\n")
}

function field(label: string, value: string, delta?: string): string {
  return `*${label}*\n${value}${delta ? `  _${delta}_` : ""}`
}

export function renderSlack(
  report: ReportData,
  insights: Insights | null,
  options: { llmSkipped: boolean },
): SlackPayload {
  const h = report.headline
  const range = dateRange(report.window.from, report.window.to)
  const ins = renderInsightsLines(insights, options.llmSkipped)
  const overflow: Overflow = { hit: false }
  const section = (text: string) => sectionBlock(text, overflow)

  // --- Summary ---------------------------------------------------------------
  const summary: Block[] = [
    {
      type: "header",
      text: {
        type: "plain_text",
        text: truncate(`MIRA chat report · ${range}`, MAX_HEADER_TEXT),
      },
    },
    contextBlock([
      `${report.days} days (UTC) · \`${report.environment}\` · changes vs. ${dateRange(report.previousWindow.from, report.previousWindow.to)}`,
    ]),
    fieldsBlock([
      field(
        "Conversations",
        plain(h.conversations.current),
        change(h.conversations),
      ),
      field("Messages", plain(h.messages.current), change(h.messages)),
      field(
        "Messages / conversation",
        `median ${plain(h.medianMessagesPerConversation.current)}`,
        change(h.medianMessagesPerConversation),
      ),
      field(
        "Conversation length",
        `1: ${h.conversationLengths.one} · 2–3: ${h.conversationLengths.twoToThree} · 4+: ${h.conversationLengths.fourPlus}`,
      ),
      field(
        "Errors",
        `${plain(h.errors.current)} (${pct(h.errorRate.current, 1)})`,
        change(h.errors),
      ),
      field(
        "Latency p50 / p95",
        `${seconds(h.latencyP50Ms.current)} / ${seconds(h.latencyP95Ms.current)}`,
        `p95 ${change(h.latencyP95Ms, "ms")}`,
      ),
      field("Cost", usd(h.costUsd.current), change(h.costUsd, "usd")),
      field("Judge cost", usd(h.judgeCostUsd)),
    ]),
    divider,
  ]

  const topics = report.topics.categories.filter((c) => c.count > 0)
  summary.push(
    section(
      [
        `*Topics* (${report.topics.scoredMessages} classified messages; a message can have several topics, so shares can exceed 100%)`,
        topics.length > 0
          ? topics
              .map(
                (c) =>
                  `• ${topicLabel(c.category)}: ${c.count} (${pct(c.share)})`,
              )
              .join("\n")
          : "_No topic scores in this window._",
      ].join("\n"),
    ),
  )

  summary.push(
    section(
      [
        "*Answer quality*",
        `• On-topic answer rate: ${rateText(report.quality.answerRateOnTopic)}`,
        `• All scored messages: ${rateText(report.quality.answerRateAll)}`,
        "_On-topic leaves out messages classified as off topic or other._",
      ].join("\n"),
    ),
  )

  summary.push(
    section(
      [
        "*Scope*",
        `• Unrelated (off-topic) requests: ${report.scope.unrelated}`,
        `• Capability gaps: ${report.scope.capabilityGaps}`,
        ...(report.scope.unclassified > 0
          ? [`• Out of scope without topic: ${report.scope.unclassified}`]
          : []),
      ].join("\n"),
    ),
  )

  // --- Details ---------------------------------------------------------------
  const details: Block[] = []

  const themeLines: string[] = ["*Themes*"]
  if (ins.note) themeLines.push(`_${escapeMrkdwn(ins.note)}_`)
  if (ins.themes.length > 0) themeLines.push(bullets(ins.themes))
  details.push(section(themeLines.join("\n")))

  if (ins.unanswered.length > 0) {
    details.push(section(`*Unanswered needs*\n${bullets(ins.unanswered)}`))
  }
  if (ins.gaps.length > 0) {
    details.push(section(`*Missing capabilities*\n${bullets(ins.gaps)}`))
  }
  if (ins.examples.length > 0) {
    details.push(
      section(`*Example questions* (paraphrased)\n${bullets(ins.examples)}`),
    )
  }

  const tools = report.tools
  details.push(
    section(
      [
        "*Tool usage*",
        tools.calls.length > 0
          ? tools.calls.map((c) => `• \`${c.name}\`: ${c.count}`).join("\n")
          : "_No tool calls._",
        `Messages with ≥1 tool: ${rateText(tools.messagesWithTool)} · with \`search-web\`: ${rateText(tools.messagesWithWebSearch)}`,
      ].join("\n"),
    ),
  )

  const fallback = [
    `MIRA chat report ${range}`,
    `${plain(h.conversations.current)} conversations`,
    `${plain(h.messages.current)} messages`,
    `answer rate ${pct(report.quality.answerRateOnTopic.rate)}`,
    `cost ${usd(h.costUsd.current)}`,
  ].join(" · ")

  const single = [...summary, divider, ...details]
  if (single.length <= MAX_BLOCKS && !overflow.hit) {
    return { main: { text: fallback, blocks: single }, thread: null }
  }

  return {
    main: {
      text: fallback,
      blocks: [
        ...summary,
        contextBlock(["Themes, tool usage and details in the thread."]),
      ].slice(0, MAX_BLOCKS),
    },
    thread: {
      text: `Details for MIRA chat report ${range}`,
      blocks: details.slice(0, MAX_BLOCKS),
    },
  }
}
