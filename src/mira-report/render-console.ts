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

function section(title: string, lines: string[]): string {
  return [`\n## ${title}`, ...lines].join("\n")
}

function row(label: string, value: string, delta?: string): string {
  const main = `  ${label.padEnd(28)} ${value}`
  return delta ? `${main.padEnd(52)} ${delta}` : main
}

export function renderInsightsLines(
  insights: Insights | null,
  skipped: boolean,
): {
  themes: string[]
  unanswered: string[]
  gaps: string[]
  examples: string[]
  note: string | null
} {
  if (!insights) {
    const note = skipped
      ? "(LLM step skipped with --no-llm)"
      : "(No user messages in the window)"
    return { themes: [], unanswered: [], gaps: [], examples: [], note }
  }
  const note =
    insights.sampledSessions < insights.totalSessions
      ? `Based on a sample of ${insights.sampledSessions} of ${insights.totalSessions} sessions.`
      : `Based on all ${insights.totalSessions} sessions.`
  return {
    themes: insights.themes.map(
      (t) => `${t.title} (${t.share}): ${t.description}`,
    ),
    unanswered: insights.unanswered.map(
      (u) => `${u.summary} (~${u.approxCount})`,
    ),
    gaps: insights.capabilityGaps.map(
      (g) => `${g.capability} (~${g.approxCount})`,
    ),
    examples: insights.exampleQuestions,
    note: `${note} Model: ${insights.model}.`,
  }
}

export function renderConsole(
  report: ReportData,
  insights: Insights | null,
  options: { llmSkipped: boolean },
): string {
  const h = report.headline
  const lengths = h.conversationLengths
  const out: string[] = []

  out.push(
    `# MIRA chat report: ${dateRange(report.window.from, report.window.to)} (${report.days} days, UTC)`,
  )
  out.push(
    `Environment: ${report.environment}. Changes compare with ${dateRange(report.previousWindow.from, report.previousWindow.to)}.`,
  )

  out.push(
    section("Headline numbers", [
      row(
        "Conversations",
        plain(h.conversations.current),
        change(h.conversations),
      ),
      row("Messages", plain(h.messages.current), change(h.messages)),
      row(
        "Messages per conversation",
        `median ${plain(h.medianMessagesPerConversation.current)}`,
        change(h.medianMessagesPerConversation),
      ),
      row(
        "  Conversation length",
        `1: ${lengths.one} · 2–3: ${lengths.twoToThree} · 4+: ${lengths.fourPlus}`,
      ),
      row(
        "Errors",
        `${plain(h.errors.current)} (${pct(h.errorRate.current, 1)})`,
        change(h.errors),
      ),
      row(
        "Latency p50",
        seconds(h.latencyP50Ms.current),
        change(h.latencyP50Ms, "ms"),
      ),
      row(
        "Latency p95",
        seconds(h.latencyP95Ms.current),
        change(h.latencyP95Ms, "ms"),
      ),
      row("Cost (MIRA)", usd(h.costUsd.current), change(h.costUsd, "usd")),
      row("Cost (judges, all envs)", usd(h.judgeCostUsd)),
    ]),
  )

  const t = report.topics
  out.push(
    section("Topics", [
      `  Share of ${t.scoredMessages} classified messages. A message can have several topics, so shares can add up to more than 100%.`,
      ...t.categories
        .filter((c) => c.count > 0)
        .map((c) =>
          row(topicLabel(c.category), `${c.count} (${pct(c.share)})`),
        ),
      ...(t.conversationTopicSets.length > 0
        ? [
            "  Top topic sets per conversation:",
            ...t.conversationTopicSets.map(
              (s) => `    ${s.count} × ${s.topics.map(topicLabel).join(" + ")}`,
            ),
          ]
        : []),
    ]),
  )

  const ins = renderInsightsLines(insights, options.llmSkipped)

  out.push(
    section("Answer quality", [
      row("Answer rate (on-topic)", rateText(report.quality.answerRateOnTopic)),
      row("Answer rate (all scored)", rateText(report.quality.answerRateAll)),
      "  On-topic leaves out messages classified as off_topic or other.",
      ...(ins.unanswered.length > 0
        ? ["  Unanswered needs:", ...ins.unanswered.map((l) => `    - ${l}`)]
        : []),
    ]),
  )

  const s = report.scope
  out.push(
    section("Scope", [
      row("Unrelated (off-topic)", String(s.unrelated)),
      row("Capability gaps", String(s.capabilityGaps)),
      ...(s.unclassified > 0
        ? [row("Out of scope, no topic", String(s.unclassified))]
        : []),
      ...(ins.gaps.length > 0
        ? ["  Missing capabilities:", ...ins.gaps.map((l) => `    - ${l}`)]
        : []),
    ]),
  )

  out.push(
    section("Themes", [
      ...(ins.note ? [`  ${ins.note}`] : []),
      ...ins.themes.map((l) => `  - ${l}`),
      ...(ins.examples.length > 0
        ? [
            "  Example questions (paraphrased):",
            ...ins.examples.map((q) => `    - ${q}`),
          ]
        : []),
    ]),
  )

  const tools = report.tools
  out.push(
    section("Tool usage", [
      ...tools.calls.map((c) => row(c.name, String(c.count))),
      ...(tools.calls.length === 0 ? ["  No tool calls."] : []),
      row("Messages with ≥1 tool", rateText(tools.messagesWithTool)),
      row("Messages with search-web", rateText(tools.messagesWithWebSearch)),
    ]),
  )

  return `${out.join("\n")}\n`
}
