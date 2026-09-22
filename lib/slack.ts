import { MOLECULE_PROJECT_BASE_URL } from "./config.js"
import type { TrlAnalysis } from "./llm.js"

const SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"

// Block Kit section text objects are capped at 3000 characters. The rationale
// is unbounded LLM output, so it has to be truncated before it is sent.
const MAX_SECTION_TEXT = 3000

// chat.postMessage allows roughly 1 message per second per channel. A forced
// run notifies for every project at once, so 429s are expected, not exceptional.
const MAX_ATTEMPTS = 4

const TRL_LABELS: Record<string, string> = {
  "pre-trl-1": "Pre-TRL 1",
  "trl-1": "TRL 1",
  "trl-2": "TRL 2",
  "trl-3": "TRL 3",
  "trl-gt-3": "TRL > 3",
}

export interface NotifySlackParams {
  oclId: string
  name: string
  shortname: string | null
  trlAnalysis: TrlAnalysis
  publishImmediately: boolean
}

/** Escapes the three characters Slack treats as markup control characters. */
function escapeMrkdwn(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
}

function truncate(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function buildBlocks({
  name,
  shortname,
  trlAnalysis,
  studioUrl,
  status,
}: {
  name: string
  shortname: string | null
  trlAnalysis: TrlAnalysis
  studioUrl: string
  status: string
}) {
  const trlLabel =
    TRL_LABELS[trlAnalysis.trl_classification] ??
    String(trlAnalysis.trl_classification)

  const rationale = truncate(
    escapeMrkdwn(trlAnalysis.rationale),
    MAX_SECTION_TEXT - "*Rationale*\n".length,
  )

  const blocks: Record<string, unknown>[] = [
    {
      type: "section",
      text: {
        type: "mrkdwn",
        text: `*<${studioUrl}|${escapeMrkdwn(name)}>*\n${status}`,
      },
    },
    {
      type: "section",
      fields: [
        { type: "mrkdwn", text: `*TRL Value*\n${escapeMrkdwn(trlLabel)}` },
        {
          type: "mrkdwn",
          text: `*Confidence*\n${escapeMrkdwn(String(trlAnalysis.confidence))}`,
        },
      ],
    },
  ]

  if (rationale.length > 0) {
    blocks.push({
      type: "section",
      text: { type: "mrkdwn", text: `*Rationale*\n${rationale}` },
    })
  }

  if (shortname) {
    const projectUrl = `${MOLECULE_PROJECT_BASE_URL}/${shortname}`
    blocks.push({
      type: "context",
      elements: [
        { type: "mrkdwn", text: `<${projectUrl}|View project on Molecule>` },
      ],
    })
  }

  return blocks
}

interface SlackResponse {
  ok: boolean
  error?: string
  ts?: string
}

/**
 * POSTs to chat.postMessage, retrying on rate limits and transient server
 * errors. Returns the message `ts` on success, or null if the message could
 * not be delivered. Never throws: a failed notification must not fail the run.
 */
async function postMessage(
  token: string,
  body: Record<string, unknown>,
): Promise<string | null> {
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      const response = await fetch(SLACK_POST_MESSAGE_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json; charset=utf-8",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(body),
      })

      if (response.status === 429 || response.status >= 500) {
        if (attempt === MAX_ATTEMPTS) {
          console.error(
            `Slack notification failed after ${MAX_ATTEMPTS} attempts (HTTP ${response.status})`,
          )
          return null
        }
        const retryAfter = Number(response.headers.get("retry-after"))
        const delayMs = Number.isFinite(retryAfter)
          ? retryAfter * 1000
          : 1000 * 2 ** (attempt - 1)
        console.log(
          `Slack rate limited or unavailable (HTTP ${response.status}), retrying in ${delayMs}ms`,
        )
        await sleep(delayMs)
        continue
      }

      // Slack returns HTTP 200 with `ok: false` for application-level errors.
      const result = (await response.json()) as SlackResponse

      if (!result.ok) {
        console.error(
          `Slack notification failed: ${result.error ?? `HTTP ${response.status}`}`,
        )
        return null
      }

      return result.ts ?? null
    } catch (error) {
      if (attempt === MAX_ATTEMPTS) {
        console.error("Failed to send Slack notification:", error)
        return null
      }
      await sleep(1000 * 2 ** (attempt - 1))
    }
  }

  return null
}

export async function notifySlack({
  oclId,
  name,
  shortname,
  trlAnalysis,
  publishImmediately,
}: NotifySlackParams): Promise<void> {
  if (process.env.ENVIRONMENT !== "production") {
    console.log("Skipping Slack notification (not production)")
    return
  }

  const token = process.env.SLACK_BOT_TOKEN
  const channel = process.env.SLACK_CHANNEL_ID

  if (!token) {
    console.log("SLACK_BOT_TOKEN not set, skipping notification")
    return
  }

  if (!channel) {
    console.log("SLACK_CHANNEL_ID not set, skipping notification")
    return
  }

  if (!process.env.SANITY_STUDIO_URL) {
    console.log("SANITY_STUDIO_URL not set, skipping notification")
    return
  }

  if (!shortname) {
    console.log(
      `No shortname for ${oclId}, omitting project link from notification`,
    )
  }

  const studioUrl = `${process.env.SANITY_STUDIO_URL}/structure/onChainLabs;onChainLab;${oclId}`

  // Slack already attributes the message to the MIRA app, so the status line
  // does not repeat it.
  const status = publishImmediately
    ? "✅ TRL published"
    : "📝 TRL draft ready for review"

  await postMessage(token, {
    channel,
    // No top-level `text`: alongside `attachments` it renders as an extra body
    // line rather than acting as a silent fallback. `fallback` covers push
    // notifications and screen readers instead.
    attachments: [
      {
        color: "#00ff00",
        fallback: `${status}: ${name}`,
        blocks: buildBlocks({
          name,
          shortname,
          trlAnalysis,
          studioUrl,
          status,
        }),
      },
    ],
  })
}
