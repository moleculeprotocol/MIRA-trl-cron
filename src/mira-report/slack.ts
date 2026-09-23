/**
 * Posts the report with chat.postMessage. Unlike the TRL notifications
 * (lib/slack.ts), a failed post here fails the run: the report is the job.
 */
import type { SlackConfig } from "./config.js"
import type { SlackMessage, SlackPayload } from "./render-slack.js"

const SLACK_POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"
const MAX_ATTEMPTS = 4

interface SlackResponse {
  ok: boolean
  error?: string
  ts?: string
  response_metadata?: { messages?: string[] }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function postMessage(
  config: SlackConfig,
  message: SlackMessage,
  threadTs?: string,
): Promise<string> {
  const body = {
    channel: config.channelId,
    text: message.text,
    blocks: message.blocks,
    unfurl_links: false,
    unfurl_media: false,
    ...(threadTs ? { thread_ts: threadTs } : {}),
  }

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const response = await fetch(SLACK_POST_MESSAGE_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        Authorization: `Bearer ${config.botToken}`,
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30_000),
    })

    if (
      (response.status === 429 || response.status >= 500) &&
      attempt < MAX_ATTEMPTS
    ) {
      const retryAfter = Number(response.headers.get("retry-after"))
      await sleep(
        Number.isFinite(retryAfter) && retryAfter > 0
          ? retryAfter * 1000
          : 1000 * 2 ** attempt,
      )
      continue
    }
    if (!response.ok) {
      throw new Error(`Slack chat.postMessage failed: HTTP ${response.status}`)
    }

    // Slack answers HTTP 200 with ok: false for application errors.
    const result = (await response.json()) as SlackResponse
    if (!result.ok || !result.ts) {
      const details = result.response_metadata?.messages?.join("; ")
      throw new Error(
        `Slack chat.postMessage failed: ${result.error ?? "no ts in response"}${details ? ` (${details})` : ""}`,
      )
    }
    return result.ts
  }
  throw new Error(
    `Slack chat.postMessage failed after ${MAX_ATTEMPTS} attempts`,
  )
}

/** Posts the summary, then the details as a thread reply if needed. Returns the summary ts. */
export async function postReport(
  config: SlackConfig,
  payload: SlackPayload,
): Promise<string> {
  const ts = await postMessage(config, payload.main)
  if (payload.thread) await postMessage(config, payload.thread, ts)
  return ts
}
