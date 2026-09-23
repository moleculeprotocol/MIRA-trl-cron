/**
 * Bi-weekly MIRA chat report: Langfuse -> aggregate -> (LLM themes) -> console | Slack.
 *
 *   pnpm report                      print the last 14 full days (UTC)
 *   pnpm report --slack              post to Slack
 *   pnpm report --dump raw.json      also write the fetched data
 *   pnpm report --from-dump raw.json re-run without calling Langfuse
 */
import { readFile, writeFile } from "node:fs/promises"
import { parseArgs } from "node:util"
import { aggregate } from "./aggregate.js"
import {
  loadInsightsConfig,
  loadLangfuseConfig,
  loadSlackConfig,
} from "./config.js"
import { generateInsights, type Insights } from "./insights.js"
import { fetchRawData, LangfuseClient } from "./langfuse.js"
import { renderConsole } from "./render-console.js"
import { renderSlack } from "./render-slack.js"
import { postReport } from "./slack.js"
import { RAW_DATA_VERSION, type RawData } from "./types.js"
import { computeWindows } from "./window.js"

const USAGE = `Usage: pnpm report [options]

  --days <n>          Window length in days (default 14)
  --from <date>       Window start, ISO date (inclusive, UTC)
  --to <date>         Window end, ISO date (exclusive, UTC; default today 00:00)
  --slack             Post to Slack instead of printing
  --no-llm            Skip the LLM themes step
  --dump <file>       Write the raw fetched data as JSON
  --from-dump <file>  Re-run from a dump without calling Langfuse
  --help              Show this help`

function parseCli(argv: string[]) {
  const { values } = parseArgs({
    args: argv,
    options: {
      days: { type: "string" },
      from: { type: "string" },
      to: { type: "string" },
      slack: { type: "boolean", default: false },
      "no-llm": { type: "boolean", default: false },
      dump: { type: "string" },
      "from-dump": { type: "string" },
      help: { type: "boolean", default: false },
    },
    strict: true,
  })

  const days = values.days === undefined ? 14 : Number(values.days)
  if (!Number.isInteger(days) || days < 1) {
    throw new Error(`--days must be a positive integer, got "${values.days}"`)
  }
  if (values["from-dump"] && (values.from || values.to || values.days)) {
    throw new Error(
      "--from-dump uses the window stored in the dump; drop --days/--from/--to",
    )
  }
  if (values["from-dump"] && values.dump) {
    throw new Error("--dump and --from-dump can't be combined")
  }

  return {
    days,
    from: values.from,
    to: values.to,
    slack: values.slack,
    llm: !values["no-llm"],
    dump: values.dump,
    fromDump: values["from-dump"],
    help: values.help,
  }
}

async function loadDump(file: string): Promise<RawData> {
  const data = JSON.parse(await readFile(file, "utf8")) as RawData
  if (data.version !== RAW_DATA_VERSION) {
    throw new Error(
      `Dump ${file} has version ${data.version}, expected ${RAW_DATA_VERSION}. Fetch a new dump.`,
    )
  }
  return data
}

function windowDays(raw: RawData): number {
  const ms =
    new Date(raw.current.window.to).getTime() -
    new Date(raw.current.window.from).getTime()
  return Math.round((ms / 86_400_000) * 10) / 10
}

async function main(): Promise<void> {
  const cli = parseCli(process.argv.slice(2))
  if (cli.help) {
    console.log(USAGE)
    return
  }

  // Validate all needed config up front, before any slow network call.
  const insightsConfig = cli.llm ? loadInsightsConfig() : null
  const slackConfig = cli.slack ? loadSlackConfig() : null

  let raw: RawData
  if (cli.fromDump) {
    raw = await loadDump(cli.fromDump)
    console.error(`Loaded ${cli.fromDump} (fetched ${raw.fetchedAt})`)
  } else {
    const langfuseConfig = loadLangfuseConfig()
    const now = new Date()
    const windows = computeWindows({
      now,
      days: cli.days,
      from: cli.from,
      to: cli.to,
    })
    console.error(
      `Fetching ${langfuseConfig.environment} for ${windows.current.from} – ${windows.current.to} (and previous period)…`,
    )
    raw = await fetchRawData(
      new LangfuseClient(langfuseConfig),
      langfuseConfig.environment,
      windows,
      now,
    )
    console.error(
      `Fetched ${raw.current.messages.length} messages, ${raw.current.scores.length} scores, ${raw.current.observations.length} observations`,
    )
    if (cli.dump) {
      await writeFile(cli.dump, `${JSON.stringify(raw, null, 2)}\n`)
      console.error(`Wrote raw data to ${cli.dump}`)
    }
  }

  const report = aggregate(raw, windowDays(raw))
  // Data warnings aren't part of the report; they go to the log instead.
  for (const warning of report.warnings) console.error(`Warning: ${warning}`)

  let insights: Insights | null = null
  if (insightsConfig) {
    console.error(`Generating themes with ${insightsConfig.model}…`)
    insights = await generateInsights(
      insightsConfig,
      raw.current.messages,
      report,
    )
  }

  if (slackConfig) {
    const payload = renderSlack(report, insights, { llmSkipped: !cli.llm })
    const ts = await postReport(slackConfig, payload)
    console.error(`Posted to Slack channel ${slackConfig.channelId} (ts ${ts})`)
  } else {
    console.log(renderConsole(report, insights, { llmSkipped: !cli.llm }))
  }
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error)
  process.exit(1)
})
