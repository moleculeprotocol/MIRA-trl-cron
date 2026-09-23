/**
 * Loads and validates the environment variables. Missing values fail loudly:
 * there are no defaults, because a silent default (for example the wrong
 * Langfuse environment) produces a plausible but wrong report.
 */

export interface LangfuseConfig {
  publicKey: string
  secretKey: string
  baseUrl: string
  environment: string
}

export interface InsightsConfig {
  anthropicApiKey: string
  model: string
}

export interface SlackConfig {
  botToken: string
  channelId: string
}

export class ConfigError extends Error {
  constructor(missing: string[]) {
    super(`Missing required environment variables: ${missing.join(", ")}`)
    this.name = "ConfigError"
  }
}

function read(
  env: NodeJS.ProcessEnv,
  names: readonly string[],
): Record<string, string> {
  const missing = names.filter((name) => !env[name]?.trim())
  if (missing.length > 0) throw new ConfigError(missing)
  return Object.fromEntries(
    names.map((name) => [name, (env[name] as string).trim()]),
  )
}

export function loadLangfuseConfig(
  env: NodeJS.ProcessEnv = process.env,
): LangfuseConfig {
  const v = read(env, [
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_ENVIRONMENT",
  ])
  return {
    publicKey: v.LANGFUSE_PUBLIC_KEY,
    secretKey: v.LANGFUSE_SECRET_KEY,
    baseUrl: v.LANGFUSE_BASE_URL.replace(/\/+$/, ""),
    environment: v.LANGFUSE_ENVIRONMENT,
  }
}

export function loadInsightsConfig(
  env: NodeJS.ProcessEnv = process.env,
): InsightsConfig {
  const v = read(env, ["ANTHROPIC_API_KEY", "INSIGHTS_MODEL"])
  return { anthropicApiKey: v.ANTHROPIC_API_KEY, model: v.INSIGHTS_MODEL }
}

export function loadSlackConfig(
  env: NodeJS.ProcessEnv = process.env,
): SlackConfig {
  const v = read(env, ["SLACK_BOT_TOKEN", "SLACK_CHANNEL_ID"])
  return { botToken: v.SLACK_BOT_TOKEN, channelId: v.SLACK_CHANNEL_ID }
}
