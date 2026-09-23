import type { Window } from "./types.js"

const DAY_MS = 24 * 60 * 60 * 1000

export interface ReportWindows {
  current: Window
  previous: Window
  days: number
}

function startOfUtcDay(date: Date): Date {
  return new Date(
    Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()),
  )
}

/** Parses YYYY-MM-DD (or a full ISO timestamp) as UTC. Throws on garbage. */
export function parseUtcDate(value: string, flag: string): Date {
  const iso = /^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T00:00:00Z` : value
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) {
    throw new Error(`Invalid date for ${flag}: "${value}"`)
  }
  return date
}

/**
 * The current window is [to - days, to). By default `to` is today 00:00 UTC,
 * so the window covers the last `days` full days. The previous window has the
 * same length and ends where the current one starts.
 */
export function computeWindows(options: {
  now: Date
  days: number
  from?: string
  to?: string
}): ReportWindows {
  const to = options.to
    ? parseUtcDate(options.to, "--to")
    : startOfUtcDay(options.now)
  const from = options.from
    ? parseUtcDate(options.from, "--from")
    : new Date(to.getTime() - options.days * DAY_MS)

  if (from.getTime() >= to.getTime()) {
    throw new Error(
      `Window start ${from.toISOString()} must be before end ${to.toISOString()}`,
    )
  }

  const lengthMs = to.getTime() - from.getTime()
  const previousFrom = new Date(from.getTime() - lengthMs)

  return {
    current: { from: from.toISOString(), to: to.toISOString() },
    previous: { from: previousFrom.toISOString(), to: from.toISOString() },
    days: Math.round((lengthMs / DAY_MS) * 10) / 10,
  }
}
