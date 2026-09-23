import type { Delta, Rate } from "./aggregate.js"

export function pct(value: number | null, digits = 0): string {
  return value === null ? "n/a" : `${(value * 100).toFixed(digits)}%`
}

export function rateText(r: Rate): string {
  return r.rate === null
    ? "n/a (no data)"
    : `${pct(r.rate)} (${r.numerator}/${r.denominator})`
}

export function usd(value: number | null): string {
  if (value === null) return "n/a"
  return value < 10 ? `$${value.toFixed(2)}` : `$${value.toFixed(0)}`
}

export function seconds(ms: number | null): string {
  return ms === null ? "n/a" : `${(ms / 1000).toFixed(1)}s`
}

export function plain(value: number | null): string {
  if (value === null) return "n/a"
  return Number.isInteger(value) ? String(value) : value.toFixed(1)
}

/**
 * Change vs. the previous period, e.g. "+12 (+30%)" or "-0.4s".
 * `kind` decides the unit of the absolute change.
 */
export function change(
  delta: Delta,
  kind: "count" | "ms" | "usd" | "ratio" = "count",
): string {
  const { current, previous } = delta
  if (current === null || previous === null) return "no comparison"
  const diff = current - previous
  const sign = diff > 0 ? "+" : diff < 0 ? "-" : "±"
  const abs = Math.abs(diff)

  let absText: string
  switch (kind) {
    case "ms":
      absText = `${sign}${(abs / 1000).toFixed(1)}s`
      break
    case "usd":
      absText = `${sign}$${abs.toFixed(2)}`
      break
    case "ratio":
      return `${sign}${(abs * 100).toFixed(1)} pp`
    default:
      absText = `${sign}${Number.isInteger(abs) ? abs : abs.toFixed(1)}`
  }
  if (previous === 0) return current === 0 ? "±0" : `${absText} (new)`
  const relative = (diff / previous) * 100
  return `${absText} (${relative > 0 ? "+" : ""}${relative.toFixed(0)}%)`
}

export function dateRange(from: string, to: string): string {
  const start = from.slice(0, 10)
  // `to` is exclusive; show the last included day.
  const end = new Date(new Date(to).getTime() - 1).toISOString().slice(0, 10)
  return `${start} – ${end}`
}

export function topicLabel(category: string): string {
  return category.replace(/_/g, " ")
}
