import { createClient } from "@sanity/client"
import { SANITY_DATASETS, SANITY_PROJECT_ID } from "./config.js"
import type { ScoringResult, TodoItem, TrlAnalysis } from "./llm.js"

type Environment = "staging" | "production"

const environment: Environment =
  process.env.ENVIRONMENT === "production" ? "production" : "staging"

const dataset = SANITY_DATASETS[environment]

console.log(`Using ${environment} environment (dataset: ${dataset})`)

const ONCHAIN_LAB_TYPE = "onChainLab"

interface NotifyDiscordParams {
  oclId: string
  name: string
  trlAnalysis: TrlAnalysis
}

const client = createClient({
  projectId: SANITY_PROJECT_ID,
  dataset,
  apiVersion: "2026-01-21",
  token: process.env.SANITY_API_TOKEN,
  useCdn: false,
})

async function notifyDiscord({
  oclId,
  name,
  trlAnalysis,
}: NotifyDiscordParams) {
  if (environment !== "production") {
    console.log("Skipping Discord notification (not production)")
    return
  }

  if (!process.env.DISCORD_WEBHOOK_URL) {
    console.log("DISCORD_WEBHOOK_URL not set, skipping notification")
    return
  }

  if (!process.env.SANITY_STUDIO_URL) {
    console.log("SANITY_STUDIO_URL not set, skipping notification")
    return
  }

  const studioUrl = `${process.env.SANITY_STUDIO_URL}/structure/onChainLabs;onChainLab;${oclId}`

  try {
    const response = await fetch(process.env.DISCORD_WEBHOOK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: "📝 New lab TRL draft by MIRA ready for review!",
        embeds: [
          {
            title: name,
            url: studioUrl,
            fields: [
              {
                name: "TRL Value",
                value: String(trlAnalysis.trl_classification),
                inline: true,
              },
              {
                name: "Confidence",
                value: String(trlAnalysis.confidence),
                inline: true,
              },
              {
                name: "Rationale",
                value: trlAnalysis.rationale,
                inline: false,
              },
            ],
            color: 0x00ff00,
          },
        ],
      }),
    })

    if (!response.ok) {
      console.error(`Discord notification failed: ${response.status}`)
    }
  } catch (error) {
    console.error("Failed to send Discord notification:", error)
  }
}

export async function dataroomHashChanged(
  oclId: string,
  currentHash: string,
): Promise<boolean> {
  const draftId = `drafts.${oclId}`

  // Check both draft and published documents - the hash is stored in the draft
  // until it gets published, so we need to check both
  const [draft, published] = await Promise.all([
    client.getDocument(draftId),
    client.getDocument(oclId),
  ])

  if (!published) {
    throw new Error(
      `onChainLab ${oclId} not found. Cannot check if dataroom hash changed.`,
    )
  }

  // If a draft exists, use its hash (most recent); otherwise use published hash
  const storedHash = draft?.dataroomHash ?? published.dataroomHash

  return storedHash !== currentHash
}

function criterionKeyToName(key: string): string {
  return key
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ")
}

function transformScoringForSanity(scoringResult: ScoringResult) {
  const { scoring } = scoringResult

  const criterionScores = Object.entries(scoring.criterion_scores).map(
    ([key, data]) => ({
      _type: "criterionScore" as const,
      _key: key,
      criterionKey: key,
      criterionName: criterionKeyToName(key),
      score: data.score,
      weight: data.weight,
      weightedScore: data.weighted_score,
      rationale: data.rationale,
    }),
  )

  return {
    _type: "projectScoring" as const,
    finalWeightedScore: scoring.final_weighted_score,
    scoreInterpretation: scoring.score_interpretation,
    criterionScores,
    scoredAt: new Date().toISOString(),
  }
}

function transformTodosForSanity(todos: TodoItem[]) {
  return todos.map((todo, index) => ({
    _type: "todoItem" as const,
    _key: `${todo.criterion_key}-${index}`,
    criterionKey: todo.criterion_key,
    title: todo.title,
    description: todo.description,
    effort: todo.effort,
    specificity: todo.specificity,
    type: todo.type,
    status: "open",
  }))
}

export async function updateTrlAndScoringAsDraft(
  oclId: string,
  name: string,
  trlAnalysis: TrlAnalysis,
  hash: string,
  scoringResult: ScoringResult | null,
  todos: TodoItem[] | null = null,
  commitHash = true,
) {
  const draftId = `drafts.${oclId}`

  const trlData: Record<string, unknown> = {
    trlValue: trlAnalysis.trl_classification,
    trlConfidence: trlAnalysis.confidence,
    trlRationale: trlAnalysis.rationale,
    trlLastUpdatedAt: new Date().toISOString(),
  }

  // Skip hash on partial extraction failure so the next run retries.
  if (commitHash) {
    trlData.dataroomHash = hash
  }

  if (scoringResult) {
    trlData.projectScoring = transformScoringForSanity(scoringResult)
  }

  if (todos && todos.length > 0) {
    trlData.todos = transformTodosForSanity(todos)
  }

  // The onChainLab document must already exist. We only add TRL/scoring data.
  const existingDraft = await client.getDocument(draftId)
  const published = await client.getDocument(oclId)

  if (!published) {
    throw new Error(`onChainLab ${oclId} not found`)
  }

  if (published._type !== ONCHAIN_LAB_TYPE) {
    throw new Error(
      `Document ${oclId} is of type "${published._type}", expected "${ONCHAIN_LAB_TYPE}"`,
    )
  }

  if (existingDraft) {
    await client.patch(draftId).set(trlData).commit()
  } else {
    await client.create({
      ...published,
      _id: draftId,
      ...trlData,
    })
  }

  const todoCount = todos?.length ?? 0
  console.log(
    `Draft created/updated for ${oclId} with trlValue: ${trlAnalysis.trl_classification}${scoringResult ? ` and scoring (final: ${scoringResult.scoring.final_weighted_score.toFixed(2)})` : ""}${todoCount > 0 ? ` and ${todoCount} todo(s)` : ""}`,
  )

  await notifyDiscord({
    oclId,
    name,
    trlAnalysis,
  })
}
