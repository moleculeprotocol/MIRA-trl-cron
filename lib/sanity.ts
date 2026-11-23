import { createClient } from "@sanity/client"
import { SANITY_DATASETS, SANITY_PROJECT_ID } from "./config.js"
import type { ScoringResult, TodoItem, TrlAnalysis } from "./llm.js"
import { notifySlack } from "./slack.js"
import { closedTodosOf, mergeTodos, type StoredTodo } from "./todos.js"

type Environment = "staging" | "production"

const environment: Environment =
  process.env.ENVIRONMENT === "production" ? "production" : "staging"

const dataset = SANITY_DATASETS[environment]

console.log(`Using ${environment} environment (dataset: ${dataset})`)

const publishImmediately = process.env.SANITY_PUBLISH_IMMEDIATELY === "true"

const ONCHAIN_LAB_TYPE = "onChainLab"

const client = createClient({
  projectId: SANITY_PROJECT_ID,
  dataset,
  apiVersion: "2026-01-21",
  token: process.env.SANITY_API_TOKEN,
  useCdn: false,
})

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

/**
 * Returns the todos the team has closed for a project. Reads the draft first
 * (the latest editorial state), falling back to the published document.
 */
export async function getClosedTodos(oclId: string): Promise<StoredTodo[]> {
  const [draft, published] = await Promise.all([
    client.getDocument(`drafts.${oclId}`),
    client.getDocument(oclId),
  ])
  return closedTodosOf(draft ?? published)
}

export async function updateTrlAndScoringAsDraft(
  oclId: string,
  name: string,
  shortname: string | null,
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

  // The onChainLab document must already exist. We only add TRL/scoring data.
  const [existingDraft, published] = await Promise.all([
    client.getDocument(draftId),
    client.getDocument(oclId),
  ])

  if (!published) {
    throw new Error(`onChainLab ${oclId} not found`)
  }

  if (published._type !== ONCHAIN_LAB_TYPE) {
    throw new Error(
      `Document ${oclId} is of type "${published._type}", expected "${ONCHAIN_LAB_TYPE}"`,
    )
  }

  // Replace the open todos, but keep the ones the team closed. Re-read them
  // here (not earlier) so todos closed during this run are not lost.
  const closedTodos = closedTodosOf(existingDraft ?? published)
  if (todos && todos.length > 0) {
    trlData.todos = mergeTodos(todos, closedTodos)
  }

  if (publishImmediately) {
    // Write directly to the published document, skipping the draft entirely.
    await client.patch(oclId).set(trlData).commit()

    // If a stale draft exists, discard it so it doesn't shadow the published data.
    if (existingDraft) {
      await client.delete(draftId)
    }
  } else if (existingDraft) {
    await client.patch(draftId).set(trlData).commit()
  } else {
    await client.create({
      ...published,
      _id: draftId,
      ...trlData,
    })
  }

  const todoCount = todos?.length ?? 0
  const todoSummary =
    todoCount > 0
      ? ` and ${todoCount} todo(s)${closedTodos.length > 0 ? ` (kept ${closedTodos.length} closed)` : ""}`
      : ""
  const action = publishImmediately ? "Published" : "Draft created/updated"
  console.log(
    `${action} for ${oclId} with trlValue: ${trlAnalysis.trl_classification}${scoringResult ? ` and scoring (final: ${scoringResult.scoring.final_weighted_score.toFixed(2)})` : ""}${todoSummary}`,
  )

  if (commitHash) {
    await notifySlack({
      oclId,
      name,
      shortname,
      trlAnalysis,
      publishImmediately,
    })
  } else {
    console.log(
      `Skipping Slack notification for ${oclId} (extraction incomplete, will retry next run)`,
    )
  }
}
