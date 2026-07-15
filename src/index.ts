import { TEST_PROJECTS } from "../lib/config.js"
import {
  assessScheduleStatus,
  calculateWeightedScore,
  extractMultipleDocuments,
  fillOutQuestionnaire,
  formatExtractedDocumentsAsText,
  generateTodos,
  getTrlAnalysis,
} from "../lib/llm.js"
import {
  getAllProjects,
  getDataRoomHash,
  getProjectDataRoomFiles,
  getPublicExtractableFiles,
} from "../lib/molecule.js"
import {
  dataroomHashChanged,
  updateTrlAndScoringAsDraft,
} from "../lib/sanity.js"

interface Project {
  oclId: string
  name: string
}

function logScoringResults(
  name: string,
  oclId: string,
  scoringResult: NonNullable<
    Awaited<ReturnType<typeof calculateWeightedScore>>
  >,
): void {
  console.log("\n========== SCORING RESULTS ==========")
  console.log(`Project: ${name} (${oclId})`)
  console.log(`TRL Level: ${scoringResult.trlLevel}`)
  console.log("\nCriterion Scores:")

  const criterionScores = scoringResult.scoring.criterion_scores
  for (const [criterion, data] of Object.entries(criterionScores)) {
    const score = data as {
      score: number
      weight: number
      weighted_score: number
      rationale: string
    }
    console.log(`\n  ${criterion.replace(/_/g, " ").toUpperCase()}:`)
    console.log(`    Score: ${score.score}/5`)
    console.log(`    Weight: ${(score.weight * 100).toFixed(0)}%`)
    console.log(`    Weighted Score: ${score.weighted_score.toFixed(2)}`)
    console.log(`    Rationale: ${score.rationale}`)
  }

  console.log(
    `\nFinal Weighted Score: ${scoringResult.scoring.final_weighted_score.toFixed(2)}`,
  )
  console.log(`\nInterpretation: ${scoringResult.scoring.score_interpretation}`)
  console.log("=====================================\n")
}

async function processProject(project: Project): Promise<void> {
  const { oclId, name } = project

  console.log("--------------------------------")
  console.log(`Processing Project: ${name} (OCL ID: ${oclId})`)

  const currentHash = await getDataRoomHash(oclId)
  const hashChanged = await dataroomHashChanged(oclId, currentHash)
  const forceProcess = process.env.FORCE_PROCESS === "true"

  if (!hashChanged && !forceProcess) {
    console.log(
      `Skipping ${oclId} - dataroom hash unchanged (set FORCE_PROCESS=true to override)`,
    )
    return
  }

  const files = await getProjectDataRoomFiles(oclId)
  const extractableFiles = getPublicExtractableFiles(files)

  if (extractableFiles.length === 0) {
    console.log(`Skipping ${oclId} - no extractable public files in dataroom`)
    return
  }

  console.log(`Found ${extractableFiles.length} extractable file(s) to process`)

  const { extractions, complete } = await extractMultipleDocuments(
    oclId,
    extractableFiles,
  )
  const formattedContent = formatExtractedDocumentsAsText(extractions)

  if (!complete) {
    console.log(
      `Extraction incomplete for ${oclId} - proceeding with best-effort content but NOT committing dataroom hash (will retry next run)`,
    )
  }

  if (!formattedContent) {
    console.log(`Skipping ${oclId} - no relevant content extracted`)
    return
  }

  // Step 1: Fill out questionnaire and get TRL analysis
  const questionnaireText = await fillOutQuestionnaire(formattedContent)
  const trlAnalysis = await getTrlAnalysis(questionnaireText)

  console.log(`TRL Classification: ${trlAnalysis.trl_classification}`)

  // Step 2: Calculate weighted score using the TRL we just determined
  // Only calculates for TRL 1, 2, 3 - returns null for other levels
  const scoringResult = await calculateWeightedScore(
    trlAnalysis.trl_classification,
    formattedContent,
  )

  if (scoringResult) {
    logScoringResults(name, oclId, scoringResult)
  }

  // Step 3: Generate todos based on scoring results and data room content
  let todos = null
  if (scoringResult) {
    todos = await generateTodos(
      trlAnalysis.trl_classification,
      scoringResult,
      formattedContent,
    )

    if (todos && todos.length > 0) {
      console.log("\n========== GENERATED TODOS ==========")
      console.log(`Project: ${name} (${oclId})`)
      console.log(`TRL Level: ${trlAnalysis.trl_classification}`)
      console.log(`\nTodos (${todos.length}):`)

      const publicTodos = todos.filter((t) => t.type === "public")
      const privateTodos = todos.filter((t) => t.type === "private")

      if (publicTodos.length > 0) {
        console.log(`\n  --- Public Todos (${publicTodos.length}) ---`)
        for (const todo of publicTodos) {
          console.log(`\n  [${todo.criterion_key}] ${todo.title}`)
          console.log(`    Type: ${todo.type}`)
          console.log(`    Effort: ${todo.effort}`)
          console.log(`    Description: ${todo.description}`)
          console.log(`    Specificity: ${todo.specificity}`)
        }
      }

      if (privateTodos.length > 0) {
        console.log(`\n  --- Private Todos (${privateTodos.length}) ---`)
        for (const todo of privateTodos) {
          console.log(`\n  [${todo.criterion_key}] ${todo.title}`)
          console.log(`    Type: ${todo.type}`)
          console.log(`    Effort: ${todo.effort}`)
          console.log(`    Description: ${todo.description}`)
          console.log(`    Specificity: ${todo.specificity}`)
        }
      }

      console.log("\n=====================================")
    }
  }

  // Step 4: Assess schedule status (traffic light) from milestone/document dates
  const scheduleStatus = await assessScheduleStatus(extractions)

  if (scheduleStatus) {
    console.log("\n========== SCHEDULE STATUS ==========")
    console.log(`Project: ${name} (${oclId})`)
    console.log(`Traffic Light: ${scheduleStatus.traffic_light.toUpperCase()}`)
    console.log(`Confidence: ${scheduleStatus.confidence}`)
    console.log(`\nRationale (investors): ${scheduleStatus.rationale}`)
    console.log(`\nFix (team): ${scheduleStatus.fix}`)
    if (scheduleStatus.signals.length > 0) {
      console.log(`\nSignals (${scheduleStatus.signals.length}):`)
      for (const s of scheduleStatus.signals) {
        const ref = s.reference_date ? ` [${s.reference_date}]` : ""
        const src = s.source_document ? ` (${s.source_document})` : ""
        console.log(`  • (${s.impact})${ref} ${s.observation}${src}`)
      }
    }
    console.log("=====================================\n")
  }

  // Step 5: Update Sanity with TRL data, scoring results, and todos
  await updateTrlAndScoringAsDraft(
    oclId,
    name,
    trlAnalysis,
    currentHash,
    scoringResult,
    todos,
    complete,
  )

  console.log(`Completed: ${name} - TRL: ${trlAnalysis.trl_classification}`)
}

/**
 * Main entry point for local development.
 * Processes test projects or all projects based on USE_ALL_PROJECTS env var.
 */
async function main(): Promise<void> {
  const useAllProjects = process.env.USE_ALL_PROJECTS === "true"

  let projects: Project[]

  if (useAllProjects) {
    console.log("Fetching all projects from GraphQL...")
    const fetchedProjects = await getAllProjects()
    projects = fetchedProjects.map((p) => ({
      oclId: p.oclId,
      name: p.name,
    }))
  } else {
    console.log("Using test projects (set USE_ALL_PROJECTS=true for all)...")
    projects = TEST_PROJECTS.map((oclId) => ({ oclId, name: oclId }))
  }

  console.log(`Processing ${projects.length} project(s)...\n`)

  for (const project of projects) {
    try {
      await processProject(project)
    } catch (error) {
      console.error(`Error processing ${project.name}:`, error)
    }
  }

  console.log("\n================================")
  console.log("All projects processed")
}

main().catch((error) => {
  console.error("Fatal error:", error)
  process.exit(1)
})
