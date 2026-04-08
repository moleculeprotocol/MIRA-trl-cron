import { TEST_PROJECTS } from "../lib/config.js"
import {
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
  documentId: string
  symbol: string
}

function logScoringResults(
  symbol: string,
  documentId: string,
  scoringResult: NonNullable<
    Awaited<ReturnType<typeof calculateWeightedScore>>
  >,
): void {
  console.log("\n========== SCORING RESULTS ==========")
  console.log(`Project: ${symbol} (${documentId})`)
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
  const { documentId, symbol } = project

  console.log("--------------------------------")
  console.log(`Processing Project: ${symbol} (ID: ${documentId})`)

  const currentHash = await getDataRoomHash(documentId)
  const hashChanged = await dataroomHashChanged(documentId, currentHash)
  const forceProcess = process.env.FORCE_PROCESS === "true"

  if (!hashChanged && !forceProcess) {
    console.log(
      `Skipping ${documentId} - dataroom hash unchanged (set FORCE_PROCESS=true to override)`,
    )
    return
  }

  const files = await getProjectDataRoomFiles(documentId)
  const extractableFiles = getPublicExtractableFiles(files)

  if (extractableFiles.length === 0) {
    console.log(
      `Skipping ${documentId} - no extractable public files in dataroom`,
    )
    return
  }

  console.log(`Found ${extractableFiles.length} extractable file(s) to process`)

  const extractedContent = await extractMultipleDocuments(
    documentId,
    extractableFiles,
  )
  const formattedContent = formatExtractedDocumentsAsText(extractedContent)

  if (!formattedContent) {
    console.log(`Skipping ${documentId} - no relevant content extracted`)
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
    logScoringResults(symbol, documentId, scoringResult)
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
      console.log(`Project: ${symbol} (${documentId})`)
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

  // Step 4: Update Sanity with TRL data, scoring results, and todos
  await updateTrlAndScoringAsDraft(
    documentId,
    symbol,
    trlAnalysis,
    currentHash,
    scoringResult,
    todos,
  )

  console.log(`Completed: ${symbol} - TRL: ${trlAnalysis.trl_classification}`)
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
      documentId: p.tokenId,
      symbol: p.symbol,
    }))
  } else {
    console.log("Using test projects (set USE_ALL_PROJECTS=true for all)...")
    projects = TEST_PROJECTS
  }

  console.log(`Processing ${projects.length} project(s)...\n`)

  for (const project of projects) {
    try {
      await processProject(project)
    } catch (error) {
      console.error(`Error processing ${project.symbol}:`, error)
    }
  }

  console.log("\n================================")
  console.log("All projects processed")
}

main().catch((error) => {
  console.error("Fatal error:", error)
  process.exit(1)
})
