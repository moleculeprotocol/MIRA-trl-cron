import * as fs from "node:fs/promises"
import * as path from "node:path"
import { anthropic } from "@ai-sdk/anthropic"
import { generateText, Output } from "ai"
import { z } from "zod"
import { EXTRACTION_MODEL, REASONING_MODEL } from "./config.js"
import type { DataRoomFile } from "./molecule.js"

const OUTPUT_BASE_DIR = path.join(process.cwd(), "output")

function getProjectOutputDir(projectId: string): string {
  return path.join(OUTPUT_BASE_DIR, projectId)
}

const QUESTIONNAIRE = `## Project Overview & Scientific Foundation
  - Project Title (Brief, descriptive name for the therapeutic program)
  - Therapeutic Concept Summary (One-paragraph overview of the approach and its novelty)
  - Target Indication (Specific disease or condition being addressed)
  - Patient Population Size (Estimated number of patients who could benefit from the therapy)
  - Mechanism of Action (Detailed explanation of how the intervention produces therapeutic effect)
  - Biological Target (Specific molecular target (protein, pathway, gene) being modulated (if considered non-confidential))
  - Development Timeline & Milestones (Next 3 major milestones (with estimated dates))

  ## Experimental Evidence & Development Stage
  This section is critical for TRL classification. Be thorough and specific.

  ### Laboratory Work Status
  - Has Experimental Work Been Conducted? (Yes/No - Answer "Yes" if ANY wet-lab experiments, assays, or empirical testing has been performed, even if preliminary or inconclusive)
  - Types of Experiments Performed (List all: binding assays, enzymatic assays, cell-based assays, animal studies, computational simulations with experimental validation, etc.)
  - Experimental Results Summary (Key quantitative findings: Kd values, IC50/EC50, % inhibition, binding affinity, selectivity ratios, p-values, etc.)

  ### Model Systems Used
  - In Vitro Systems (Cell lines used, biochemical assays, cell-free systems - specify if immortalized lines vs primary cells)
  - Advanced In Vitro Models (Patient-derived cells, organoids, 3D co-culture systems, tissue explants - specify if disease phenotypes demonstrated)
  - In Vivo Models (Animal models used: species, genetic background, disease model type - e.g., xenograft, transgenic, chemically-induced)
  - Disease Relevance of Models (Do the models recapitulate key pathophysiological features of the target human disease? Explain how)

  ### Efficacy Evidence
  - Proof-of-Concept Results (Has therapeutic effect been demonstrated? Describe outcomes: tumor size reduction, functional improvement, biomarker changes, disease progression delay)
  - Dose-Response Data (Has dose-response relationship been established? Provide data if available)
  - Reproducibility (Have key results been replicated? Number of independent experiments, statistical significance)
  - Controls and Quality (Were appropriate controls used? Describe experimental rigor)

  ### Safety & Development Readiness
  - Preliminary Safety Observations (Any toxicity signals observed in preclinical tests? Tolerability at efficacious doses?)
  - Therapeutic Window (Evidence of separation between efficacious and toxic doses?)
  - Manufacturing Considerations (GMP feasibility assessed? CMC challenges identified? Delivery route viability?)
  - Regulatory Pathway (IND-enabling studies planned or completed? Regulatory interactions?)

  ## Team Information
  - Key Personnel List (Names and roles of primary team members)
  - Affiliations (Names and location of the institution that team members work in/with)
  - Team Expertise Areas (Relevant backgrounds and capabilities of the team)

  ## Intellectual Property Status
  - Patent Application Status (Whether patents are provisional, filed, or granted)
  - Scope of IP Protection (Type of claims being pursued (composition of matter, method of use, target))
  - Freedom-to-Operate Assessment (Known blocking patents or FTO concerns)`

const TRL_INSTRUCTIONS = `## Instruction for Classification

Objective: Classify the project into TRL 1, 2, or 3 based on the presence and maturity of experimental evidence. Classification must be discrete with no intermediate assignments. If a project exhibits characteristics of multiple TRLs, assign the lower TRL to maintain conservative and rigorous evaluation standards.

Process: Follow the systematic decision tree using the stage definitions and key criteria to determine the appropriate TRL. Classification serves as a gateway decision that determines which detailed scoring rubric will be applied in subsequent evaluation. The classification system leverages machine-readable formats and standardized criteria, enabling AI models to automate classification while maintaining flexibility for human expert review in ambiguous cases.

## TRL 1: Concept Alone (Theoretical Stage)
### Stage Definition

At TRL 1, research focuses purely on conceptual or theoretical work, with no experimental evidence or practical applications. This stage is characterized by hypothesis formulation, literature synthesis, and theoretical modeling without empirical validation. Projects remain speculative, grounded in scientific reasoning and existing knowledge rather than direct experimental observation.

### Key Classification Criteria

1. Evidence of Research Activities
- No experimental research or laboratory testing has been conducted
- Documentation consists exclusively of theoretical models, literature reviews, white papers, or computational predictions
- Concept exists only in written or computational form

2. Nature of Work
- Project is entirely focused on theoretical or conceptual ideas
- No proof-of-concept experiments are present or planned in current scope
- Work involves analysis, synthesis, or modeling of existing knowledge

3. Scientific Hypothesis
- A biologically plausible hypothesis may exist but lacks any experimental validation
- Hypothesis strength is evaluated based on alignment with existing scientific knowledge and published literature
- Mechanism of action is proposed but unproven

### Boundary Considerations

TRL 1 vs. TRL 2 Distinction: If ANY experimental work has been conducted (even preliminary, inconclusive, or failed experiments), the project advances to TRL 2. The presence of empirical data, regardless of quality, indicates progression beyond pure concept.

## TRL 2: Speculative Research Conducted (Early Feasibility Stage)
### Stage Definition

At TRL 2, initial feasibility studies or speculative research are performed to test theoretical concepts. Experiments may be limited in scope, preliminary in nature, or inconclusive in results, but the presence of experimental data distinguishes this stage from purely conceptual work. Research at this level explores whether theoretical hypotheses can be validated through empirical observation, even if validation remains incomplete or uncertain.

### Key Classification Criteria
1. Evidence of Experimental Research
- Early feasibility or speculative research has been conducted (e.g., initial in vitro experiments, biochemical assays, computational simulations with experimental validation)
- Experimental design may lack robustness, use simplified systems, or have methodological limitations, but provides preliminary empirical insights
- Laboratory work has generated data, even if results are ambiguous or require further validation

2. Support for Hypothesis
- Early-stage experiments provide some support for the hypothesis but are not definitive
- Results may indicate potential mechanisms of action, initial feasibility, or preliminary target engagement
- Data suggest the concept is worth pursuing but require additional validation

3. Emerging Evidence
- Emerging experimental evidence validating the concept or mechanism of action is present, even if incomplete or preliminary
- Data demonstrate that the proposed approach is technically feasible in simplified systems
- Initial biological activity, target engagement, or phenotypic effects observed

### Boundary Considerations

TRL 2 vs. TRL 3 Distinction: If experiments have been conducted in disease-relevant in vivo models (animal disease models, patient-derived organoids) showing early efficacy, the project advances to TRL 3. The key differentiator is validation in physiologically relevant systems that model human disease, rather than simplified in vitro systems.

## TRL 3: Early-Stage Evidence for Practical Use (Foundational Research)
### Stage Definition

At TRL 3, foundational research provides evidence supporting practical application, with experiments confirming early feasibility in disease-relevant preclinical models. This stage is characterized by proof-of-concept in systems that recapitulate key aspects of human disease biology, demonstrating that the therapeutic approach shows efficacy in models predictive of clinical outcomes.

### Key Classification Criteria
1. Experimental Evidence in Disease-Relevant Models
- Experiments conducted in relevant in vivo models (e.g., rodent disease models, genetically engineered animals) or advanced in vitro models (e.g., patient-derived organoids, tissue explants, 3D co-culture systems) confirm early feasibility
- Evidence must demonstrate potential applicability to real-world therapeutic use cases
- Models recapitulate key pathophysiological features of the target human disease

2. Efficacy Demonstration
- Early-stage efficacy is shown in preclinical disease models with measurable, clinically relevant outcomes (e.g., tumor size reduction, functional improvement, biomarker modulation, disease progression delay)
- Results support the hypothesis with quantitative, reproducible outcomes
- Dose-response relationships or pharmacodynamic effects observed

3. Preliminary Safety and Scalability Considerations
- Initial indications of safety profile (e.g., absence of major toxicities in preclinical tests, tolerability at efficacious doses)
- Early consideration of development factors such as Good Manufacturing Practice (GMP) feasibility, Chemistry, Manufacturing, and Controls (CMC) requirements, or delivery route viability
- Preliminary assessment of therapeutic window and dose range

### Boundary Considerations

TRL 3 vs. TRL 4+ Distinction: Progression beyond TRL 3 typically involves IND-enabling studies, formal toxicology packages, GMP manufacturing, and regulatory pathway definition. While ERAF-AI focuses on TRL 1-3, projects demonstrating comprehensive preclinical packages may be candidates for transition to clinical development frameworks.

## Classification Decision Process
### Systematic Decision Tree

Question 1: Has ANY experimental laboratory work been conducted?
- NO: Project is TRL 1 (Concept Alone)
- YES: Proceed to Question 2

Question 2: Have experiments been conducted in disease-relevant in vivo models or advanced disease-modeling systems?
- NO: Project is TRL 2 (Speculative Research Conducted)
- YES: Proceed to Question 3

Question 3: Do results demonstrate early proof-of-concept efficacy in disease models with measurable outcomes?
- NO: Project is TRL 2 (experiments conducted but not yet validated in relevant models)
- YES: Project is TRL 3 (Early-Stage Evidence for Practical Use)

## Edge Cases and Ambiguity Resolution
### Conservative Classification Principle

When a project exhibits characteristics of multiple TRLs or falls in a gray area, assign the lower TRL. This ensures that projects are held to appropriate evidence standards before advancing to more stringent evaluation criteria.

### Common Ambiguous Scenarios

Computational Predictions with Experimental Validation: If computational models are supported by limited experimental validation, classify as TRL 2. Pure computational work without wet-lab validation remains TRL 1.

In Vitro Studies with Disease-Relevant Cells: Patient-derived cells in 2D culture generally constitute TRL 2. Patient-derived organoids or 3D tissue models with demonstrated disease phenotypes may qualify as TRL 3 if they show therapeutic effects.

Failed or Negative Experiments: Projects with completed experiments that yielded negative or inconclusive results should be classified based on the sophistication of models tested (TRL 2 for in vitro, TRL 3 for in vivo). Do not return to TRL 1; the presence of data indicates progression beyond concept stage.

Preliminary In Vivo Studies: Single pilot experiments in animal models without replication, dose-response data, or statistical rigor may remain TRL 2. Progression to TRL 3 requires reproducible efficacy with quantitative outcomes.

## Special Classifications
### Pre-TRL 1

Projects that lack a coherent scientific hypothesis or require substantial concept refinement before evaluation. These projects should be flagged for strategic redirection or additional foundational work before formal assessment.

### TRL >3

Projects demonstrating comprehensive preclinical packages beyond the scope of ERAF-AI (e.g., IND-enabling studies completed, GMP manufacturing established, formal toxicology packages). These projects should be flagged for evaluation under clinical-stage frameworks.

## Output Requirements

### Required Output Format:

TRL Classification: Pre-TRL 1, TRL 1, TRL 2, TRL 3, or TRL >3

Confidence: High, Medium, or Low

Rationale: Brief explanation (2-3 sentences) citing specific evidence from the project that supports the classification decision

Key Indicators: List the specific pieces of evidence that determined the classification
- Indicator 1
- Indicator 2
- Indicator 3

### Example Output:

TRL Classification: TRL 2 (Speculative Research Conducted)

Confidence: High

Rationale: The project has conducted initial in vitro binding assays demonstrating target engagement with measurable affinity (Kd = 50 nM). However, no experiments have been performed in disease-relevant animal models or organoid systems. The presence of preliminary experimental data advances the project beyond pure concept (TRL 1), but lack of in vivo validation keeps it at TRL 2.

Key Indicators:
- Biochemical binding data from cell-free assays (Kd = 50 nM)
- No disease-relevant in vivo or organoid studies conducted
- Experimental design uses simplified in vitro systems only`

const EXTRACTION_PROMPT = `You are extracting content from a biotech/science project document to support Technology Readiness Level (TRL) assessment.

## Your Task
Extract ALL scientifically relevant content from this document. Create one content block for each distinct section, figure, table, or page that contains meaningful information.

## What Makes Content TRL-Relevant
Focus on evidence that helps classify a project as TRL 1 (concept only), TRL 2 (early experiments), or TRL 3 (disease-model validation):

**Experimental Evidence** (most critical):
- What experiments were done? (assays, studies, screens)
- What model systems? (cell lines, organoids, animal models, patient samples)
- What results? (quantitative data, efficacy measures, dose-response)
- What quality indicators? (controls, replication, statistical significance)

**Development Progress**:
- Current stage (discovery, lead optimization, preclinical, IND-enabling)
- Manufacturing/CMC status
- Regulatory pathway and interactions
- Timeline and upcoming milestones

**Project Context**:
- Therapeutic target and mechanism of action
- Indication and patient population
- Team expertise and affiliations
- IP status (patents filed, granted, FTO)
- Funding status

## Content Block Guidelines
- Create separate blocks for: each page with distinct content, each figure/diagram, each table, each major text section
- For figures: describe what the figure shows, the experiment type, and extract any quantitative values (IC50, EC50, % inhibition, p-values, etc.)
- For tables: describe columns and extract key data points
- For text: summarize claims AND the evidence supporting them
- Include page numbers (1-indexed) when the document has multiple pages

## Skip These
- Logos, stock photos, building images
- Generic marketing language without data
- Contact info, legal disclaimers
- Content that adds nothing to TRL assessment

If the entire document is irrelevant (pure branding, logos only), just provide a brief summary explaining why and set relevance to "none".`

type SupportedMimeType =
  | "application/pdf"
  | "image/png"
  | "image/jpeg"
  | "image/gif"
  | "image/webp"

const trlValueSchema = z.object({
  trl_classification: z.enum([
    "pre-trl-1",
    "trl-1",
    "trl-2",
    "trl-3",
    "trl-gt-3",
  ]),
  confidence: z.enum(["high", "medium", "low"]),
  rationale: z.string(),
  key_indicators: z.array(z.string()),
})

const contentBlockSchema = z.object({
  page: z.number().optional().describe("Page number (1-indexed) if applicable"),
  type: z.string().describe("text, figure, table, diagram, image, or mixed"),
  title: z.string().optional().describe("Section or figure title if present"),
  content: z.string().describe("Extracted content, description, or data"),
})

const extractedDocumentSchema = z.object({
  document_type: z
    .string()
    .describe(
      "pitch_deck, experiment_report, publication, regulatory, presentation, figure, or other",
    ),
  relevance: z
    .enum(["high", "medium", "low", "none"])
    .describe("Relevance to TRL assessment"),
  summary: z.string().describe("2-3 sentence overview of the document"),

  // Flexible content extraction - works for any document type
  content_blocks: z
    .array(contentBlockSchema)
    .describe("Extracted content sections, figures, tables, etc."),

  // Key structured findings (simple strings, AI extracts what's relevant)
  key_findings: z
    .array(z.string())
    .describe(
      "Most important findings for TRL assessment: experiments, results, IP, regulatory, team, etc.",
    ),
})

async function saveCachedExtraction(
  projectId: string,
  cacheFilename: string,
  data: ExtractedDocument,
  filePath: string,
): Promise<void> {
  await ensureOutputDir(projectId)
  const cachePath = path.join(getProjectOutputDir(projectId), cacheFilename)
  const dataWithPath = { file_path: filePath, ...data }
  await fs.writeFile(cachePath, JSON.stringify(dataWithPath, null, 2), "utf-8")
}

function getMimeType(
  contentType: string,
  filePath: string,
): SupportedMimeType | null {
  const supportedTypes: SupportedMimeType[] = [
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
  ]

  if (supportedTypes.includes(contentType as SupportedMimeType)) {
    return contentType as SupportedMimeType
  }

  const extMap: Record<string, SupportedMimeType> = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
  }

  const ext = path.extname(filePath).toLowerCase()
  return extMap[ext] ?? null
}

async function generateCacheFilename(file: DataRoomFile): Promise<string> {
  const input = `${file.did}:${file.version}:${file.path}`
  const encoder = new TextEncoder()
  const data = encoder.encode(input)
  const hashBuffer = await crypto.subtle.digest("SHA-256", data)
  const hashArray = Array.from(new Uint8Array(hashBuffer))
  const hash = hashArray.map((b) => b.toString(16).padStart(2, "0")).join("")
  return `${hash}.json`
}

async function ensureOutputDir(projectId: string): Promise<void> {
  const projectDir = getProjectOutputDir(projectId)
  try {
    await fs.access(projectDir)
  } catch {
    await fs.mkdir(projectDir, { recursive: true })
  }
}

async function getCachedExtraction(
  projectId: string,
  cacheFilename: string,
): Promise<ExtractedDocument | null> {
  try {
    const cachePath = path.join(getProjectOutputDir(projectId), cacheFilename)
    const content = await fs.readFile(cachePath, "utf-8")
    return JSON.parse(content) as ExtractedDocument
  } catch {
    return null
  }
}

export async function fillOutQuestionnaire(inputText: string): Promise<string> {
  console.log("Starting AI questionnaire generation...")

  const prompt = `You are a biotech, science and pharma expert and your job is to fill out the following biotech project questionnaire based on the raw project data.
  This questionnaire will later be used to assess the Technology Readiness Levels (TRL) of the project, so keep that in mind when filling out the questionnaire.

  <questionnaire>
  ${QUESTIONNAIRE}
  </questionnaire>

  <raw-project-data>
  ${inputText}
  </raw-project-data>

  Fill out each field in the questionnaire based on the information found in the raw project data. If a field cannot be determined from the data, write "Not specified" for that field.

  Important:
  Format your response as markdown with a clear header for each field using ### headings`

  const { text } = await generateText({
    model: anthropic(REASONING_MODEL),
    prompt,
  })

  console.log("Questionnaire generation complete")
  return text
}

export type TrlAnalysis = z.infer<typeof trlValueSchema>

export async function getTrlAnalysis(
  questionnaireText: string,
): Promise<TrlAnalysis> {
  console.log("Starting AI TRL classification...")

  const prompt = `${TRL_INSTRUCTIONS}

<questionnaire>
${questionnaireText}
</questionnaire>`

  const { output } = await generateText({
    model: anthropic(REASONING_MODEL),
    prompt,
    output: Output.object({ schema: trlValueSchema }),
  })

  console.log(`TRL classification complete: ${output.trl_classification}`)
  return output
}

export type ExtractedDocument = z.infer<typeof extractedDocumentSchema>
export type ContentBlock = z.infer<typeof contentBlockSchema>

export async function extractDocumentContent(
  projectId: string,
  file: DataRoomFile,
): Promise<ExtractedDocument | null> {
  const cacheFilename = await generateCacheFilename(file)

  // Check cache first
  const cached = await getCachedExtraction(projectId, cacheFilename)
  if (cached) {
    console.log(`Using cached extraction for: ${file.path}`)
    return cached
  }

  const mimeType = getMimeType(file.contentType, file.path)
  if (!mimeType) {
    console.log(`Unsupported file type: ${file.path} (${file.contentType})`)
    return null
  }

  console.log(`Extracting content from: ${file.path}`)

  const response = await fetch(file.downloadUrl)
  if (!response.ok) {
    console.error(`Failed to download ${file.path}: ${response.status}`)
    return null
  }

  const fileData = Buffer.from(await response.arrayBuffer())
  const isPdf = mimeType === "application/pdf"

  try {
    const { output } = await generateText({
      model: anthropic(EXTRACTION_MODEL),
      messages: [
        {
          role: "user",
          content: isPdf
            ? [
                { type: "file" as const, data: fileData, mediaType: mimeType },
                { type: "text" as const, text: EXTRACTION_PROMPT },
              ]
            : [
                {
                  type: "image" as const,
                  image: fileData,
                  mediaType: mimeType,
                },
                { type: "text" as const, text: EXTRACTION_PROMPT },
              ],
        },
      ],
      output: Output.object({ schema: extractedDocumentSchema }),
    })

    await saveCachedExtraction(projectId, cacheFilename, output, file.path)
    console.log(`Extraction complete and cached: ${file.path}`)

    return output
  } catch (error) {
    console.error(`Failed to extract content from ${file.path}:`, error)
    return null
  }
}

export async function extractMultipleDocuments(
  projectId: string,
  files: DataRoomFile[],
): Promise<Map<string, ExtractedDocument>> {
  const results = new Map<string, ExtractedDocument>()

  for (const file of files) {
    const extracted = await extractDocumentContent(projectId, file)
    if (extracted) {
      results.set(file.path, extracted)
    }
  }

  return results
}

/**
 * Formats extracted documents into a single text string with filename and page context.
 * This preserves document structure for downstream processing like questionnaire filling.
 */
export function formatExtractedDocumentsAsText(
  extractions: Map<string, ExtractedDocument>,
): string {
  if (extractions.size === 0) {
    return ""
  }

  const sections: string[] = []

  for (const [filename, doc] of extractions) {
    // Skip documents with no relevance to TRL assessment
    if (doc.relevance === "none") {
      continue
    }

    const lines: string[] = []
    lines.push(`${"=".repeat(60)}`)
    lines.push(`FILE: ${filename}`)
    lines.push(`Type: ${doc.document_type} | Relevance: ${doc.relevance}`)
    lines.push(`${"=".repeat(60)}`)
    lines.push("")
    lines.push(`Summary: ${doc.summary}`)
    lines.push("")

    // Group content blocks by page for better readability
    const blocksByPage = new Map<number | undefined, ContentBlock[]>()
    for (const block of doc.content_blocks) {
      const page = block.page
      const existing = blocksByPage.get(page)
      if (existing) {
        existing.push(block)
      } else {
        blocksByPage.set(page, [block])
      }
    }

    // Sort pages (undefined pages come last)
    const sortedPages = [...blocksByPage.keys()].sort((a, b) => {
      if (a === undefined) return 1
      if (b === undefined) return -1
      return a - b
    })

    for (const page of sortedPages) {
      const blocks = blocksByPage.get(page)
      if (!blocks) continue
      if (page !== undefined) {
        lines.push(`--- Page ${page} ---`)
      }

      for (const block of blocks) {
        if (block.title) {
          lines.push(`[${block.type.toUpperCase()}] ${block.title}`)
        } else {
          lines.push(`[${block.type.toUpperCase()}]`)
        }
        lines.push(block.content)
        lines.push("")
      }
    }

    if (doc.key_findings.length > 0) {
      lines.push("--- Key Findings ---")
      for (const finding of doc.key_findings) {
        lines.push(`• ${finding}`)
      }
      lines.push("")
    }

    sections.push(lines.join("\n"))
  }

  return sections.join("\n\n")
}

// =============================================================================
// Weighted Scoring System
// =============================================================================

const SCORING_PROMPT = `Apply a comprehensive set of core evaluation criteria to fairly evaluate projects across the different TRL levels corresponding to the project's maturity/progress. Assign an integer score to each criterion using the scoring system (1-5) where 5 = strongest performance and 1 = weakest performance, then multiply each score by its weighted importance to obtain a weighted score. Keep a record of the individual criterion scores, along with the weighted importance value, the weighted scores calculated from the weighted scoring operation, and their final evaluation score for transparency.
Fill out the checklist below by evaluating each criterion based on your project's current status. Use the scoring system (1-5) where 5 = strongest/best and 1 = weakest/worst. Multiply each score by its weighted importance to obtain a weighted score.`

const TRL_1_GUIDE = `
## TRL 1 Scoring Guidelines
### Therapeutic Relevance of Mechanism of Action
Weighted Importance: 60%
Questions to Consider:
- Is the proposed mechanism scientifically plausible?
- Does it target a biologically relevant disease mechanism?
Score (1-5):
Weighted Score:


### Therapeutic Optionality
Weighted Importance: 15%
Questions to Consider:
- Can this mechanism address multiple therapeutic areas?
- What is the flexibility of the concept?
Score (1-5):
Weighted Score:


### Intellectual Property
Weighted Importance: 25%
Questions to Consider:
- Is the concept novel enough for patenting?
- Are there risks of prior art or competition?
Score (1-5):
Weighted Score:`

const TRL_2_GUIDE = `
## TRL 2 Scoring Guidelines
### Therapeutic Relevance of Mechanism of Action
Weighted Importance: 45%
Questions to Consider:
- Do early experimental results support the hypothesis?
- Does this reinforce the biological relevance?
Score (1-5):
Weighted Score:


### Therapeutic Optionality
Weighted Importance: 5%
Questions to Consider:
- Are there emerging findings suggesting alternative applications?
- What new pathways have been identified?
Score (1-5):
Weighted Score:


### Intellectual Property
Weighted Importance: 15%
Questions to Consider:
- Can early results strengthen IP filings?
- Is it time to formalize patent applications?
Score (1-5):
Weighted Score:


### Utility of Candidates
Weighted Importance: 20%
Questions to Consider:
- Are potential drug candidates emerging?
- How viable are they in achieving therapeutic effects?
Score (1-5):
Weighted Score:


### Prospects for Safety
Weighted Importance: 15%
Questions to Consider:
- Are there early signs of safety concerns or toxicity?
- How manageable are these risks?
Score (1-5):
Weighted Score:`

const TRL_3_GUIDE = `
## TRL 3 Scoring Guidelines
### Therapeutic Relevance of Mechanism of Action
Weighted Importance: 20%
Questions to Consider:
- Does the mechanism have clear therapeutic relevance?
- Is there strong experimental validation?
Score (1-5):
Weighted Score:


### Therapeutic Optionality
Weighted Importance: 5%
Questions to Consider:
- Are there clear indications for expanding into other disease areas?
- What additional applications are possible?
Score (1-5):
Weighted Score:


### Intellectual Property
Weighted Importance: 15%
Questions to Consider:
- Is there a robust IP strategy in place?
- Are patents being filed or licensing opportunities considered?
Score (1-5):
Weighted Score:


### Utility of Candidates
Weighted Importance: 20%
Questions to Consider:
- Do drug candidates show early efficacy in disease models?
- Can they meaningfully affect the target?
Score (1-5):
Weighted Score:


### Prospects for Safety
Weighted Importance: 15%
Questions to Consider:
- Is there evidence for safety and low toxicity?
- What further safety studies are needed?
Score (1-5):
Weighted Score:


### Prospects for GMP/CMC for IND Filing
Weighted Importance: 5%
Questions to Consider:
- Can candidates be produced at scale under GMP conditions?
- What are the manufacturing challenges?
Score (1-5):
Weighted Score:


### Prospects for Clinical Development
Weighted Importance: 10%
Questions to Consider:
- What are the next steps for clinical trials?
- Are there regulatory or logistical concerns?
Score (1-5):
Weighted Score:


### Commercial Potential
Weighted Importance: 5%
Questions to Consider:
- How competitive is the market?
- Are there clear commercial applications?
Score (1-5):
Weighted Score:


### Organization and Team Fit
Weighted Importance: 5%
Questions to Consider:
- Does the team have the expertise and resources?
- Are there gaps in key areas?
Score (1-5):
Weighted Score:`

// Zod schemas for criterion scores
// Note: Cannot use .min()/.max() as Anthropic doesn't support these constraints
const criterionScoreSchema = z.object({
  score: z
    .number()
    .describe("Score from 1-5 (5 = best/strongest, 1 = worst/weakest)"),
  weight: z.number().describe("Weight as decimal (e.g., 0.60 for 60%)"),
  weighted_score: z.number().describe("score * weight"),
  rationale: z.string().describe("Brief justification for the score"),
})

// TRL 1 scoring schema (3 criteria)
const trl1ScoringSchema = z.object({
  criterion_scores: z.object({
    therapeutic_relevance: criterionScoreSchema.describe(
      "Therapeutic Relevance of Mechanism of Action (60% weight)",
    ),
    therapeutic_optionality: criterionScoreSchema.describe(
      "Therapeutic Optionality (15% weight)",
    ),
    intellectual_property: criterionScoreSchema.describe(
      "Intellectual Property (25% weight)",
    ),
  }),
  final_weighted_score: z.number().describe("Sum of all weighted_scores"),
  score_interpretation: z
    .string()
    .describe("Overall assessment interpretation and summary"),
})

// TRL 2 scoring schema (5 criteria)
const trl2ScoringSchema = z.object({
  criterion_scores: z.object({
    therapeutic_relevance: criterionScoreSchema.describe(
      "Therapeutic Relevance of Mechanism of Action (45% weight)",
    ),
    therapeutic_optionality: criterionScoreSchema.describe(
      "Therapeutic Optionality (5% weight)",
    ),
    intellectual_property: criterionScoreSchema.describe(
      "Intellectual Property (15% weight)",
    ),
    utility_of_candidates: criterionScoreSchema.describe(
      "Utility of Candidates (20% weight)",
    ),
    prospects_for_safety: criterionScoreSchema.describe(
      "Prospects for Safety (15% weight)",
    ),
  }),
  final_weighted_score: z.number().describe("Sum of all weighted_scores"),
  score_interpretation: z
    .string()
    .describe("Overall assessment interpretation and summary"),
})

// TRL 3 scoring schema (9 criteria)
const trl3ScoringSchema = z.object({
  criterion_scores: z.object({
    therapeutic_relevance: criterionScoreSchema.describe(
      "Therapeutic Relevance of Mechanism of Action (20% weight)",
    ),
    therapeutic_optionality: criterionScoreSchema.describe(
      "Therapeutic Optionality (5% weight)",
    ),
    intellectual_property: criterionScoreSchema.describe(
      "Intellectual Property (15% weight)",
    ),
    utility_of_candidates: criterionScoreSchema.describe(
      "Utility of Candidates (20% weight)",
    ),
    prospects_for_safety: criterionScoreSchema.describe(
      "Prospects for Safety (15% weight)",
    ),
    prospects_for_gmp_cmc: criterionScoreSchema.describe(
      "Prospects for GMP/CMC for IND Filing (5% weight)",
    ),
    prospects_for_clinical_development: criterionScoreSchema.describe(
      "Prospects for Clinical Development (10% weight)",
    ),
    commercial_potential: criterionScoreSchema.describe(
      "Commercial Potential (5% weight)",
    ),
    organization_and_team_fit: criterionScoreSchema.describe(
      "Organization and Team Fit (5% weight)",
    ),
  }),
  final_weighted_score: z.number().describe("Sum of all weighted_scores"),
  score_interpretation: z
    .string()
    .describe("Overall assessment interpretation and summary"),
})

// Export types
export type CriterionScore = z.infer<typeof criterionScoreSchema>
export type Trl1Scoring = z.infer<typeof trl1ScoringSchema>
export type Trl2Scoring = z.infer<typeof trl2ScoringSchema>
export type Trl3Scoring = z.infer<typeof trl3ScoringSchema>
export type TrlScoring = Trl1Scoring | Trl2Scoring | Trl3Scoring

// Supported TRL levels for scoring
type ScoringTrlLevel = "trl-1" | "trl-2" | "trl-3"

function isScoringTrlLevel(
  trlLevel: TrlAnalysis["trl_classification"],
): trlLevel is ScoringTrlLevel {
  return trlLevel === "trl-1" || trlLevel === "trl-2" || trlLevel === "trl-3"
}

function getTrlGuide(trlLevel: ScoringTrlLevel): string {
  switch (trlLevel) {
    case "trl-1":
      return TRL_1_GUIDE
    case "trl-2":
      return TRL_2_GUIDE
    case "trl-3":
      return TRL_3_GUIDE
  }
}

function getTrlSchema(trlLevel: ScoringTrlLevel) {
  switch (trlLevel) {
    case "trl-1":
      return trl1ScoringSchema
    case "trl-2":
      return trl2ScoringSchema
    case "trl-3":
      return trl3ScoringSchema
  }
}

export interface ScoringResult {
  trlLevel: ScoringTrlLevel
  scoring: TrlScoring
}

/**
 * Calculates a weighted score for a project based on its TRL level.
 * Only supports TRL 1, 2, and 3. Returns null for other TRL levels.
 *
 * @param trlLevel - The TRL classification of the project
 * @param rawProjectData - The formatted extracted documents text
 * @returns The scoring result or null if TRL level is not supported
 */
export async function calculateWeightedScore(
  trlLevel: TrlAnalysis["trl_classification"],
  rawProjectData: string,
): Promise<ScoringResult | null> {
  if (!isScoringTrlLevel(trlLevel)) {
    console.log(`Scoring not supported for ${trlLevel}, skipping...`)
    return null
  }

  console.log(`Calculating weighted score for ${trlLevel}...`)

  const guide = getTrlGuide(trlLevel)
  const schema = getTrlSchema(trlLevel)

  const prompt = `${SCORING_PROMPT}

${guide}

<raw-project-data>
${rawProjectData}
</raw-project-data>

Based on the raw project data and TRL scoring guidelines above, evaluate the project according to the scoring criteria.
Remember: A score of 5 indicates strongest performance, 1 indicates weakest performance.
Calculate the weighted_score for each criterion by multiplying score * weight.
The final_weighted_score should be the sum of all weighted_scores.`

  const { output } = await generateText({
    model: anthropic(REASONING_MODEL),
    prompt,
    output: Output.object({ schema }),
  })

  console.log(`Scoring complete for ${trlLevel}`)

  return {
    trlLevel,
    scoring: output as TrlScoring,
  }
}

// =============================================================================
// Todo Generation
// =============================================================================

const todoItemSchema = z.object({
  criterion_key: z
    .string()
    .describe("Key matching a criterion from the scoring rubric"),
  title: z.string().describe("Short actionable title, max 80 chars"),
  description: z
    .string()
    .describe(
      "2-3 sentences: what exists, what's missing, what to do. Reference specific files from the data room by name.",
    ),
  effort: z
    .enum(["low", "medium", "high"])
    .describe(
      "low: team likely has this, just upload. medium: requires some work. high: requires new experiments/filings",
    ),
  specificity: z
    .string()
    .describe("Exactly what type of document or evidence to upload"),
  type: z
    .enum(["public", "private"])
    .describe(
      "public: the evidence/document is non-sensitive and can be reviewed by AI (e.g. pitch decks, published papers, granted patents, public presentations). private: the evidence/document contains sensitive or confidential data that AI cannot access or track (e.g. raw experiment data, patent applications in progress, proprietary formulations, clinical trial data, internal safety studies).",
    ),
})

const todoListSchema = z.array(todoItemSchema)

export type TodoItem = z.infer<typeof todoItemSchema>

const TODO_GENERATION_PROMPT = `You are a biotech project advisor. Generate actionable todo items to help improve a project's TRL scores.

## Scoring System
- Scores range from 1 (worst) to 5 (best)
- A score of 5 means perfect - no improvement needed
- Scores of 1, 2, 3, or 4 mean there's room for improvement - generate todos for these

## Your Goal

Generate two sets of todos — **public** and **private** — each containing 1 to 10 items, for a combined total of 2 to 20 todos.
Focus on the BEST effort-to-impact ratio. Quick wins first.

## CRITICAL: Respect Project Scope, Roadmap & Milestones

**Only suggest todos that fit within the project's current scope and near-term plans.**

Before generating any todo, check the data room contents for:
- **Roadmap** or development timeline information
- **Milestones** and their target dates
- **Current development stage** and immediate next steps
- **Project scope** definitions or boundaries

### Scoping Rules

1. **Stay within stated scope**: If the project defines its current focus (e.g., "completing lead optimization" or "preparing for IND filing"), don't suggest todos for later stages (e.g., clinical trial design, commercial manufacturing).

2. **Respect milestone sequence**: If the project lists milestones like "1. Complete binding assays → 2. In vivo studies → 3. IND-enabling", don't suggest IND-related todos if they're still on step 1.

3. **Match current TRL**: A TRL 1 project focused on concept validation shouldn't get todos about GMP manufacturing or clinical development plans.

4. **Near-term actionability**: Prioritize todos that can realistically be addressed in the project's current phase. A todo for "detailed Phase 3 trial design" is not helpful for an early-stage project.

5. **When in doubt, stay conservative**: If the data room doesn't mention a particular capability or future plan, don't assume the team should be working on it. Focus on what's clearly within their stated scope.

## CRITICAL: Public vs Private Classification

Every todo MUST be classified as either "public" or "private". This determines whether our AI system can later track the todo's progress by reviewing the uploaded evidence.

### Public Todos (type: "public")
These request documents/evidence that are **non-sensitive** and can be reviewed by AI to track progress.

Examples of PUBLIC documents:
- Pitch decks and investor presentations
- Published scientific papers and preprints
- Granted/registered patents (publicly available)
- Public conference posters and presentations
- Press releases and public announcements
- Published clinical trial registrations (e.g. ClinicalTrials.gov entries)
- Team CVs and publicly available bios
- Literature reviews and meta-analyses
- Public regulatory guidance documents
- Non-confidential project summaries and overviews

### Private Todos (type: "private")
These request documents/evidence that contain **sensitive, confidential, or proprietary data**. Our AI system CANNOT access or track these — only the team can confirm completion.

Examples of PRIVATE documents:
- Raw experimental data (binding assays, dose-response curves, cell-based assay results)
- Internal lab notebooks and experiment protocols
- Patent applications in progress (pre-grant / provisional filings)
- Freedom-to-operate (FTO) analyses
- Proprietary compound structures and formulations
- Internal safety/toxicology study reports
- Unpublished preclinical study data (in vivo efficacy, PK/PD)
- Manufacturing process details (GMP/CMC documentation)
- Confidential licensing or partnership agreements
- Internal clinical development plans
- Proprietary biomarker or target validation data
- Trade secrets and know-how documentation

### How to Decide
Ask: "Could this document be shared publicly or reviewed by an external AI without confidentiality concerns?"
- YES → type: "public"
- NO → type: "private"

When in doubt, mark as "private" — it's safer to assume sensitivity.

## CRITICAL: Effort Estimation Rules

Effort estimation is crucial. Get this right:

### Low Effort
ONLY mark as "low" effort if the item is **explicitly referenced or mentioned** somewhere in the data room:
- A pitch deck mentions "Phase 1 toxicity study completed" → the study likely exists, just upload it
- A document references "Patent Application No. XYZ" → the patent filing exists, just upload it
- A paper cites "binding assay showed Kd of 50nM" → the assay data exists, just upload it
- Team bios reference publications → those publications exist, just upload them

### Medium Effort
The team needs to do some work to compile, format, or produce this:
- Consolidating existing data into a report
- Writing up methods or protocols that exist but aren't documented
- Preparing summaries from multiple sources

### High Effort
**CAUTION:** If something is NOT referenced anywhere in the data room, DO NOT assume it can easily be uploaded!
- Missing experiment data that was never mentioned → the experiment may not have been done yet
- Safety data with no reference → studies may need to be conducted
- IP filings not mentioned → may require new patent work

Ask yourself: "Is there evidence this actually exists?" If not, it's likely high effort.

## Impact Based on Criterion Weight

Consider impact based on the criterion's weight:
- **high impact**: criterion weight >= 15%
- **medium impact**: criterion weight 10-14%
- **low impact**: criterion weight < 10%

## Selection & Ordering

You may generate todos for ANY criterion where score < 5. There is NO limit per criterion.

For EACH type (public and private), pick the 1-10 best todos by effort/impact ratio:
- **Best value**: low effort + high impact (these should be first)
- **Good value**: low effort + medium impact, OR medium effort + high impact
- **Moderate value**: medium effort + medium impact, OR low effort + low impact
- **Lower value**: high effort items (only include if they're critical)

## CRITICAL: Final Ordering by Actionability

As the LAST step, order the final todo lists (both public and private) by **immediate actionability**:

### Order Priority (first to last):

1. **Immediately actionable (upload existing)**: Items the team likely already has and just needs to upload
   - Documents referenced in the data room but not uploaded
   - Published papers, granted patents, existing presentations
   - Effort = "low", team just needs to find and upload

2. **Quick compilation/formatting**: Items requiring minimal work to produce
   - Consolidating existing data into a summary
   - Formatting existing protocols or methods
   - Effort = "low" to "medium", no new experiments needed

3. **Moderate work required**: Items requiring real but bounded effort
   - Writing new documentation
   - Preparing analyses from existing data
   - Effort = "medium"

4. **Complex work (experiments/filings)**: Items requiring substantial new work
   - New experiments, studies, or assays
   - New patent filings or regulatory submissions
   - Effort = "high", requires lab work, legal work, or significant time

This ordering ensures teams see quick wins at the top and can make immediate progress, with more complex items they can plan for later.

## Rules

1. Generate 1-10 PUBLIC todos AND 1-10 PRIVATE todos (at least one of each if any score < 5)
2. criterion_key MUST exactly match a key from the scores JSON
3. Be specific - reference actual files from the data room when mentioning referenced items
4. Order each group by effort/impact ratio (best value first)
5. Every todo MUST have a "type" field set to either "public" or "private"
6. Aim for a reasonable mix — most projects will have both public-facing gaps and private data gaps`

/**
 * Generates actionable todo items based on the project's data room contents,
 * TRL level, and scoring results.
 *
 * @param trlLevel - The TRL classification of the project
 * @param scoringResult - The scoring result from calculateWeightedScore
 * @param dataRoomContent - The formatted extracted documents text
 * @returns Array of todo items, or null if TRL level is not supported
 */
export async function generateTodos(
  trlLevel: TrlAnalysis["trl_classification"],
  scoringResult: ScoringResult,
  dataRoomContent: string,
): Promise<TodoItem[] | null> {
  if (!isScoringTrlLevel(trlLevel)) {
    console.log(`Todo generation not supported for ${trlLevel}, skipping...`)
    return null
  }

  console.log(`Generating todos for ${trlLevel}...`)

  // Debug: show which criteria need improvement
  const criteriaScores = scoringResult.scoring.criterion_scores
  const criteriaWithImprovementNeeded = Object.entries(criteriaScores)
    .filter(([, data]) => (data as { score: number }).score < 5)
    .map(([key, data]) => `${key}: ${(data as { score: number }).score}`)

  console.log(
    `Criteria needing improvement: ${criteriaWithImprovementNeeded.length > 0 ? criteriaWithImprovementNeeded.join(", ") : "none (all scores are 5)"}`,
  )

  if (criteriaWithImprovementNeeded.length === 0) {
    console.log("All criteria have perfect scores (5), no todos needed")
    return []
  }

  const scoringGuide = getTrlGuide(trlLevel)
  const scoresJson = JSON.stringify(
    scoringResult.scoring.criterion_scores,
    null,
    2,
  )

  // Build weight lookup from scores for explicit weight reference
  const weightLookup = Object.entries(criteriaScores)
    .map(([key, data]) => {
      const d = data as { weight: number; score: number }
      return `- ${key}: weight=${(d.weight * 100).toFixed(0)}%, current_score=${d.score}`
    })
    .join("\n")

  const prompt = `${TODO_GENERATION_PROMPT}

## Project

Current TRL: ${trlLevel}

## Criterion Weights and Current Scores

IMPORTANT: Use these weights to determine impact. Higher weight = more important for final score.

${weightLookup}

## Scoring Criteria Reference for ${trlLevel}

<scoring_rubric>
${scoringGuide}
</scoring_rubric>

## Current Scores (Full Details)

These are the scores from the most recent evaluation (remember: 5 is best, 1 is worst):

<scores>
${scoresJson}
</scores>

## Data Room Contents

These are all files and announcements currently in the project's data room.
READ CAREFULLY - look for items that are MENTIONED or REFERENCED but where the actual evidence is NOT uploaded.

<data_room>
${dataRoomContent}
</data_room>

## Your Task

1. SCAN the data room for referenced items that aren't uploaded:
   - Look for mentions of experiments, studies, patents, publications, data
   - If referenced but not uploaded → effort="low" (team likely has the item)
   - These are your highest-value todos

2. IDENTIFY other improvement opportunities for criteria where score < 5:
   - Be CAUTIOUS with items NOT referenced anywhere - they may require new work
   - If no evidence it exists → effort="high" (don't assume it can be easily uploaded)

3. CLASSIFY each todo as "public" or "private":
   - Public: non-sensitive documents an AI can review (published papers, pitch decks, granted patents, public presentations)
   - Private: sensitive/confidential data only the team can verify (raw experiment data, patent applications, proprietary formulations, internal safety studies)
   - When unsure, default to "private"

4. CHECK PROJECT SCOPE before including any todo:
   - Look for roadmap, milestones, timeline, or scope information in the data room
   - Only include todos that fit within the project's stated current phase and near-term plans
   - Do NOT suggest todos for future stages the project hasn't reached yet
   - Match todos to the current TRL level

5. SELECT the best 1-10 PUBLIC todos and 1-10 PRIVATE todos by effort/impact ratio:
   - Prioritize: low effort + high impact (high-weight criteria)
   - Then: low effort + medium impact, medium effort + high impact
   - Include high effort items only if critical AND within scope

6. ORDER final todo lists (both public and private) by ACTIONABILITY as the last step:
   - FIRST: Immediately actionable items (just upload existing documents)
   - THEN: Quick compilation/formatting tasks
   - THEN: Moderate work items
   - LAST: Complex work requiring experiments, new filings, or significant effort

Use the EXACT criterion keys from the scores JSON (e.g., "therapeutic_relevance", "intellectual_property", "utility_of_candidates").

You MUST generate at least 1 public AND at least 1 private todo if ANY criterion has a score < 5.`

  const { output } = await generateText({
    model: anthropic(REASONING_MODEL),
    prompt,
    output: Output.object({ schema: todoListSchema }),
  })

  console.log(`Todo generation complete: ${output.length} todos generated`)

  return output
}
