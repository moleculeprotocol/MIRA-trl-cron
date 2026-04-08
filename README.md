# MIRA TRL Assessment Pipeline

Automated [Technology Readiness Level](https://en.wikipedia.org/wiki/Technology_readiness_level) (TRL) assessment for biotech projects on the [Molecule](https://www.molecule.xyz/) platform.

MIRA fetches documents from a project's dataroom, uses AI to extract and analyze their content, classifies the project into a TRL stage, scores it against weighted criteria, and generates actionable improvement todos.

## How it works

The pipeline processes each project through five sequential AI steps:

```
Dataroom Documents (PDFs, images)
        |
        v
  1. Content Extraction         (Claude Sonnet 4.5)
        |
        v
  2. Questionnaire Filling      (Claude Opus 4.5)
        |
        v
  3. TRL Classification         (Claude Opus 4.5)
        |
        v
  4. Weighted Scoring            (Claude Opus 4.5)
        |
        v
  5. Todo Generation             (Claude Opus 4.5)
```

### Step 1: Content Extraction

Each public document (PDF, PNG, JPEG, GIF, WebP) is downloaded and sent to Claude Sonnet 4.5 with a prompt focused on extracting scientifically relevant content. The model returns structured output containing:

- **Document type** (pitch deck, experiment report, publication, etc.)
- **Relevance** to TRL assessment (high / medium / low / none)
- **Content blocks** organized by page, each typed as text, figure, table, or diagram
- **Key findings** for TRL assessment

Extractions are cached to disk as JSON files keyed by a SHA-256 hash of the file's identity (`did:version:path`). On subsequent runs, unchanged files are loaded from cache instead of re-extracted.

All extracted documents are then merged into a single formatted text that preserves file boundaries, page structure, and block types. Documents with `relevance: "none"` are excluded.

### Step 2: Questionnaire Filling

The merged content is sent to Claude Opus 4.5 with a biotech project questionnaire covering:

- **Project overview** -- title, therapeutic concept, target indication, patient population, mechanism of action, biological target, development timeline
- **Experimental evidence** -- lab work status, experiment types, results, model systems (in vitro, advanced in vitro, in vivo), efficacy evidence, dose-response data, reproducibility, safety observations
- **Team information** -- key personnel, affiliations, expertise
- **Intellectual property** -- patent status, scope of protection, freedom-to-operate

The model fills each field based on what it found in the documents. Fields that cannot be determined are marked "Not specified".

### Step 3: TRL Classification

The completed questionnaire is sent to Claude Opus 4.5 with a systematic decision tree that classifies the project into one of five levels:

| Level | Name | Definition |
|-------|------|------------|
| **Pre-TRL 1** | No coherent hypothesis | Project lacks a coherent scientific hypothesis |
| **TRL 1** | Concept Alone | Purely theoretical -- no experimental evidence exists |
| **TRL 2** | Speculative Research | Some experiments conducted, but only in simplified systems (e.g., basic in vitro) |
| **TRL 3** | Early Evidence | Proof-of-concept demonstrated in disease-relevant models (in vivo or advanced in vitro) |
| **TRL >3** | Beyond scope | IND-enabling studies, GMP manufacturing established |

The decision tree follows three questions:

1. Has **any** experimental lab work been conducted? No -> TRL 1
2. Have experiments been conducted in **disease-relevant in vivo models** or advanced disease-modeling systems? No -> TRL 2
3. Do results demonstrate **early proof-of-concept efficacy** in disease models? No -> TRL 2, Yes -> TRL 3

When ambiguous, the **lower TRL** is assigned (conservative principle).

The output includes: `trl_classification`, `confidence` (high/medium/low), `rationale`, and `key_indicators`.

### Step 4: Weighted Scoring

Only projects classified as TRL 1, 2, or 3 are scored (Pre-TRL 1 and TRL >3 are skipped).

Each TRL level has a different set of criteria with different weights. The raw project data is sent to Claude Opus 4.5 along with the TRL-specific scoring rubric.

**TRL 1** (3 criteria):

| Criterion | Weight |
|-----------|--------|
| Therapeutic Relevance of Mechanism of Action | 60% |
| Therapeutic Optionality | 15% |
| Intellectual Property | 25% |

**TRL 2** (5 criteria):

| Criterion | Weight |
|-----------|--------|
| Therapeutic Relevance of Mechanism of Action | 45% |
| Therapeutic Optionality | 5% |
| Intellectual Property | 15% |
| Utility of Candidates | 20% |
| Prospects for Safety | 15% |

**TRL 3** (9 criteria):

| Criterion | Weight |
|-----------|--------|
| Therapeutic Relevance of Mechanism of Action | 20% |
| Therapeutic Optionality | 5% |
| Intellectual Property | 15% |
| Utility of Candidates | 20% |
| Prospects for Safety | 15% |
| Prospects for GMP/CMC for IND Filing | 5% |
| Prospects for Clinical Development | 10% |
| Commercial Potential | 5% |
| Organization and Team Fit | 5% |

Each criterion is scored 1-5 (5 = strongest), multiplied by its weight, and summed into a `final_weighted_score` (max 5.0). The model also produces a `score_interpretation` summary.

### Step 5: Todo Generation

For any criterion that scored below 5, the pipeline generates actionable improvement items. The model receives the full scoring context plus the raw dataroom content and produces 2-20 todos.

Each todo contains:

| Field | Description |
|-------|-------------|
| `criterion_key` | The scoring criterion this addresses |
| `title` | Short actionable title (max 80 chars) |
| `description` | 2-3 sentences referencing specific dataroom files |
| `effort` | `low` (upload existing), `medium` (compile/format), `high` (new experiments/filings) |
| `specificity` | Exactly what document or evidence to upload |
| `type` | `public` (non-sensitive, AI-reviewable) or `private` (confidential, team-verified only) |

Key generation rules:

- **Effort is evidence-based**: "low" only if the item is explicitly referenced in the dataroom. If there's no evidence it exists, it's "high".
- **Impact maps to criterion weight**: >=15% = high impact, 10-14% = medium, <10% = low.
- **Scope-aware**: Todos respect the project's stated roadmap and current development phase. A TRL 1 project won't get todos about GMP manufacturing.
- **Ordered by actionability**: Immediately uploadable items first, complex new work last.

## Project structure

```
src/index.ts          Main entry point and pipeline orchestrator
lib/config.ts         Constants: model IDs, API endpoints, contract addresses, test projects
lib/llm.ts            All AI logic: extraction, questionnaire, classification, scoring, todos
lib/molecule.ts       Molecule GraphQL API client (fetches projects and dataroom files)
lib/sanity.ts         Sanity CMS integration (storage and change detection)
output/               Cached extraction results, organized by project ID
```

## Tech stack

| Dependency | Purpose |
|------------|---------|
| [Vercel AI SDK](https://sdk.vercel.ai/) (`ai`) | Unified interface for LLM calls with structured output via Zod schemas |
| [@ai-sdk/anthropic](https://sdk.vercel.ai/providers/ai-sdk-providers/anthropic) | Anthropic provider -- Claude Sonnet 4.5 for extraction, Claude Opus 4.5 for reasoning |
| [Zod](https://zod.dev/) v4 | Schema validation for all structured LLM outputs |
| TypeScript | Strict mode, ES2021, NodeNext modules |
| Node.js 22 | Runtime |
| pnpm | Package manager |

## Running locally

```bash
pnpm install
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=...
MOLECULE_API_KEY=...
SANITY_API_TOKEN=...
```

Run against the hardcoded test projects:

```bash
pnpm dev
```

Run against all projects from the Molecule API:

```bash
pnpm dev:all
```

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude models |
| `MOLECULE_API_KEY` | Yes | API key for Molecule's GraphQL endpoint |
| `SANITY_API_TOKEN` | Yes | Authentication token for Sanity CMS |
| `DISCORD_WEBHOOK_URL` | No | Discord webhook for notifications (production only) |
| `SANITY_STUDIO_URL` | No | Base URL for Sanity Studio (required for Discord notification links) |
| `ENVIRONMENT` | No | `staging` or `production` -- controls Sanity dataset |
| `USE_ALL_PROJECTS` | No | `true` to fetch all projects; otherwise uses test projects |
| `FORCE_PROCESS` | No | `true` to skip change detection and reprocess everything |

## Scheduled execution

A GitHub Actions workflow (`.github/workflows/trl-assessment.yml`) runs the pipeline daily at 08:00 UTC. It processes all projects, commits updated extraction caches back to the repo, and can also be triggered manually with configurable options for environment, project scope, and force-processing.
