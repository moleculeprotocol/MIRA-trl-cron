# MIRA TRL Assessment Pipeline

Automated [Technology Readiness Level](https://en.wikipedia.org/wiki/Technology_readiness_level) (TRL) assessment for biotech projects on the [Molecule](https://www.molecule.xyz/) platform.

MIRA fetches documents from a project's dataroom, uses AI to extract and analyze their content, classifies the project into a TRL stage, scores it against weighted criteria, and generates actionable improvement todos.

The repo also holds a second, unrelated tool: the [MIRA chat report](#mira-chat-report) (`pnpm report`), a bi-weekly summary of the MIRA chat assistant built from Langfuse data.

## How it works

The pipeline processes each project through five sequential AI steps:

```
Dataroom Documents (PDFs, images)
        |
        v
  1. Content Extraction         (Claude Sonnet 5.5)
        |
        v
  2. Questionnaire Filling      (Claude Opus 5.5)
        |
        v
  3. TRL Classification         (Claude Opus 5.5)
        |
        v
  4. Weighted Scoring            (Claude Opus 5.5)
        |
        v
  5. Todo Generation             (Claude Opus 5.5)
```

### Step 1: Content Extraction

Each public document (PDF, PNG, JPEG, GIF, WebP) is downloaded and sent to Claude Sonnet 5.5 with a prompt focused on extracting scientifically relevant content. The model returns structured output containing:

- **Document type** (pitch deck, experiment report, publication, etc.)
- **Relevance** to TRL assessment (high / medium / low / none)
- **Content blocks** organized by page, each typed as text, figure, table, or diagram
- **Key findings** for TRL assessment

Extractions are cached to disk as JSON files keyed by a SHA-256 hash of the file's identity (`did:version:path`). On subsequent runs, unchanged files are loaded from cache instead of re-extracted.

All extracted documents are then merged into a single formatted text that preserves file boundaries, page structure, and block types. Documents with `relevance: "none"` are excluded.

### Step 2: Questionnaire Filling

The merged content is sent to Claude Opus 5.5 with a biotech project questionnaire covering:

- **Project overview** -- title, therapeutic concept, target indication, patient population, mechanism of action, biological target, development timeline
- **Experimental evidence** -- lab work status, experiment types, results, model systems (in vitro, advanced in vitro, in vivo), efficacy evidence, dose-response data, reproducibility, safety observations
- **Team information** -- key personnel, affiliations, expertise
- **Intellectual property** -- patent status, scope of protection, freedom-to-operate

The model fills each field based on what it found in the documents. Fields that cannot be determined are marked "Not specified".

### Step 3: TRL Classification

The completed questionnaire is sent to Claude Opus 5.5 with a systematic decision tree that classifies the project into one of five levels:

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

Each TRL level has a different set of criteria with different weights. The raw project data is sent to Claude Opus 5.5 along with the TRL-specific scoring rubric.

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

## MIRA chat report

A second, independent tool lives in `src/mira-report/`. It has nothing to do with TRL assessment: it summarises how the MIRA chat assistant performed over a time window. It reads traces, scores, and costs from [Langfuse](https://langfuse.com/), aggregates them, asks an LLM to group the user messages into themes, and prints the result or posts it to Slack.

```bash
pnpm report                        # last 14 full UTC days, printed to stdout
pnpm report --slack                # post to Slack instead of printing
pnpm report --days 7               # different window length
pnpm report --from 2026-09-01 --to 2026-09-15
pnpm report --no-llm               # skip the themes step (no Anthropic call)
pnpm report --dump raw.json        # also write the fetched Langfuse data
pnpm report --from-dump raw.json   # re-run from a dump, without Langfuse
pnpm report --help
```

The report has five sections:

| Section | Content |
|---------|---------|
| Headline numbers | Conversations, messages, median messages per conversation, errors, latency p50/p95, MIRA cost, judge cost -- each compared with the previous window of equal length |
| Topics | Share of classified messages per topic category, plus the most common topic sets per conversation |
| Answer quality | On-topic and all-scored answer rates, plus grouped unanswered needs |
| Scope | Off-topic requests and requests that need a capability MIRA lacks |
| Themes | 3-5 LLM-generated themes, missing capabilities, and paraphrased example questions |

The themes step (`src/mira-report/insights.ts`) is the only LLM call. It uses the model named in `INSIGHTS_MODEL` (currently `claude-sonnet-4-6`), caps input at 60k characters, samples sessions with a deterministic PRNG so re-runs over the same dump are stable, and filters example questions for verbatim text and PII. Metric names expected in Langfuse (observation name, score names, topic categories) are pinned in `src/mira-report/constants.ts` with no fallbacks -- if something is renamed in Langfuse, change it there.

## Project structure

```
src/index.ts          Main entry point and pipeline orchestrator
src/mira-report/      MIRA chat report: Langfuse -> aggregate -> LLM themes -> console | Slack
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
| [@ai-sdk/anthropic](https://sdk.vercel.ai/providers/ai-sdk-providers/anthropic) | Anthropic provider -- Claude Sonnet 5.5 for extraction, Claude Opus 5.5 for reasoning, `INSIGHTS_MODEL` for the chat report |
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

# Only needed for `pnpm report` (the MIRA chat report)
INSIGHTS_MODEL=claude-sonnet-4-6
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_ENVIRONMENT=mira-v1-production
SLACK_BOT_TOKEN=...
SLACK_CHANNEL_ID=...
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
| `SLACK_BOT_TOKEN` | No | Slack bot user OAuth token (`xoxb-...`) for notifications (production only) |
| `SLACK_CHANNEL_ID` | No | Slack channel ID (`C...`) to post notifications to |
| `SANITY_STUDIO_URL` | No | Base URL for Sanity Studio (required for Slack notification links) |
| `ENVIRONMENT` | No | `staging` or `production` -- controls Sanity dataset |
| `SANITY_PUBLISH_IMMEDIATELY` | No | `true` to write TRL/scoring data directly to the published document instead of a draft |
| `OCL_ID` | No | Process only this single OCL ID; overrides `USE_ALL_PROJECTS` when set |
| `USE_ALL_PROJECTS` | No | `true` to fetch all projects; otherwise uses test projects (ignored when `OCL_ID` is set) |
| `FORCE_PROCESS` | No | `true` to skip change detection and reprocess everything |

The MIRA chat report (`pnpm report`) uses its own set. There are no defaults: a missing value fails before any network call.

| Variable | Required | Description |
|----------|----------|-------------|
| `LANGFUSE_PUBLIC_KEY` | Yes | Langfuse API public key |
| `LANGFUSE_SECRET_KEY` | Yes | Langfuse API secret key |
| `LANGFUSE_BASE_URL` | Yes | Langfuse host, e.g. `https://cloud.langfuse.com` |
| `LANGFUSE_ENVIRONMENT` | Yes | Langfuse environment to read, e.g. `mira-v1-production` |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key (not needed with `--no-llm`) |
| `INSIGHTS_MODEL` | Yes | Model for the themes step, e.g. `claude-sonnet-4-6` (not needed with `--no-llm`) |
| `SLACK_BOT_TOKEN` | Only with `--slack` | Slack bot user OAuth token (`xoxb-...`) |
| `SLACK_CHANNEL_ID` | Only with `--slack` | Slack channel ID (`C...`) to post the report to |

Not needed when running from a dump (`--from-dump`): the Langfuse variables.

## Scheduled execution

Two GitHub Actions workflows run on a schedule:

| Workflow | Schedule | What it does |
|----------|----------|--------------|
| `.github/workflows/trl-assessment.yml` | Daily, 08:00 UTC | Runs the TRL pipeline over all projects, commits updated extraction caches back to the repo |
| `.github/workflows/mira-report.yml` | Mondays, 08:00 UTC, even ISO weeks only | Builds the MIRA chat report for the last 14 days and posts it to Slack |

Both can be triggered manually. The TRL workflow takes options for environment, project scope, and force-processing. The report workflow takes a window length, a flag to print to the job log instead of posting to Slack, and a flag to ignore the even-week check (cron cannot express "every 14 days", so the first step skips odd ISO weeks). If the report fails, the workflow posts a failure notice to the same Slack channel.
