# Financial AI Pipeline

![Badge](https://img.shields.io/badge/Python-3.11%2B-blue) ![Badge](https://img.shields.io/badge/LangGraph-StateGraph-orange) ![Badge](https://img.shields.io/badge/LLM-DeepSeek%20(OpenAI--compatible)-green) ![Badge](https://img.shields.io/badge/Tests-480%2B%20passing-brightgreen)

> **A self-correcting LLM workflow that turns unstructured financial announcements (M&A, dividends) into strictly-typed, versioned, fully-traceable records. The flow is fixed in code (a LangGraph state machine); two steps call the LLM and code checks each one. Every document ends in exactly one of four places, and a CI scenario gate checks each path.**

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Architecture](#-architecture)
- [Key Design Features](#-key-design-features)
- [Technologies](#-technologies)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [Testing & CI](#-testing--ci)

---

## 🎯 Overview

Financial institutions ingest hundreds of unstructured announcements daily — SEC 8-K filings, dividend declarations, M&A press releases — each with different phrasing and no fixed template. Regex- and template-based parsers can't generalize across that variety, don't self-correct, and produce outputs that can't survive an audit.

This pipeline combines an LLM extraction step with a deterministic, self-correcting state machine (LangGraph) to produce structured, versioned, traceable extraction results from raw `.txt` announcements:

- **Reader → Classifier → Classification Validator → Extractor → Extraction Validator**, each an independent LangGraph node reading/writing a shared `PipelineState`. Only the Classifier and the Extractor call an LLM; the Reader and both validators are plain code
- **Two self-correction loops, each with its own retry budget**: (1) *classify* — an unparsable label, or an evidence quote that does not occur in the source text, sends the document back to the Classifier with the error type as feedback (`classify_max_retries`, default 2); (2) *extract* — output that doesn't parse into the schema, or a value that is present but invalid, routes back to the Extractor (`max_retries`, default 3). **Retry policy** (`src/schemas/retry_policy.py`): a missing required value (`missing_info`) or a broken cross-field rule (`business_rule`) is never retried, because the feedback would push the model to invent a value; it goes to human review. Transient API failures are retried by the LLM client (SDK backoff, `llm_max_retries`=2, 60 s timeout), not by these loops. When a budget is spent, the failure's category decides where the document goes (see *Four outcomes* below)
- **Classifier labels**: `M&A`, `Dividend`, `none` (clearly neither) and `unclear` (cannot decide). `none` / `unclear` end the document at the classification stage, so no extraction is spent on them: `none` is out of scope (no queue), `unclear` goes to human review
- **Stage-tagged failures**: every failure records its stage, error code and category (plus the classifier output and the per-cycle correction history) in the DLQ or the review queue, so failures can be counted per stage and code instead of grepping error text
- **Four outcomes, two queues**: one rule (`src/schemas/outcome.py::outcome_of`) sends every run to exactly one place: `extractions` (stored), `review_queue` (a *person* decides: `unclear`, a missing REQUIRED field, a broken business rule, a quote never found), `dead_letter_queue` (an *engineer* acts: infrastructure, input, bugs, output that never parsed), or no queue for out-of-scope filings. `documents.last_status` records the outcome; evaluation uses the same function
- **Two-model data isolation**: the LLM only ever fills business fields (`MandaExtraction`, `DividendExtraction`); the pipeline itself owns provenance (`doc_id`, `Metadata`, audit trail) so the model can never corrupt controlled fields
- **SHA-256 idempotency + SCD Type 2 versioning**: re-running on unchanged content is a no-op; re-running on an edited file archives the old version and inserts a new one — nothing is ever overwritten
- **Optional RAG-augmented extraction**: validated past extractions are embedded (SentenceTransformer, local) and stored in ChromaDB per document type, then retrieved as few-shot context on first-pass extraction. **Off by default** (`RAG_ENABLED=false`): with RAG on, a batch run feeds earlier documents' results to later ones, so results depend on processing order — it stays off for evaluation, and its effect on accuracy has not been measured.
- **CI scenario gate**: one synthetic case per pipeline path, run end to end with the real model; each case must reach its expected outcome (stored / out of scope / human review / DLQ) and stored cases must match on every REQUIRED field. Outcome and field accuracy are gated separately, never pooled

Currently supports two document types (**M&A**, **Dividend**) via a registry pattern — adding a new type requires a new schema file and a few dict entries, no changes to graph topology or node logic.

---

## 🏗️ Architecture

```
source *.txt
      │
      ▼
 ┌────────────────┐
 │ Reader         │  code: stores the bytes + SHA-256; undecodable input → DLQ
 └──────┬─────────┘
        ▼
 ┌────────────────┐
 │ Classifier     │  LLM → "M&A" | "Dividend" | "none" | "unclear"
 │                │        + a verbatim evidence_quote
 └──────┬─────────┘
        ▼                         ┌─ bad format / quote not in the text
 ┌────────────────┐               │  → retry the Classifier with feedback (max 2)
 │ Classification │───────────────┘
 │ Validator      │  code, no LLM
 └──────┬─────────┘
        │ M&A / Dividend          (none → out of scope · unclear → human review)
        ▼
 ┌────────────────┐    ┌──────────────────┐
 │ Extractor      │◄───│ RAG retriever    │  optional, off by default
 │                │    │ (ChromaDB)       │
 └──────┬─────────┘    └──────────────────┘
        ▼                         ┌─ bad format / a present but invalid value
 ┌────────────────┐               │  → retry the Extractor with feedback (max 3)
 │ Extraction     │───────────────┘
 │ Validator      │  code: REQUIRED fields, formats, cross-field rules
 └──────┬─────────┘
        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ outcome_of(state): one rule, from the failure category       │
 │ (any step above can stop early and land here)                │
 └───┬────────────────┬─────────────────┬────────────────────┬──┘
     ▼                ▼                 ▼                    ▼
  stored         human review      out of scope       dead-letter queue
  extractions    review_queue      status only,       dead_letter_queue
  (gold, SCD2)   a person decides  no queue           an engineer acts
```

Every run is persisted in a medallion layout — passed **and** failed documents alike:

| Layer | What | Where |
|-------|------|-------|
| Bronze (raw) | The original file **bytes**, stored before decoding, written once and never modified, named by their SHA-256 (matches `sha256sum`). Decoding: UTF-8 → Windows-1252 fallback (recorded); undecodable or binary files go to the DLQ as `read` / `DECODE_FAILED` instead of being skipped | `data/raw/<sha256>.txt` + `documents` table (registry: `raw_uri`, size, encoding, first/last seen, last run, last status) |
| Silver (staging) | One `step_runs` row per node execution — every classifier / extractor call and every validation, passed, rejected or fatal: raw LLM reply, parsed output, verdict, error code, model, prompt version (hash of the exact system message), tokens, latency. Plus `dead_letter_queue` (engineering), `review_queue` (human decisions) and `correction_logs` | `data/pipeline.db` |
| Gold (curated) | Validated extractions only, SCD Type 2 versioned; each row carries `run_id` and `source_hash`, linking it to its step trail and its exact source text | `data/pipeline.db` → `extractions` |

There is no LLM-as-judge yet, so "gold" means *passed the deterministic validators*, not *verified correct*.

Failure stages: `read`, `classify`, `classify_validate`, `extract`, `validate` (and `pipeline` for a crash outside the graph). Categories: `transient`, `config_error`, `structure`, `reference`, `missing_info`, `business_rule`, `out_of_scope`, `ambiguous`, `input_error`, `internal_error`, `unknown`. The category decides both the retry (`src/schemas/retry_policy.py`) and the outcome (`src/schemas/outcome.py`). Older databases are migrated in place (`ALTER TABLE … ADD COLUMN`) the next time `init_db()` runs; rows written before the migration keep `NULL` in the new columns.

**See [DOCS_SYSTEM_DESIGN.md](./DOCS_SYSTEM_DESIGN.md) for the full system design white paper**, including the state schema, the two-model isolation protocol, the idempotency guard implementation, RAG integration details, and the CI evaluation engine.

---

## ✨ Key Design Features

| Feature | Implementation |
|---------|---------------|
| **State-as-Graph** | Every node reads/writes a single `PipelineState` `TypedDict` compiled via LangGraph's `StateGraph` |
| **Two-Model Isolation** | LLM-facing `BaseModel` (business fields only, all `Optional`) is promoted to a pipeline-owned `BaseDoc` (with `doc_id` + `Metadata`) via a `from_extraction()` classmethod — the LLM never touches provenance fields |
| **Idempotent Persistence** | SHA-256 hash of raw file content is the true dedup key (not the per-run UUID `doc_id`) — re-running on unchanged content is a guaranteed no-op |
| **Immutable Version History** | SCD Type 2 (`version` + `is_current` columns) — file edits archive the old row and insert a new version, never overwrite |
| **Self-Correction Loops** | Conditional LangGraph edges route a rejected classification back to the Classifier and a failed extraction back to the Extractor, each with the validator's structured error feedback injected into the next prompt and each with an independent retry budget |
| **Evidence-checked Classification** | The Classifier must return a verbatim `evidence_quote`; plain code checks that it occurs in the source text (whitespace-, case- and typographic-quote-insensitive). The model's self-reported `confidence` is logged but never used for routing |
| **Stage-tagged failures** | Every fatal exit carries a `failure` record (stage, code, category); the DLQ or the review queue stores it in columns together with the classifier output and the full correction history |
| **One routing rule** | `outcome_of` maps every run to stored / human review / out of scope / DLQ from the failure category; storage and evaluation call the same function |
| **Retry by failure category** | Format and quote errors are retried with feedback; a missing REQUIRED value or a broken business rule is never retried (feedback pushes the model to invent a value) and goes to a person |
| **Field presence levels** | REQUIRED / EXPECTED / OPTIONAL per field in the validation profiles: only REQUIRED fields may not be empty, an empty EXPECTED field is stored with a warning |
| **Zero-Code Extension** | Adding a document type = 1 new schema file + 3 registry dict entries in `src/nodes/extractor.py` — no changes to graph topology or node logic (optional extras: validator rules, a RAG collection in `COLLECTION_MAP`, and `FIELD_TYPE_MAP` for the scenario gate) |
| **RAG-Augmented Extraction** | ChromaDB collection per document type; embeddings via local `SentenceTransformer` (`all-MiniLM-L6-v2`); only validated extractions are embedded; superseded versions are de-indexed and a file is never retrieved as its own example; first-pass extraction only (retries reuse the same context). Off by default (`RAG_ENABLED=false`) |
| **Async Concurrency** | `asyncio.Semaphore` + `asyncio.gather` — batch mode runs each file's LangGraph state machine independently and concurrently |
| **Operation Visibility** | One Streamlit app (`make dash`, `src/app.py`) with three pages: the E4 end-to-end funnel, E1 runs, and the pipeline database (four outcomes, the human review queue, the DLQ, per-`doc_id` self-correction audit drill-downs) |
| **CI Scenario Gate** | `evals/run_scenarios.py` runs one synthetic case per pipeline path end to end; the build fails if any case misses its expected outcome or a REQUIRED field |

---

## 🛠️ Technologies

### Orchestration & LLM
- **LangGraph** — state machine orchestration (`StateGraph`, conditional edges)
- **OpenAI Python SDK** (`AsyncOpenAI`) — LLM client, pointed at DeepSeek's OpenAI-compatible endpoint
- **DeepSeek** (OpenAI-compatible API) — extraction + classification model

### Data Validation
- **Pydantic v2** — two-model schema contracts, field validators
- **Pydantic Settings** — environment-driven configuration

### Storage & Async
- **aiosqlite** — async SQLite (WAL mode)
- **aiofiles** — async file I/O

### RAG
- **ChromaDB** — vector store, per-doc-type collections
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embeddings

### Tooling & Ops
- **Streamlit** + **Plotly** — dashboards (pipeline database, E1 and E4 runs)
- **loguru** — structured logging
- **pytest** / **pytest-asyncio** / **pytest-cov** — test suite
- **black** — formatting, enforced in CI
- **GitHub Actions** — CI pipeline (unit tests → scenario gate → lint)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- A DeepSeek API key (or any OpenAI-compatible endpoint)

### Installation

```bash
git clone https://github.com/ChloeA27/financial-ai-pipeline.git
cd financial-ai-pipeline

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt            # runtime dependencies
pip install -r requirements-dev.txt        # pytest + black, used in CI
pip install -r requirements-dashboard.txt  # optional: Streamlit monitoring dashboard

cp .env.example .env
# edit .env and set DEEPSEEK_API_KEY
```

### Run on a single file

```bash
python -m src.main --file evals/datasets/scenarios/inputs/ma_microsoft_activision.txt
```

### Run on a directory (concurrent batch mode)

```bash
python -m src.main --dir evals/datasets/scenarios/inputs --concurrency 5
```

### Open the dashboards

Requires the optional dashboard dependencies (`pip install -r requirements-dashboard.txt`).

```bash
make dash
# → http://localhost:8501, pages: E4 end to end · E1 classification · pipeline database
```

Other shortcuts: `make test` (offline tests), `make scenarios` (scenario gate, costs tokens), `make e4-estimate` (E4 case count and token estimate, no LLM).

### Run the test suite

```bash
pytest -m "not integration" --tb=short -q
```

### Run the scenario gate (real LLM, end to end)

```bash
python evals/run_scenarios.py
```

### Run the classification-stage evaluation (E1)

Compares classifier input strategies on a frozen list of labelled SEC EDGAR filings (offline paired comparison). The dataset is kept outside this repo and only read; results are written under `results/e1/`. The test split is refused until the prompt and input strategy are final.

Point the runner at the dataset once in `.env` (`E1_DATASET=<E1>/e1_dataset_draft.jsonl`), or pass `--dataset` on each call:

```bash
# no LLM calls: is the gold evidence inside each strategy's window?
python evals/run_e1.py --coverage-only
# no LLM calls: planned calls and tokens
python evals/run_e1.py --estimate
# real run: head / structured / keyword, 2 repetitions, temperature 0
python evals/run_e1.py --reps 2 --max-calls 630
```

Browse a run (variant comparison, outcome funnel, per-document trace with the classifier's window and the model's evidence quote highlighted). Served on localhost only (`.streamlit/config.toml`); cases that are not human-verified are hidden in the trace unless you opt in:

```bash
make dash   # then open the "E1 · classification" page
```

Results go to `results/e1/<UTC timestamp>_<tag>/` (`manifest.json` with dataset/input hashes, model, temperature, git commit; `summary.json`; per-case CSV). Gold labels are projected onto the two supported types (`mgmt_change`-only filings count as `none`). Per-case files never show draft labels of cases that are not yet human-verified.

### Run the whole pipeline on the E1 dev split (E4)

```bash
python -m evals.run_e4 --estimate                          # no LLM: cases and token estimate
python -m evals.run_e4 --tag dev --model deepseek-flash    # real run
```

Each document's outcome (stored / human review / out of scope / DLQ) is reported per label, separately for human-verified and draft labels, in `results/e4/<UTC timestamp>_<tag>/`. Field values are not scored here; that is E2 (`evals/E2_DESIGN.md`, designed, not built).

---

## 📁 Project Structure

```
financial-ai-pipeline/
├── DOCS_SYSTEM_DESIGN.md         ← Full system design white paper
├── DOCS_REVIEW_WRITEBACK.md      ← Proposal: human review decisions → gold (not decided)
├── README.md                     ← You are here
├── Makefile                      ← make dash / test / scenarios / e4-estimate
│
├── evals/                         ← probabilistic evaluation (see evals/README.md)
│   ├── datasets/scenarios/        ← inputs/ + cases/ (one synthetic case per pipeline path)
│   ├── datasets/e1/README.md      ← E1 classification set lives outside the repo
│   ├── evaluators/                ← scenarios.py (outcome + field gates), e1.py (classification), e4.py (end to end)
│   ├── analysis/field_presence.py ← how often fields are mentioned in real filings (no LLM)
│   ├── run_scenarios.py           ← CI scenario gate: end to end, real LLM
│   ├── run_e1.py                  ← classification-stage experiments
│   ├── run_e4.py                  ← whole pipeline on the E1 dev split
│   └── E2_DESIGN.md               ← field-level extraction evaluation (design only)
│
├── results/
│   ├── SUMMARY.md                 ← Project state + latest measured numbers
│   ├── e1/                        ← One folder per E1 run (index: results/e1/README.md)
│   ├── e4/                        ← One folder per E4 run (index: results/e4/README.md)
│   ├── scenarios/                 ← One folder per local scenario-gate run
│   └── field_presence/            ← Field mention rates in real filings (no LLM)
│
├── .github/workflows/
│   └── ci.yml                     ← CI: unit tests → scenario gate → black lint
├── .streamlit/config.toml         ← Dashboards bind to localhost only
│
├── src/
│   ├── main.py                    ← CLI entry point (--file / --dir)
│   ├── config.py                  ← Pydantic Settings (env-driven)
│   ├── app.py                     ← one Streamlit entry point for all pages (make dash)
│   ├── dashboard.py               ← page: pipeline database (four outcomes, review queue, DLQ)
│   ├── e1_dashboard.py            ← page: E1 runs (results/e1/)
│   ├── e4_dashboard.py            ← page: E4 runs (results/e4/)
│   │
│   ├── workflows/edges.py         ← routing functions (where to go next)
│   ├── workflows/graph.py         ← wiring: nodes + edges → compiled StateGraph
│   │
│   ├── nodes/
│   │   ├── reader.py               ← async file I/O + SHA-256 hashing
│   │   ├── classifier.py           ← LLM doc-type label + evidence quote
│   │   ├── input_window.py         ← classifier input strategies (head / structured / keyword)
│   │   ├── classification_validator.py ← code checks on the classification (no LLM)
│   │   ├── extractor.py            ← LLM extraction + registry pattern
│   │   ├── extraction_validator.py ← rule-based quality gate for extraction results
│   │   ├── retry.py                ← shared retry bookkeeping (classify + extract loops)
│   │   └── step_runs.py            ← per-execution audit records
│   │
│   ├── schemas/                    ← contracts (no logic depends on a node)
│   │   ├── state.py                ← PipelineState (graph state, reducers)
│   │   ├── events.py               ← StepRun / FailureInfo / CorrectionLog audit records
│   │   ├── failure.py              ← failure vocabulary (stages, categories, codes)
│   │   ├── outcome.py              ← outcome_of: stored / human review / out of scope / DLQ
│   │   ├── retry_policy.py         ← which failure categories may be retried
│   │   ├── classification.py       ← labels + ClassificationResult (LLM output)
│   │   ├── base.py                 ← Metadata + BaseDoc
│   │   └── extraction/             ← per-doc-type two-model contracts + field presence levels
│   │
│   ├── rag/                        ← embedder, ChromaDB client, retriever (off by default)
│   ├── llm/client.py                ← OpenAI-SDK LLM client (explicit retries + timeout)
│   ├── llm/errors.py                ← final API error → config_error / input_error / transient
│   ├── prompts/                     ← prompt assets: templates/*.md + registry.py loader
│   │
│   ├── utils/hashing.py             ← SHA-256 content hash (shared key)
│   ├── viz/                         ← data shaping for the dashboard pages (no Streamlit, unit-tested)
│   │
│   └── storage/
│       ├── db.py                    ← aiosqlite schema DDL + in-place migrations
│       ├── raw_store.py             ← bronze: write-once content-addressed source texts
│       └── repository.py            ← routes by outcome: gold / review_queue / DLQ + step_runs
│
└── tests/                          ← unit + integration tests, mirrors src/
    ├── test_nodes/  test_workflows/  test_storage/  test_llm/  test_rag/
    ├── test_schemas/  test_prompts/  test_eval/  test_cli/  test_viz/
    ├── fake_embedding.py           ← offline embedder used by the test suite
    └── conftest.py                 ← markers + per-test vector-store / raw-store isolation
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [DOCS_SYSTEM_DESIGN.md](./DOCS_SYSTEM_DESIGN.md) | Full system design white paper — graph topology, two-model isolation, idempotency defense, SCD2 versioning, self-correction loop, RAG architecture, CI scenario gate, concurrency model |
| [results/SUMMARY.md](./results/SUMMARY.md) | Current state of the project, latest measured evaluation numbers, open issues |
| [evals/README.md](./evals/README.md) | Evaluation suite: datasets, evaluators, how to run the scenario gate and E1 |
| [results/e1/README.md](./results/e1/README.md) | Index of E1 evaluation runs and how a run folder is laid out |
| [results/e4/README.md](./results/e4/README.md) · [results/scenarios/README.md](./results/scenarios/README.md) | Index of end-to-end runs and scenario-gate runs |
| [evals/E2_DESIGN.md](./evals/E2_DESIGN.md) | Design of the field-level extraction evaluation (not built) |
| [DOCS_REVIEW_WRITEBACK.md](./DOCS_REVIEW_WRITEBACK.md) | Proposal for writing human review decisions to gold (open questions) |

---

## 🧪 Testing & CI

- **480+ unit tests** across nodes, schemas, storage, graph routing, the RAG layer, the dashboards' data layer and the evaluation code (`pytest -m "not integration"` or `make test`, no LLM calls; 4 more `integration` tests call the real API). Each test gets a private vector store and an offline fake embedder; add `and not real_embedder` to skip the few tests that load the real model: `pytest -m "not integration and not real_embedder"`
- **Scenario gate** (`evals/run_scenarios.py`) runs one synthetic case per pipeline path end to end with the real model. It gates on the expected outcome and, for stored cases, on every REQUIRED field (field-type-aware comparison: `enum`, `numeric` ±1 %, `bool`, `date`, `text_lower`; per-field overrides `date_range`, `fuzzy_match`, `accept_null`). It is a regression gate, not a quality measurement: quality on real filings is measured by E1
- **GitHub Actions CI** (`.github/workflows/ci.yml`) runs three sequential gates on every push to `main` and every pull request into it: unit tests → scenario gate → `black` lint check. The scenario gate needs the `DEEPSEEK_API_KEY` secret: it is skipped (with a warning) on fork PRs, but a push without the secret fails the build

---

## 👤 Author

**Qi An (Chloe)**
- GitHub: [@ChloeA27](https://github.com/ChloeA27)
- LinkedIn: [Qi An](https://www.linkedin.com/in/qianchloe/)
