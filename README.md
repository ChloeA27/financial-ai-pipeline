# Financial AI Pipeline

![Badge](https://img.shields.io/badge/Python-3.11%2B-blue) ![Badge](https://img.shields.io/badge/LangGraph-StateGraph-orange) ![Badge](https://img.shields.io/badge/LLM-DeepSeek%20(OpenAI--compatible)-green) ![Badge](https://img.shields.io/badge/Tests-200%2B%20passing-brightgreen)

> **A self-correcting, multi-agent LLM pipeline that turns unstructured financial announcements (M&A, dividends) into strictly-typed, versioned, fully-traceable JSON — orchestrated as a LangGraph state machine with RAG-augmented extraction and a CI golden-dataset quality gate.**

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

- **Reader → Classifier → Extractor → Validator**, each an independent LangGraph node reading/writing a shared `PipelineState`
- **Self-correction loop**: failed validations route back to the Extractor with structured feedback for a bounded number of retries (`max_retries` retries after the first attempt, covering both rule failures and LLM output that doesn't parse into the schema) before the document is quarantined to a Dead Letter Queue
- **Two-model data isolation**: the LLM only ever fills business fields (`MandaExtraction`, `DividendExtraction`); the pipeline itself owns provenance (`doc_id`, `Metadata`, audit trail) so the model can never corrupt controlled fields
- **SHA-256 idempotency + SCD Type 2 versioning**: re-running on unchanged content is a no-op; re-running on an edited file archives the old version and inserts a new one — nothing is ever overwritten
- **RAG-augmented extraction**: validated past extractions are embedded (SentenceTransformer, local) and stored in ChromaDB per document type, then retrieved as few-shot context on first-pass extraction
- **CI golden-dataset gate**: every push runs the pipeline against hand-labelled golden files with a field-type-aware comparison engine (`enum`, `numeric` ±1 %, `bool`, `date`, `text_lower`; per-field overrides `date_range`, `fuzzy_match`, `accept_null`) and fails the build below an accuracy threshold

Currently supports two document types (**M&A**, **Dividend**) via a registry pattern — adding a new type requires a new schema file and a few dict entries, no changes to graph topology or node logic.

---

## 🏗️ Architecture

```
raw_data/*.txt
      │
      ▼
 ┌──────────┐
 │  Reader  │  async file I/O, SHA-256 fingerprint
 │  Agent   │
 └────┬─────┘
      │ raw_content
      ▼
 ┌──────────┐
 │Classifier│  LLM-as-Judge → doc_type ("M&A" | "Dividend" | "Unknown")
 │  Agent   │
 └────┬─────┘
      │ doc_type
      ▼
 ┌──────────┐    ┌─────────────────┐
 │ Extractor│◄───│ RAG Retriever   │  Semantic few-shot from
 │  Agent   │    │ (ChromaDB)      │  historical extractions
 └────┬─────┘    └─────────────────┘
      │ extracted_data         ▲
      ▼                        │ on validated pass
 ┌──────────┐                  │
 │ Validator│──────────────────┘
 │  Agent   │  self-correction loop ← ─ ─ ─ ─ ─ ─
 └────┬─────┘                                  │
      │ (pass/fail)                             │
      ▼                                         │
 ┌──────────┐     ┌──────────────┐              │
 │   PASS   │     │ FAIL, budget │──────────────┘
 │ (persist)│     │  left        │  → Extractor (with feedback)
 └──────────┘     └──────┬───────┘
                         │ FAIL (retry budget spent)
                         ▼
                    ┌──────────┐
                    │   DLQ    │
                    │(dead letter)│
                    └──────────┘
```

Each validated result is written to a SQLite database (`data/pipeline.db`, 4 tables: `extractions`, `extraction_sources`, `correction_logs`, `dead_letter_queue`) with SCD Type 2 versioning on the `extractions` table. If `--output <dir>` is given, a JSON file per document is also written there (including failed ones).

**See [DOCS_SYSTEM_DESIGN.md](./DOCS_SYSTEM_DESIGN.md) for the full system design white paper**, including the state schema, the two-model isolation protocol, the idempotency guard implementation, RAG integration details, and the CI evaluation engine.

---

## ✨ Key Design Features

| Feature | Implementation |
|---------|---------------|
| **State-as-Graph** | Every agent reads/writes a single `PipelineState` `TypedDict` compiled via LangGraph's `StateGraph` |
| **Two-Model Isolation** | LLM-facing `BaseModel` (business fields only, all `Optional`) is promoted to a pipeline-owned `BaseDoc` (with `doc_id` + `Metadata`) via a `from_extraction()` classmethod — the LLM never touches provenance fields |
| **Idempotent Persistence** | SHA-256 hash of raw file content is the true dedup key (not the per-run UUID `doc_id`) — re-running on unchanged content is a guaranteed no-op |
| **Immutable Version History** | SCD Type 2 (`version` + `is_current` columns) — file edits archive the old row and insert a new version, never overwrite |
| **Self-Correction Loop** | A conditional LangGraph edge routes failed validations back to the Extractor with the Validator's structured error feedback injected into the next prompt |
| **Zero-Code Extension** | Adding a document type = 1 new schema file + 3 registry dict entries in `src/nodes/extractor.py` — no changes to graph topology or node logic (optional extras: validator rules, a RAG collection in `COLLECTION_MAP`, and `FIELD_TYPE_MAP` for the golden eval) |
| **RAG-Augmented Extraction** | ChromaDB collection per document type; embeddings via local `SentenceTransformer` (`all-MiniLM-L6-v2`); only validated extractions are embedded; superseded versions are de-indexed and a file is never retrieved as its own example; first-pass extraction only (retries reuse the same context) |
| **Async Concurrency** | `asyncio.Semaphore` + `asyncio.gather` — batch mode runs each file's LangGraph state machine independently and concurrently |
| **Operation Visibility** | Streamlit dashboard (`src/dashboard.py`) reads `data/pipeline.db` live for KPI cards, a Dead Letter panel, and per-`doc_id` self-correction audit drill-downs |
| **CI Golden Dataset Gate** | `scripts/evaluate.py` runs the full pipeline against hand-labelled golden JSON files with a field-type-aware comparison engine, gating the build on a configurable accuracy threshold |

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
- **Streamlit** + **Plotly** — monitoring dashboard
- **loguru** — structured logging
- **pytest** / **pytest-asyncio** / **pytest-cov** — test suite
- **black** — formatting, enforced in CI
- **GitHub Actions** — CI pipeline (unit tests → golden eval → lint)

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
python -m src.main --file raw_data/sample/ma_microsoft_activision.txt
```

### Run on a directory (concurrent batch mode)

```bash
python -m src.main --dir raw_data/sample --concurrency 5
```

### Launch the monitoring dashboard

Requires the optional dashboard dependencies (`pip install -r requirements-dashboard.txt`).

```bash
streamlit run src/dashboard.py
# → http://localhost:8501
```

### Run the test suite

```bash
pytest -m "not integration" --tb=short -q
```

### Run the golden dataset evaluation

```bash
python scripts/evaluate.py --threshold 0.85
```

---

## 📁 Project Structure

```
financial-ai-pipeline/
├── DOCS_SYSTEM_DESIGN.md         ← Full system design white paper
├── README.md                     ← You are here
│
├── raw_data/sample/               ← Sample .txt announcements
├── golden/
│   ├── expected_pass/            ← Golden files pipeline must extract correctly
│   └── expected_fail/            ← Golden files pipeline must reject
│
├── scripts/
│   └── evaluate.py                ← Golden dataset evaluation runner
│
├── .github/workflows/
│   └── ci.yml                     ← CI: unit tests → golden eval → black lint
│
├── src/
│   ├── main.py                    ← CLI entry point (--file / --dir)
│   ├── config.py                  ← Pydantic Settings (env-driven)
│   ├── dashboard.py                ← Streamlit monitoring UI
│   │
│   ├── graph/builder.py           ← LangGraph StateGraph construction
│   ├── state/pipeline_state.py    ← TypedDict state schema
│   │
│   ├── nodes/
│   │   ├── reader.py               ← async file I/O + SHA-256 hashing
│   │   ├── classifier.py           ← LLM-as-Judge doc type classification
│   │   ├── extractor.py            ← LLM extraction + registry pattern
│   │   ├── validator.py            ← rule-based quality gate
│   │   └── retry.py                ← shared retry bookkeeping
│   │
│   ├── schemas/
│   │   ├── base.py                 ← Metadata + BaseDoc
│   │   └── extraction/             ← per-doc-type two-model contracts
│   │
│   ├── rag/                        ← embedder, ChromaDB client, retriever
│   ├── llm/client.py                ← OpenAI-SDK LLM client (DeepSeek-compatible)
│   │
│   └── storage/
│       ├── db.py                    ← aiosqlite schema DDL
│       ├── json_writer.py           ← JSON dual-write
│       └── repository.py            ← SCD Type 2 upsert + idempotency guard
│
└── tests/                          ← unit + integration tests
    ├── test_graph/  test_cli/  test_eval/  test_nodes/
    ├── fake_embedding.py           ← offline embedder used by the test suite
    └── conftest.py                 ← markers + per-test vector-store isolation
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [DOCS_SYSTEM_DESIGN.md](./DOCS_SYSTEM_DESIGN.md) | Full system design white paper — graph topology, two-model isolation, idempotency defense, SCD2 versioning, self-correction loop, RAG architecture, CI golden-dataset gate, concurrency model |

---

## 🧪 Testing & CI

- **200+ unit tests** across nodes, schemas, storage, graph routing and the RAG layer (`pytest -m "not integration"`, no LLM calls). Each test gets a private vector store and an offline fake embedder; add `and not real_embedder` to skip the few tests that load the real model: `pytest -m "not integration and not real_embedder"`
- **Golden dataset evaluation** (`scripts/evaluate.py`) runs the full pipeline end-to-end against hand-labelled expected outputs using a field-type-aware comparison engine (`enum`, `numeric` ±1 %, `bool`, `date`, `text_lower`; per-field overrides `date_range`, `fuzzy_match`, `accept_null`), including per-field override rules (e.g. `date_range`, custom `fuzzy_match` thresholds)
- **GitHub Actions CI** (`.github/workflows/ci.yml`) runs three sequential gates on every push: unit tests → golden evaluation → `black` lint check. The golden evaluation needs the `DEEPSEEK_API_KEY` secret: it is skipped (with a warning) on fork PRs, but a push without the secret fails the build

---

## 👤 Author

**Qi An (Chloe)**
- GitHub: [@ChloeA27](https://github.com/ChloeA27)
- LinkedIn: [Qi An](https://www.linkedin.com/in/qianchloe/)
