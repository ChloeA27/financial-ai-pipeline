# Financial AI Pipeline — System Design White Paper

> **Document Version:** 1.0.0  
> **Status:** Active — ratified  
> **Audience:** Senior AI Data Engineer (interview showcase), Platform Architects, Data Consumers  
> **Repository:** `financial-ai-pipeline`

---

## Table of Contents

1. [Context, Goals & Non-Goals](#1-context-goals--non-goals)
2. [Architecture Overview](#2-architecture-overview)
3. [LangGraph State Machine Topology](#3-langgraph-state-machine-topology)
4. [Two-Model Data Isolation Protocol](#4-two-model-data-isolation-protocol)
5. [Hash-Based Idempotency Defense](#5-hash-based-idempotency-defense)
6. [SCD Type 2 Version Tracking](#6-scd-type-2-version-tracking)
7. [Self-Correction Loop](#7-self-correction-loop)
8. [Storage Architecture](#8-storage-architecture)
9. [Multi-Business-Line Extension (Registry Pattern)](#9-multi-business-line-extension-registry-pattern)
10. [RAG-Augmented Extraction (Phase 2)](#10-rag-augmented-extraction-phase-2)
11. [CI Golden Dataset Gate (Phase 3-C)](#11-ci-golden-dataset-gate-phase-3-c)
12. [Concurrency & Performance](#12-concurrency--performance)
13. [Appendix — Physical File Map](#13-appendix--physical-file-map)

---

## 1. Context, Goals & Non-Goals

### 1.1 Why This Pipeline Exists

Every major financial institution (investment banks, asset managers, hedge funds) ingests **hundreds of unstructured announcements daily** — SEC 8-K filings, dividend declarations, M&A press releases, management change notices. These texts are:

- **Non-standardised** — each issuer uses different phrasing, formatting, and level of detail.
- **Time-sensitive** — actionable information (e.g., ex-dividend date, acquisition price) must be extracted before market open.
- **High-stakes** — a single missed decimal in `total_value_usd` or an inverted acquirer/target can cause P&L damage.

Traditional approaches (regex scraping, template-based parsers) fail because:

1. They cannot generalise across the infinite variety of natural-language financial prose.
2. They do not self-correct — a regex that misses a pattern stays broken forever.
3. They produce unversioned, non-traceable outputs that cannot survive audit.

**The Financial AI Pipeline solves this by combining Large Language Models (LLM) with a deterministic, self-correcting state machine** — extracting structured, versioned, fully-traceable assets from noisy raw text.

### 1.2 Goals

| Goal | Description | Measured By |
|------|-------------|-------------|
| **G1 — Structured Extraction** | Convert raw `.txt` announcements into strictly-typed JSON with validated business fields. | Pydantic model parse success rate; validation pass rate |
| **G2 — Full Traceability** | Every output asset carries immutable provenance: source file path, LLM model, pipeline version, timestamp, retry history. | Presence of `Metadata` on every `BaseDoc` subtype |
| **G3 — Self-Correction** | Failed validation triggers an automatic retry with structured error feedback sent back to the LLM. Documents that exhaust all retries go to a Dead Letter Queue without corrupting core assets. | Conditional edge routing in LangGraph; DLQ isolation |
| **G4 — Idempotent Re-Run** | Running the pipeline twice on the same (unchanged) file produces exactly one database row — no duplicates, no side effects. | SHA-256 hash collision check; zero rows added on repeat run |
| **G5 — Versioned History** | When a source file is amended and re-processed, the old extraction is archived (`is_current=0`) and a new version is created — never overwritten. | SCD Type 2 `version` + `is_current` columns |
| **G6 — Zero-Code Extension** | Adding a new document type requires schema definition only — no changes to graph topology, node logic, or persistence code. | Registry Pattern: `EXTRACTOR_MODELS`, `EXTRACTOR_RESULT_MODELS`, `EXTRACTOR_PROMPTS` |
| **G7 — Async Concurrency** | Batch-processing N documents must not degrade to serial execution; each file runs its own independent LangGraph state machine. | `asyncio.Semaphore` + `asyncio.gather`; wall-clock time < N × single-file time |
| **G8 — Operation Visibility** | Streamlit dashboard exposes KPI cards, error panels, and audit-trail drill-downs querying the live database. | Dashboard visible at `http://localhost:8501` |
| **G9 — RAG-Augmented Extraction** | Historical validated extractions are embedded and stored in ChromaDB (per doc-type collection). On first-pass extraction, semantically similar past results are retrieved as few-shot context to improve LLM accuracy. | ChromaDB collection row count; retrieval relevance |
| **G10 — CI Golden Dataset Gate** | Every push/PR must pass a golden dataset evaluation (field-type-aware comparison against hand-labelled expected outputs) with a configurable accuracy threshold. Pipeline regressions fail the build before deployment. | GitHub Actions CI; `scripts/evaluate.py --threshold 0.85` exit code check |

### 1.3 Non-Goals

| Non-Goal | Rationale |
|----------|-----------|
| **NG1 — We do not provide a web UI for document upload.** | This is a batch-processing **platform** (CLI-driven), not a SaaS application. Future consumer applications can build on the output SQLite database. |
| **NG2 — We do not support PDF, DOCX, or scanned-image input.** | The Reader Node reads plain `.txt` files only. PDF/image OCR is a separate ingestion concern outside this pipeline's scope. |
| **NG3 — We do not host or manage the LLM.** | The pipeline consumes an external LLM API (OpenAI-compatible). Model hosting, fine-tuning, and rate-limit management are the consumer's responsibility. |
| **NG4 — We do not generate natural-language summaries.** | The output is structured JSON, not prose. Downstream teams can build LLM-generated summaries on top of the extracted data if needed. |
| **NG5 — We do not enforce a global uniqueness constraint on `doc_id`.** | `doc_id` is a per-run UUID4 for traceability, not a dedup key. The true idempotency anchor is `source_hash` (SHA-256 of raw content). |
| **NG6 — We do not support real-time / streaming ingestion.** | The pipeline processes static files on disk. Streaming ingestion (e.g., Kafka, WebSocket) would require a different entry-point graph. |
| **NG7 — We do not perform sentiment analysis or market-impact scoring.** | The extraction is purely factual (dates, amounts, counterparties). Sentiment is a higher-order analysis layer above this platform. |
| **NG8 — We do not provide an official REST or gRPC API.** | The pipeline exposes a CLI (`python -m src.main --file ...`). A containerised microservice wrapper can be added later but is out of scope for v1. |

### 1.4 Boundary Diagram

```
                   ┌──────────────────────────────────────────┐
                   │          BOUNDARY OF THIS PIPELINE        │
                   │                                          │
 raw_data/*.txt ──►│  Reader → Classifier → Extractor → Val.  │──► extractions (SCD2)
                   │                    ↻ (self-correct)      │──► JSON files
                   │                                          │──► dead_letter_queue
                   └──────────────────────────────────────────┘
                         │                           │
                    [OUT OF SCOPE]              [OUT OF SCOPE]
                    PDF/OCR ingestion            REST API / UI
                    Sentiment analysis           Real-time streaming
                    LLM model hosting            Fine-tuning
```

---

## 2. Architecture Overview

The Financial AI Pipeline is an industrial-grade **multi-agent extraction platform** that transforms unstructured financial announcements into strictly-typed, fully-traceable JSON assets. It is orchestrated as a **directed state machine** via LangGraph, with an embedded **self-correction loop** for quality assurance.

### 2.1 High-Level Flow

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
 │Classifier│  LLM-as-Judge → doc_type {"M&A", "Dividend", …}
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
                    │ (cemetery)│
                    └──────────┘
```

Output that does not even parse into the extraction schema (bad enum, malformed date, …) loops straight back from the Extractor to itself, without passing through the Validator — see §3.3.

### 2.2 Design Principles

| Principle | Implementation |
|-----------|---------------|
| **State-as-Graph** | Every Agent reads/writes a single `PipelineState` TypedDict |
| **Traceability by Contract** | Every output inherits `BaseDoc` → `Metadata` embedded at extraction time |
| **Idempotent Persistence** | SHA-256 hash of raw content is the true dedup key, not LLM-generated UUID |
| **Immutable Version History** | SCD Type 2 — archive-on-change, never overwrite |
| **Self-Correction** | Validator → Extractor conditional edge with structured error feedback |
| **Zero-Code Extension** | Registry Pattern: add 1 schema file + 3 dict entries = new business line |
| **RAG-Augmented Extraction** | ChromaDB per doc-type collection; embedding via SentenceTransformer (all-MiniLM-L6-v2); validated extractions written as vectors; first-pass extraction retrieves top-k semantic neighbours as few-shot context; retry passes do NOT re-query (avoid stale context) |

---

## 3. LangGraph State Machine Topology

### 3.1 Graph Definition

The pipeline is compiled as a `StateGraph` with `PipelineState` as its schema:

```python
# src/graph/builder.py
workflow = StateGraph(state_schema=PipelineState)
workflow.add_node("reader", reader_node)
workflow.add_node("classifier", classifier_node)
workflow.add_node("extractor", extractor_node)
workflow.add_node("validator", validator_node)

workflow.set_entry_point("reader")
workflow.add_edge("reader", "classifier")     # linear forward
workflow.add_edge("classifier", "extractor")

# Self-correction edge #1: schema-parse failures retry without the Validator
workflow.add_conditional_edges(
    source="extractor",
    path=_route_after_extraction,
    path_map={"validator": "validator", "extractor": "extractor", END: END},
)

# Self-correction edge #2: rule failures retry via the Extractor
workflow.add_conditional_edges(
    source="validator",
    path=_route_after_validation,
    path_map={
        "extractor": "extractor",   # retry loop
        END: END,                   # successful exit
        "error": END,               # exhausted retries
    },
)
```

### 3.2 State Schema (`PipelineState`)

Defined as a `TypedDict` in `src/state/pipeline_state.py`:

| Key | Origin | Type | Description |
|-----|--------|------|-------------|
| `file_path` | Reader | `str` | Absolute path to source `.txt` |
| `raw_content` | Reader | `str` | Full raw text content |
| `doc_type` | Classifier | `Optional[str]` | `"M&A"`, `"Dividend"`, `"Management_Change"` or `"Unknown"`. Only types registered in the Extractor have an extraction path; the rest end in the DLQ. |
| `extracted_data` | Extractor | `Optional[dict]` | Serialised Pydantic extraction result |
| `validation_passed` | Validator | `Optional[bool]` | Quality gate pass/fail |
| `validation_report` | Validator | `Optional[str]` | Detailed validation verdict |
| `correction_logs` | Validator / Extractor | `list[CorrectionLog]` | Chronological history, one entry per failed attempt (`cycle` = 1-indexed attempt number) |
| `retry_count` | Validator / Extractor | `int` | Retries actually performed (0 on the first pass) |
| `max_retries` | Reader | `int` | Retry budget (configurable, default 3): a document gets `max_retries + 1` extraction attempts in total |
| `error` | Any node | `Optional[str]` | Fatal error message |

> **Note:** RAG context is retrieved inside the Extractor Node — it is **not** stored in `PipelineState`. On the first pass the Extractor calls `await retrieve_context(doc_type, query_text, exclude_file_path=file_path)`, prepends the result to the user prompt, and discards it after generation. This keeps the state schema lean and avoids serialising embedding vectors through the graph.

### 3.3 Conditional Edge Router

```python
def _route_after_validation(state: PipelineState) -> Literal["extractor", "__end__", "error"]:
```

Three exit paths:
- **`__end__`** — `validation_passed == True`: document is clean, proceed to persistence
- **`extractor`** — `validation_passed == False` & `retry_count <= max_retries`: send back with feedback for re-extraction
- **`error`** — fatal error, or the retry budget is spent: abort, route to DLQ

`retry_count` is the number of retries already *performed*, so with `max_retries = 3` a document is extracted at most four times (first attempt + 3 retries). The node that records the last allowed failure (`src/nodes/retry.py::register_failure`) leaves `retry_count` at `max_retries` and sets `error`, which is what sends the document to the DLQ. The router's own `retry_count > max_retries` check is only a defensive backstop.

A second conditional edge sits **after the Extractor**: if the LLM's output does not even parse into the schema (bad enum, malformed date, out-of-range number, invalid JSON), the Extractor logs a correction cycle itself and the router sends it straight back for another attempt, skipping the Validator. Transport / API failures remain fatal.

---

## 4. Two-Model Data Isolation Protocol

### 4.1 Motivation

The LLM must **never** see pipeline-controlled fields (`doc_id`, `metadata`, `raw_text_snippet`). If it did, it could hallucinate or corrupt them. The Two-Model Protocol enforces a strict separation of concerns:

```
              ┌──────────────────┐
              │   Model A        │  ← LLM fills this
              │   (BaseModel)    │     Pure business fields only
              │                  │     All Optional → graceful degradation
              └────────┬─────────┘
                       │ `from_extraction()` classmethod
                       ▼
              ┌──────────────────┐
              │   Model B        │  ← Pipeline owns this
              │   (BaseDoc)      │     Inherits Metadata + doc_id + snippet
              │                  │     Immutable audit trail
              └──────────────────┘
```

### 4.2 Model A — LLM-Facing (Business-Only)

File: `src/schemas/extraction/manda.py` (example)

```python
class MandaExtraction(BaseExtractionModel):          # a pydantic BaseModel
    ValidationProfile: ClassVar = MandaValidationProfile

    acquirer: Optional[str] = Field(default=None, min_length=1)
    target: Optional[str] = Field(default=None, min_length=1)
    total_value_usd: Optional[float] = Field(default=None, ge=0.0)
    payment_method: Optional[str] = Field(default=None)   # checked against the profile
    announcement_date: Optional[str] = Field(default=None)  # YYYY-MM-DD, auto-checked
    # ... all Optional — the pipeline never crashes on missing data
```

Key properties:
- Inherits `BaseExtractionModel` (a plain pydantic model, NOT `BaseDoc`) — no `doc_id`, no `metadata`
- Every field is `Optional` — the LLM can output partial data gracefully
- Allowed values live in a `ValidationProfile` (`src/schemas/extraction/profiles.py`, e.g. `ALLOWED_PAYMENT_METHODS`); `ProfileValidatorMixin` injects the matching `@field_validator` automatically, and also a `YYYY-MM-DD` check for every `*_date` field. The result model (Model B) uses the same profile, so the two cannot drift apart.
- Violations raise a pydantic `ValidationError` at parse time. The Extractor catches it and feeds it to the self-correction loop (§3.3) instead of failing the document.

### 4.3 Model B — Pipeline-Facing (Full Traceability)

File: `src/schemas/base.py`

```python
class Metadata(BaseModel):
    source_path: str          # absolute path to raw file
    processed_at: datetime    # UTC extraction timestamp
    model_name: str           # e.g. "deepseek-chat"
    doc_type: str             # e.g. "M&A"
    pipeline_version: str     # e.g. "1.0.0"
    retry_count: int          # self-correction count
    extra: dict[str, Any]     # future-proof catch-all

class BaseDoc(BaseModel):
    doc_id: str               # UUID4 hex — generated by pipeline, NOT the LLM
    metadata: Metadata        # embedded provenance object
    raw_text_snippet: str     # first 500 chars for human-in-the-loop debugging
```

### 4.4 Promotion Bridge

Each concrete type implements a `from_extraction()` classmethod:

```python
class MandaExtractionResult(BaseDoc):
    @classmethod
    def from_extraction(cls, extraction: MandaExtraction, metadata: Metadata,
                        raw_text_snippet: str = "", *, doc_id: str | None = None
    ) -> MandaExtractionResult: ...
```

This is called in `extractor_node()` after the LLM responds:

```python
llm_result = await client.generate_structured(..., response_model=llm_model)
real_doc_id = uuid.uuid4().hex
final_result = result_model.from_extraction(
    extraction=llm_result,
    metadata=metadata,
    raw_text_snippet=raw_content[:500],
    doc_id=real_doc_id,
)
```

---

## 5. Hash-Based Idempotency Defense

### 5.1 Problem

A document must not produce duplicate rows in the database when the pipeline is re-run on unchanged content. The `doc_id` is a **UUID generated fresh every run** — it cannot serve as the dedup key.

### 5.2 Solution — Source Hash as the True Dedup Key

File: `src/storage/repository.py`

```python
def _hash_content(raw_content: str) -> str:
    """SHA-256 hex digest of the raw source text."""
    return hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
```

### 5.3 Guard Logic

Before any insert, the repository checks:

```python
hash_matches = await conn.execute_fetchall(
    "SELECT id, version FROM extractions "
    "WHERE source_hash = ? AND is_current = 1",
    (source_hash,),
)

if hash_matches:
    logger.info("⏭️  [Idempotency Guard] static content detected → skipping write")
    return  # EXIT — no writes, no side effects
```

### 5.4 Guarantees

| Scenario | `source_hash` match? | Behavior |
|----------|---------------------|----------|
| First run of new file | No (no row exists) | Insert v1 |
| Second run, file unchanged | Yes (hash identical) | **Skip** — idempotent no-op |
| Second run, file edited | No (hash differs) | Archive v1 (`is_current=0`), insert v2 |

---

## 6. SCD Type 2 Version Tracking

### 6.1 Model

The platform uses a **Slowly Changing Dimension Type 2** variant for version management:

| Column | Purpose |
|--------|---------|
| `doc_id` | Pipeline-generated UUID4 (not the dedup key) |
| `file_path` | Absolute path to the source file |
| `source_hash` | SHA-256 of raw content — **the true dedup key** |
| `version` | Monotonically incrementing integer per `file_path` |
| `is_current` | `1` = active version, `0` = archived (historical) |
| `extracted_data` | JSON blob of the business extraction result |
| `created_at` / `updated_at` | ISO-8601 timestamps |

### 6.2 Version Evolution Rules

```
File: ma_microsoft_activision.txt
                              is_current
─── Time ────►  v1  (hash A)    1        ← first run
                v1  (hash A)    0        ← file edited, hash changes
                v2  (hash B)    1        ← new version becomes current
```

Implemented in `_upsert_extraction()`:

```python
# 1. Check hash-based idempotency (see §5)
# 2. If no hash match, check if file_path has a current version
existing = await conn.execute_fetchall(
    "SELECT id, version, source_hash, doc_id, doc_type FROM extractions "
    "WHERE file_path = ? AND is_current = 1",
    (file_path,),
)

if existing:
    # Archive old version
    await conn.execute(
        "UPDATE extractions SET is_current = 0, updated_at = ? WHERE id = ?",
        (_now(), existing[0]["id"]),
    )
    new_version = existing[0]["version"] + 1
else:
    new_version = 1

# Insert new version with is_current = 1, commit, then write the new vector
# to the RAG index and delete the archived version's vector (see §10.3)
```

### 6.3 Audit Trail

The `correction_logs` table captures every self-correction cycle:

| Column | Description |
|--------|-------------|
| `doc_id` | Links back to the extraction asset |
| `cycle` | 1-indexed number of the failed attempt |
| `error_summary` | Concise description of validation failures |
| `llm_raw_response` | The full validation report for that attempt (the `raw_feedback` field of the `CorrectionLog`; the LLM's own raw text is not stored) |
| `created_at` | Timestamp of the correction event |

> **Scope:** `correction_logs` rows are written only for documents that eventually pass validation (Branch A). A document that exhausts its retries goes to the DLQ with its final error and `retry_count`, but its per-cycle history is not persisted to the database (it is still included in the optional JSON output).

---

## 7. Self-Correction Loop

### 7.1 Mechanism

The self-correction loop is a **conditional edge** in LangGraph.

```
Validator Node
      │
      ├── validation_passed == True  → END (persist)
      │
      └── validation_passed == False
                │
                ├── retry_count <= max_retries
                │       │
                │       └──→ Extractor Node (with structured feedback)
                │               │
                │               └──→ Validator Node (re-evaluate)
                │
                └── retry budget spent (error flag set) → END (DLQ)
```

### 7.2 Correction Feedback

When the Extractor is called for a retry, the `_build_correction_feedback()` function injects the Validator's error summary into the LLM prompt:

```python
user_prompt += (
    "\n\n─── PREVIOUS VALIDATION FEEDBACK (correct these issues) ───\n"
    f"{correction_feedback}"
)
```

This ensures the LLM is aware of its previous mistakes and can correct them without fabricating data.

### 7.3 Exhaustion → Dead Letter Queue

When the retry budget is spent (`max_retries` retries performed, default 3, and the last attempt still failed), the document is routed to `dead_letter_queue` via Branch B of the repository:

```python
async def _insert_dead_letter(self, final_state: dict) -> None:
    await conn.execute(
        "INSERT INTO dead_letter_queue "
        "(file_path, source_hash, doc_type, last_error, retry_count, failed_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ...
    )
```

DLQ documents are **never mixed** with core assets — the `dead_letter_queue` table is a separate cemetery for manual investigation.

There is one DLQ row per `(file_path, source_hash)`: re-running an unchanged file that fails again refreshes that row (latest error, `retry_count`, `failed_at`) instead of adding a duplicate; editing the file produces a new row. In batch mode a pipeline that *raises* (rather than returning a failed state) is quarantined the same way, with `last_error = "pipeline raised <ExceptionType>: <message>"`.

---

## 8. Storage Architecture

### 8.1 Dual-Write Strategy

Each pipeline result is persisted in **two formats** simultaneously:

1. **JSON File** — human-readable, git-friendly, directory listing
   - Writer: `src/storage/json_writer.py`
   - Output path: `output/{filename}_result.json`

2. **SQLite Database** — queryable, versioned, machine-parseable
   - Connection: `aiosqlite` with WAL mode
   - Schema: `src/storage/db.py` (4 tables)

### 8.2 Database Schema

```sql
-- Table 1: extractions — Core asset table (SCD Type 2)
CREATE TABLE extractions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    source_hash     TEXT NOT NULL,
    doc_type        TEXT NOT NULL,
    extracted_data  TEXT NOT NULL,        -- JSON blob
    version         INTEGER NOT NULL DEFAULT 1,
    is_current      INTEGER NOT NULL DEFAULT 1,
    pipeline_version TEXT NOT NULL DEFAULT '1.0.0',
    model_name      TEXT NOT NULL,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

-- Table 2: extraction_sources — Cold storage for raw text
CREATE TABLE extraction_sources (
    doc_id           TEXT NOT NULL PRIMARY KEY,
    raw_text_snippet TEXT NOT NULL
);

-- Table 3: correction_logs — Self-correction audit trail
CREATE TABLE correction_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id          TEXT NOT NULL,
    cycle           INTEGER NOT NULL,
    error_summary   TEXT NOT NULL,
    llm_raw_response TEXT,
    created_at      TEXT NOT NULL
);

-- Indexes: extractions(doc_id, is_current), extractions(source_hash),
--          extractions(doc_type), correction_logs(doc_id)

-- Table 4: dead_letter_queue — Exhausted retry cemetery
CREATE TABLE dead_letter_queue (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path       TEXT NOT NULL,
    source_hash     TEXT NOT NULL,
    doc_type        TEXT,
    last_error      TEXT NOT NULL,
    retry_count     INTEGER NOT NULL,
    failed_at       TEXT NOT NULL
);
```

### 8.3 Repository Layer

The `ExtractionRepository` class (`src/storage/repository.py`) acts as the single **estate manager**:

```python
class ExtractionRepository:
    async def save_extraction(self, final_state: dict) -> None:
        # Step 1: JSON debug output, only if an output dir was given (--output)
        if self._output_dir:
            await write_json_output(final_state, self._output_dir)

        # Step 2: Route to correct branch
        if validated:
            await self._upsert_extraction(final_state)    # Branch A
        else:
            await self._insert_dead_letter(final_state)   # Branch B
```

---

## 9. Multi-Business-Line Extension (Registry Pattern)

### 9.1 The Pattern

Adding a new document type (e.g., "Dividend") requires **zero changes** to graph topology, node logic, or repository code. The pattern is purely additive:

```
src/schemas/extraction/dividend.py      ← new schema (two models)
src/nodes/extractor.py                  ← 3 dict entries (no logic changes)
src/schemas/extraction/__init__.py      ← 1 export line
```

Those are the only **required** changes. Three further registries are optional, and each degrades gracefully when skipped:

| Registry | Without an entry |
|----------|------------------|
| `_VALIDATION_RULESETS` / `_ALLOWED_VALUES` in `src/nodes/validator.py` | The Validator passes the document silently (only schema-level checks apply) |
| `COLLECTION_MAP` in `src/rag/chroma_client.py` | No RAG for this type: retrieval returns nothing and the vector write is logged as a non-fatal error |
| `FIELD_TYPE_MAP` in `scripts/evaluate.py` | New fields fall back to case-insensitive text comparison in the golden eval |

### 9.2 Registration Points

In `src/nodes/extractor.py`:

```python
# ── LLM-facing model (Model A) ──
EXTRACTOR_MODELS: dict[str, type] = {
    "M&A": MandaExtraction,
    "Dividend": DividendExtraction,         # ← add one line
}

# ── Pipeline-facing result model (Model B) ──
EXTRACTOR_RESULT_MODELS: dict[str, type] = {
    "M&A": MandaExtractionResult,
    "Dividend": DividendExtractionResult,   # ← add one line
}

# ── System prompt for the LLM ──
EXTRACTOR_PROMPTS: dict[str, str] = {
    "M&A": """...""",
    "Dividend": """...""",                  # ← add one line
}
```

### 9.3 Classifier Compatibility

The Classifier node (`src/nodes/classifier.py`) uses an LLM-as-Judge approach — it is **prompt-driven**, not code-driven. New doc types only need to be listed in the system prompt:

```python
SYSTEM_PROMPT = """... ONE of:
- M&A               — Merger, acquisition, asset purchase, tender offer, etc.
- Dividend          — Dividend declaration, distribution, payout ratio change.
- Management_Change — Executive appointment/resignation, board changes, C-suite moves.
- Unknown           — None of the above, or ambiguous."""
```

---

## 10. RAG-Augmented Extraction (Phase 2)

### 10.1 Motivation

Cold-start LLM extraction — where the model sees only the system prompt and raw text — can miss domain-specific conventions (e.g., "record date" vs "ex-dividend date" in dividend announcements). A RAG (Retrieval-Augmented Generation) layer solves this by providing semantically similar past extractions as few-shot examples during the first-pass LLM call.

### 10.2 Architecture

```
┌──────────────┐     on validated pass
│  Validator   │──────────┐
│  (pass)      │          │
└──────────────┘          ▼
                    ┌──────────────┐
                    │   Embedder   │  SentenceTransformer
                    │  (local)     │  → all-MiniLM-L6-v2
                    └──────┬───────┘
                           │ embedding
                           ▼
                    ┌──────────────┐
                    │  ChromaDB    │  Per doc-type collection:
                    │  (vector DB) │  extractions_manda / extractions_dividend
                    └──────────────┘
                           ▲
                    ┌──────┴───────┐
                    │   Retriever  │  top-k semantic search
                    │  (async)     │  → prepended to Extractor user prompt
                    └──────────────┘
                           │
                    ┌──────┴───────┐
                    │  Extractor   │  First-pass only; retries
                    │  (first-pass)│  do NOT re-query RAG
                    └──────────────┘
```

### 10.3 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Per doc-type collections** | M&A and Dividend extractions have fundamentally different schemas; searching across types would retrieve irrelevant results. |
| **Embedding via local SentenceTransformer** | No external API call — embeddings are fast, free, and deterministic. Current model: `all-MiniLM-L6-v2` (384-dim). Future migration path: Amazon Bedrock Titan Embeddings for production scale. |
| **Write-on-validate** | Vectors are persisted **only after** the Validator passes. Retry / DLQ results are never embedded — avoiding noise in the vector store. |
| **First-pass only** | RAG context is retrieved once before the initial LLM call. Retry passes (self-correction loop) reuse the same context; re-querying would risk retrieving the (incorrect) first attempt. |
| **Never its own example** | The retriever skips extractions whose source file is the one currently being processed (`exclude_file_path`). Otherwise a re-run — especially after the file was edited — would feed the file's previous, possibly outdated extraction back to the LLM as a "validated reference". |
| **Superseded versions are de-indexed** | When SCD Type 2 archives a version, its vector is deleted from the collection. The archived row stays in SQLite as history but is never served as RAG context. |
| **Cold-start graceful degradation** | On first run (empty ChromaDB), the Retriever returns an empty string — the Extractor falls back to zero-shot extraction with no performance penalty. |

### 10.4 Module Map

| File | Responsibility |
|------|---------------|
| `src/rag/embedder.py` | `get_embedding_function()` — lazy singleton wrapping Chroma's `SentenceTransformerEmbeddingFunction`; `embed_text()` for ad-hoc use |
| `src/rag/chroma_client.py` | Singleton ChromaDB client, per-doc-type `COLLECTION_MAP`, and `override_chroma_path()` for test/eval isolation |
| `src/rag/repository.py` | `save_to_vector_store()` (upsert a validated extraction) and `delete_from_vector_store()` (drop a superseded version); both non-fatal |
| `src/rag/retriever.py` | `retrieve_context()` — top-k semantic search (run in a thread pool), optionally excluding the current source file, formatted as a few-shot context string |

### 10.5 Integration in Extractor Node

```python
# src/nodes/extractor.py — inside extractor_node()
user_prompt = f"Extract from this announcement:\n\n{raw_content[:6000]}"

if not correction_feedback:                    # first pass only
    rag_context = await retrieve_context(
        doc_type=doc_type,
        query_text=raw_content[:1000],
        exclude_file_path=state.get("file_path") or None,   # never its own example
    )
    if rag_context:
        user_prompt = rag_context + "\n" + user_prompt
else:                                          # retries: validator / schema feedback
    user_prompt += "\n\n─── PREVIOUS VALIDATION FEEDBACK ... ───\n" + correction_feedback

llm_result = await client.generate_structured(
    system_prompt=EXTRACTOR_PROMPTS[doc_type],
    user_prompt=user_prompt,
    response_model=llm_model,
)
```

### 10.6 Possible Migration to Bedrock Titan *(not implemented)*

Embedding is currently purely local (`all-MiniLM-L6-v2` via SentenceTransformer). A production deployment at scale could swap in a hosted model, e.g.:

1. Put `src/rag/embedder.py` behind a small provider interface
2. Implement a `BedrockTitanEmbedder` using `boto3` to call `amazon.titan-embed-text-v2`
3. Select the provider through a new setting (none exists today)

**Re-indexing is required.** Vectors from different embedding models are not comparable — even when the dimensions happen to match — so every existing collection must be rebuilt from the stored extractions after any model change (the model name is not currently recorded in the Chroma metadata, so a change has to be tracked by hand).

---

## 11. CI Golden Dataset Gate (Phase 3-C)

### 11.1 Motivation

Without a regression safety net, a change to the Extractor prompt, Validator rules, or LLM model can silently degrade extraction quality. The Golden Dataset Gate solves this by running the full pipeline against a **hand-labelled golden dataset** on every CI push/PR.

### 11.2 Directory Structure

```
golden/
├── expected_pass/               ← Pipeline must produce ≥85% accuracy
│   ├── dividend_apple_2026.golden.json
│   ├── ma_microsoft_activision.golden.json
│   └── test_ma_pass.golden.json
│
└── expected_fail/               ← Pipeline must reject the document
    └── test_ma_fail_loop.golden.json
```

### 11.3 Eval Engine — `scripts/evaluate.py`

The evaluator runs each golden file through the pipeline with its SQLite and ChromaDB paths pointed at a temp directory, then compares the output against the golden JSON using a **field-type-aware comparison engine**. Each field's mode comes from `FIELD_TYPE_MAP` in `scripts/evaluate.py` (default: `text_lower`):

| Mode | Match Rule | Example Fields |
|------|-----------|----------------|
| `enum` | Exact string match | `payment_method`, `dividend_type`, `currency`, `frequency` |
| `numeric` | Relative error ≤ 1 % (`NUMERIC_RELATIVE_TOLERANCE`); expected `0` requires `|actual| < 1e-6` | `total_value_usd`, `stake_percentage`, `dividend_cash_amount` |
| `bool` | Strict boolean equality | `requires_shareholder_approval` |
| `date` | YYYY-MM-DD string equality | `announcement_date`, `ex_dividend_date` |
| `text_lower` | Case-insensitive trimmed string match | `acquirer`, `target`, `ticker` |

**Per-field overrides** — stored as `_field_eval_overrides` in each golden JSON file — adjust the comparison for one field without touching the engine:

| Override | Effect |
|----------|--------|
| `accept_null: true` | A `null` from the pipeline counts as a pass |
| `mode: "date_range"` + `range_start` / `range_end` | Any date inside the (inclusive) range passes |
| `mode: "fuzzy_match"` + `threshold` (default 0.85) | `difflib.SequenceMatcher` ratio ≥ threshold, for names with minor variations |

Example from `golden/expected_pass/test_ma_pass.golden.json`:

```json
{
  "_field_eval_overrides": {
    "expected_close_date": {
      "mode": "date_range",
      "accept_null": true,
      "range_start": "2026-04-01",
      "range_end": "2026-06-30"
    }
  }
}
```

The reported accuracy is `passed checks / total checks` pooled over all golden files (each field of an expected-pass file is one check; each expected-fail file is one check).

### 11.4 Expected-Fail Mode

Documents in `golden/expected_fail/` are known to contain corrupt/insufficient data. The evaluator only checks that the pipeline **legitimately rejected** the document:
1. `validation_passed == False` (retries exhausted), or
2. the Classifier succeeded but returned a type with no registered extractor (e.g. `Unknown`).

An infrastructure error (LLM unreachable, bad API key, unreadable file) scores **0** — it proves nothing about extraction quality.

Field-level comparison is skipped entirely.

### 11.5 CI Workflow — `.github/workflows/ci.yml`

```yaml
jobs:
  test-and-eval:
    steps:
      - name: Install CPU-only PyTorch        # avoids GBs of unused CUDA libs
      - name: Cache embedding model           # ~/.cache/huggingface
      - name: Unit tests (no LLM calls)
        run: pytest -m "not integration" --tb=short -q
      - name: Check LLM secret availability   # see below
      - name: Golden dataset evaluation
        if: steps.llm.outputs.available == 'true'
        run: python scripts/evaluate.py --threshold 0.85 --json
      - name: Lint check (black)
        run: black --check --diff src/ tests/ scripts/
```

Three gates, sequential:
1. **Unit tests** — 200+ tests, zero LLM calls. Every test runs against a private ChromaDB directory and an offline fake embedding function (`tests/conftest.py`), so nothing touches `data/chromadb` and nothing needs the network — except tests marked `real_embedder`, which load the actual sentence-transformers model. For a fully offline run: `pytest -m "not integration and not real_embedder"`.
2. **Golden evaluation** — full pipeline end-to-end with real LLM, exit code 0/1 gates the build
3. **Black lint** — format consistency, with `black==25.11.0` pinned in `requirements-dev.txt` and installed alongside the test tooling

**Secrets and forks.** Repository secrets are not exposed to pull requests from forks. On a pull request without `DEEPSEEK_API_KEY` the golden evaluation is skipped with a visible warning; on a **push** (to `main`) a missing key fails the build, so the quality gate cannot be bypassed silently.

### 11.6 Isolation During Eval

For each golden file the evaluator swaps `settings.sqlite_db_path` and `settings.chromadb_path` to a temporary directory and resets the ChromaDB singleton via `override_chroma_path`, restoring both afterwards. `evaluate.py` only runs the graph (`run_pipeline`) and never calls the repository, so nothing is persisted and the RAG index stays empty — the golden gate therefore measures **zero-shot** extraction quality, not RAG-augmented quality. Files are evaluated sequentially (the settings swap is process-global, so the runner is not safe to parallelise).

---

## 12. Concurrency & Performance

### 12.1 Async I/O Everywhere

| Layer | Mechanism |
|-------|-----------|
| File reading | `aiofiles` + `asyncio.gather` |
| LLM calls | `await client.generate_structured()` |
| Database | `aiosqlite` (async SQLite) |
| Batch processing | `asyncio.Semaphore(concurrency)` + `asyncio.gather` |

### 12.2 Batch Pipeline Execution

```python
async def _gather_with_semaphore(file_paths: list[str], concurrency: int = 5):
    sem = asyncio.Semaphore(concurrency)

    async def _bounded_pipeline(fp: str) -> dict:
        async with sem:
            return await run_pipeline(fp)

    tasks = [_bounded_pipeline(fp) for fp in file_paths]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

Each `run_pipeline()` call independently invokes the full LangGraph state machine on its own file. The semaphore prevents overwhelming the LLM API; its size comes from `--concurrency` (default 5, must be ≥ 1). Results are persisted one by one after all pipelines in the batch have finished, so documents in the same batch never see each other in the RAG index.

---

## 13. Appendix — Physical File Map

```
financial-ai-pipeline/
│
├── DOCS_SYSTEM_DESIGN.md         ← This document
├── README.md                     ← Project overview and usage
├── requirements*.txt             ← runtime / dev / dashboard dependencies
├── .env                          ← Secrets (LLM API key, etc.) — copy from .env.example
│
├── raw_data/
│   └── sample/                   ← Test fixtures
│       ├── ma_microsoft_activision.txt
│       ├── test_ma_pass.txt
│       ├── test_ma_fail_loop.txt
│       └── dividend_apple_2026.txt
│
├── scripts/
│   └── evaluate.py               ← Golden dataset evaluation runner (Phase 3-C)
│
├── golden/
│   ├── expected_pass/            ← Pipeline must produce ≥85% accuracy
│   │   ├── dividend_apple_2026.golden.json
│   │   ├── ma_microsoft_activision.golden.json
│   │   └── test_ma_pass.golden.json
│   └── expected_fail/            ← Pipeline must reject the document
│       └── test_ma_fail_loop.golden.json
│
├── .github/workflows/
│   └── ci.yml                    ← CI gates: unit tests → golden eval → lint
│
├── src/
│   ├── main.py                   ← CLI entry point (argparse)
│   ├── config.py                 ← Pydantic Settings
│   ├── dashboard.py              ← Streamlit monitoring UI
│   │
│   ├── graph/
│   │   └── builder.py            ← LangGraph StateGraph construction
│   │
│   ├── state/
│   │   └── pipeline_state.py     ← TypedDict state schema
│   │
│   ├── nodes/
│   │   ├── reader.py             ← async file I/O
│   │   ├── classifier.py         ← LLM-as-Judge doc type classification
│   │   ├── extractor.py          ← LLM extraction + Registry
│   │   ├── validator.py          ← Rule-based quality gate
│   │   └── retry.py              ← Shared retry bookkeeping (Validator + Extractor)
│   │
│   ├── schemas/
│   │   ├── base.py               ← Metadata + BaseDoc (foundation)
│   │   └── extraction/
│   │       ├── __init__.py       ← Public exports
│   │       ├── base_model.py     ← BaseExtractionModel (shared validators)
│   │       ├── manda.py          ← M&A two-model contract
│   │       ├── dividend.py       ← Dividend two-model contract
│   │       └── profiles.py       ← ValidationProfiles (allowed-value sets per business line)
│   │
│   ├── rag/                      ← Phase 2 — RAG-Augmented Extraction
│   │   ├── __init__.py
│   │   ├── embedder.py           ← SentenceTransformer embedding function (singleton)
│   │   ├── chroma_client.py      ← Singleton ChromaDB client + per-type collections
│   │   ├── repository.py         ← save / delete vectors for validated extractions
│   │   └── retriever.py          ← top-k semantic search → few-shot context
│   │
│   ├── llm/
│   │   └── client.py             ← OpenAI-SDK client (DeepSeek-compatible) + JSON parsing
│   │
│   └── storage/
│       ├── __init__.py           ← Repository + init_db exports
│       ├── db.py                 ← aiosqlite connection + schema DDL
│       ├── json_writer.py        ← JSON file dual-write
│       └── repository.py         ← SCD Type 2 upsert + Idempotency Guard
│
├── output/                       ← JSON dual-write artefacts
├── data/                         ← SQLite database
├── logs/                         ← Log output
│
├── tests/
│   ├── conftest.py               ← Markers + autouse vector-store isolation
│   ├── fake_embedding.py         ← Offline deterministic embedding function
│   ├── test_graph/
│   │   └── test_builder.py       ← Routers + scripted-LLM whole-graph runs
│   ├── test_cli/
│   │   └── test_main_concurrency.py
│   ├── test_eval/
│   │   └── test_expected_fail.py ← Golden-eval expected_fail scoring
│   └── test_nodes/
│       ├── test_reader.py
│       ├── test_schemas.py
│       ├── test_validator.py
│       ├── test_extractor.py     ← Schema-parse failures → self-correction
│       ├── test_retry.py
│       ├── test_storage.py
│       └── test_rag.py           ← Phase 2 — RAG tests
│
├── requirements.txt              ← Runtime dependencies
├── requirements-dev.txt          ← Test + lint tooling (pytest, black==25.11.0)
├── requirements-dashboard.txt    ← Optional: Streamlit dashboard deps
├── pyproject.toml                ← Pytest configuration
├── .env.example                  ← Environment template
└── .gitignore
```

---

> **End of White Paper — Financial AI Pipeline v1.0.0**  
> *"Every asset has a version; every version has a hash; every hash has a proof."*
