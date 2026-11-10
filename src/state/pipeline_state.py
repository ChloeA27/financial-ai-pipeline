"""
Pipeline graph state — the single source of truth flowing through LangGraph nodes.

Each key is a node's input and/or output.  The typed dict structure ensures
that every agent knows exactly what it can read and what it must produce.
"""

from __future__ import annotations

import operator
from typing import Any, Optional

from typing_extensions import Annotated, TypedDict


class CorrectionLog(TypedDict):
    """Record of a single self-correction cycle."""

    cycle: int
    """Which attempt of this loop failed (1-indexed, counted per ``loop``)."""
    error_summary: str
    """Human-readable description of what the Validator found wrong."""
    raw_feedback: str
    """The raw validation report text for audit trail."""
    loop: str
    """Which self-correction loop this belongs to: ``"classify"`` or ``"extract"``."""


class FailureInfo(TypedDict):
    """
    Structured description of why a document ended up in the DLQ.

    See ``src.state.failure`` for the stage / category / code vocabulary.
    """

    stage: str
    """Where the document stopped (read, classify, classify_validate, extract, validate, pipeline)."""
    code: str
    """Stable machine-readable reason, e.g. ``EVIDENCE_QUOTE_NOT_FOUND``."""
    category: str
    """Kind of problem (transient, structure, reference, missing_info, ...)."""
    message: str
    """Human-readable message (same text as ``error`` for fatal failures)."""
    details: Optional[dict[str, Any]]
    """Optional extras, e.g. every rule that failed in the last attempt."""


class StepRun(TypedDict):
    """
    Audit record of ONE node execution (silver layer, table ``step_runs``).

    Written for every attempt of every LLM call and every validation, passed
    or not, so a document's whole path can be replayed from the database:
    what the model was asked (prompt version), what it answered (raw text),
    what the pipeline made of it (output), and what the verdict was.
    """

    run_id: str
    """One pipeline invocation for one document (joins all steps of that run)."""
    source_hash: str
    """SHA-256 of the processed text — the key into the raw layer."""
    file_path: str
    stage: str
    """classify, classify_validate, extract, validate."""
    loop: str
    """classify or extract — which self-correction loop the step belongs to."""
    attempt: int
    """1-indexed attempt within that loop (the classify validator shares the classifier's number)."""
    outcome: str
    """passed (continue) · rejected (retry follows) · fatal (the document stops here)."""
    error_code: Optional[str]
    error_category: Optional[str]
    output: Optional[Any]
    """What the step produced: parsed LLM output, or the validator's verdict."""
    model_name: Optional[str]
    prompt_version: Optional[str]
    """Hash of the exact system message sent (changes when the prompt or schema changes)."""
    raw_response: Optional[str]
    """The LLM's raw text before parsing (also kept when parsing failed)."""
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    latency_ms: Optional[int]
    started_at: str


class PipelineState(TypedDict):
    """
    Global state flowing through the LangGraph computation graph.

    ┌──────────────┐     ┌──────────────┐     ┌────────────────┐
    │  Reader Node │────>│ Classifier   │────>│  Extractor     │
    │              │     │   Node       │     │   Node         │
    └──────────────┘     └──────────────┘     └───────┬────────┘
                                                      │
                                              ┌───────▼────────┐
                                              │  Validator     │◄──── self-correction
                                              │   Node         │────── loop ──────►
                                              └────────────────┘
    """

    # ── Input (set by Reader Node) ──
    file_path: str
    """Absolute path to the raw text file being processed."""

    raw_content: str
    """Full raw text content of the announcement file."""

    source_hash: str
    """SHA-256 of ``raw_content``; key of the raw (bronze) layer."""

    run_id: str
    """Unique id of this pipeline invocation (links every ``step_runs`` row)."""

    # ── Classification (set by Classifier Node, checked by the Classification Validator) ──
    doc_type: Optional[str]
    """
    Classifier label: "M&A", "Dividend", "none" (clearly neither) or
    "unclear" (cannot decide).  Only "M&A" and "Dividend" have an extraction
    path; the other two end in the DLQ.  ``None`` until a classification
    attempt has been accepted.
    """

    classification: Optional[dict[str, Any]]
    """
    Latest classifier output: ``label``, ``evidence_quote``, ``confidence``,
    ``reasoning``.  ``confidence`` is the model's self-report and is kept for
    logging only — it is never used to route documents.
    """

    classification_passed: Optional[bool]
    """False while the latest classification was rejected and is being retried."""

    classify_retry_count: int
    """Classification retries performed so far (separate budget from ``retry_count``)."""

    classify_max_retries: int
    """Classification retry budget."""

    # ── Extraction (set by Extractor Node) ──
    extracted_data: Optional[dict[str, Any]]
    """
    Parsed extraction result as a dict (serialised from Pydantic model).
    `None` until the Extractor Node has run successfully.
    """

    # ── Validation (set by Validator Node) ──
    validation_passed: Optional[bool]
    """Whether the extraction passed all quality checks."""

    validation_report: Optional[str]
    """Detailed report from the Validator (JSON or free-text)."""

    correction_logs: list[CorrectionLog]
    """History of all self-correction cycles for this document."""

    retry_count: int
    """Extraction retries performed so far (used in conditional edges)."""

    max_retries: int
    """Maximum allowed extraction retries before the document is flagged as failed."""

    # ── Final output ──
    error: Optional[str]
    """Fatal error message if the pipeline failed for this document."""

    failure: Optional[FailureInfo]
    """Structured stage / code / category for a fatal failure (written to the DLQ)."""

    # ── Audit trail ──
    step_runs: Annotated[list[StepRun], operator.add]
    """
    One record per node execution, appended by each node.  The ``operator.add``
    reducer makes LangGraph *concatenate* what nodes return instead of
    replacing the list, so a node only returns its own new record.
    """
