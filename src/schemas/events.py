"""
Audit and failure event records (contracts shared by nodes and storage).

  * ``StepRun``       one node execution → table ``step_runs`` (silver layer)
  * ``FailureInfo``   why a document stopped → DLQ columns (vocabulary: ``src.schemas.failure``)
  * ``CorrectionLog`` one self-correction cycle → table ``correction_logs``

They are kept apart from ``PipelineState`` so the graph state and the records
written about it can evolve separately.
"""

from __future__ import annotations

from typing import Any, Optional

from typing_extensions import TypedDict


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

    See ``src.schemas.failure`` for the stage / category / code vocabulary.
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
    """SHA-256 of the source file's bytes — the key into the raw (bronze) layer."""
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
