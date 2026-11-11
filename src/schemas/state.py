"""
Pipeline graph state — the single source of truth flowing through LangGraph nodes.

Each key is a node's input and/or output.  The typed dict structure ensures
that every agent knows exactly what it can read and what it must produce.
"""

from __future__ import annotations

import operator
from typing import Any, Optional

from typing_extensions import Annotated, TypedDict

from src.schemas.events import CorrectionLog, FailureInfo, StepRun


class PipelineState(TypedDict):
    """
    Global state flowing through the LangGraph computation graph.

        Reader → Classifier ⇄ Classification Validator → Extractor ⇄ Validator
                 (classify loop)                          (extract loop)

    Reducers: ``step_runs`` is appended to (``operator.add``); every other key,
    ``correction_logs`` included, is replaced by the value a node returns, so
    a node that adds a correction log returns the whole list.
    """

    # ── Input (set by Reader Node) ──
    file_path: str
    """Absolute path to the raw text file being processed."""

    raw_content: str
    """Full raw text content of the announcement file."""

    source_hash: str
    """SHA-256 of the source file's bytes; key of the raw (bronze) layer."""

    raw_uri: Optional[str]
    """Where the unchanged source bytes are stored (bronze)."""

    source_encoding: Optional[str]
    """How the bytes were decoded: utf-8, utf-8-bom or cp1252 (None if decoding failed)."""

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

    classifier_input: Optional[str]
    """The exact text the classifier was given (the evidence quote must come from it)."""

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
