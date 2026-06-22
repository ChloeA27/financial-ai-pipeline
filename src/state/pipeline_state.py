"""
Pipeline graph state — the single source of truth flowing through LangGraph nodes.

Each key is a node's input and/or output.  The typed dict structure ensures
that every agent knows exactly what it can read and what it must produce.
"""

from __future__ import annotations

from typing import Any, Optional

from typing_extensions import TypedDict


class CorrectionLog(TypedDict):
    """Record of a single self-correction cycle."""

    cycle: int
    """Which retry attempt this is (1-indexed)."""
    error_summary: str
    """Human-readable description of what the Validator found wrong."""
    raw_feedback: str
    """The raw validation report text for audit trail."""


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

    # ── Classification (set by Classifier Node) ──
    doc_type: Optional[str]
    """
    Classified document type.
    One of: "M&A", "Dividend", "Management_Change", or "Unknown".
    """

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
    """Number of retries attempted so far (used in conditional edges)."""

    max_retries: int
    """Maximum allowed retries before the document is flagged as failed."""

    # ── Final output ──
    error: Optional[str]
    """Fatal error message if the pipeline failed for this document."""
