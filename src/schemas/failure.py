"""
Structured failure vocabulary for the Dead Letter Queue (DLQ).

Every document that leaves the pipeline without a validated extraction carries
a ``failure`` record that says *where* it failed (``stage``), *what* went wrong
(``code``) and *what kind* of problem it is (``category``).  The DLQ stores
these as separate columns, so failures can be counted per stage and per code
instead of grepping free-text error messages.

Stages (where in the graph the document stopped)::

    read → classify → classify_validate → extract → validate
                                  (+ "pipeline" for crashes outside the graph)

Categories (what kind of problem, loosely following the error taxonomy in the
design notes)::

    transient     infrastructure / LLM transport problem — says nothing about quality
    structure     output has the wrong shape (bad JSON, bad enum, bad date format)
    reference     output cites evidence that is not in the source text
    missing_info  a required value is absent
    business_rule a cross-field business rule is violated
    out_of_scope  the document is not a type this pipeline handles (expected, not an error)
    ambiguous     the classifier could not decide (needs a human)
    input_error   the input itself is unusable (missing file, empty text)
    unknown       anything not yet classified
"""

from __future__ import annotations

from typing import Any, Optional

from src.schemas.events import FailureInfo

# ── Stages ──
STAGE_READ = "read"
STAGE_CLASSIFY = "classify"
STAGE_CLASSIFY_VALIDATE = "classify_validate"
STAGE_EXTRACT = "extract"
STAGE_VALIDATE = "validate"
STAGE_PIPELINE = "pipeline"

# ── Categories ──
CAT_TRANSIENT = "transient"
CAT_STRUCTURE = "structure"
CAT_REFERENCE = "reference"
CAT_MISSING_INFO = "missing_info"
CAT_BUSINESS_RULE = "business_rule"
CAT_OUT_OF_SCOPE = "out_of_scope"
CAT_AMBIGUOUS = "ambiguous"
CAT_INPUT_ERROR = "input_error"
CAT_UNKNOWN = "unknown"

# ── Codes that are not derived from a validation rule ──
LLM_CALL_FAILED = "LLM_CALL_FAILED"
FILE_NOT_FOUND = "FILE_NOT_FOUND"
DECODE_FAILED = "DECODE_FAILED"
NO_FILE_PATH = "NO_FILE_PATH"
EMPTY_DOCUMENT = "EMPTY_DOCUMENT"
MISSING_DOC_TYPE = "MISSING_DOC_TYPE"
UNSUPPORTED_DOC_TYPE = "UNSUPPORTED_DOC_TYPE"
CLASSIFICATION_UNCLEAR = "CLASSIFICATION_UNCLEAR"
CLASSIFICATION_PARSE_FAILED = "CLASSIFICATION_PARSE_FAILED"
CLASSIFICATION_MISSING = "CLASSIFICATION_MISSING"
CLASSIFICATION_INVALID_LABEL = "CLASSIFICATION_INVALID_LABEL"
EVIDENCE_QUOTE_MISSING = "EVIDENCE_QUOTE_MISSING"
EVIDENCE_QUOTE_TOO_SHORT = "EVIDENCE_QUOTE_TOO_SHORT"
EVIDENCE_QUOTE_NOT_FOUND = "EVIDENCE_QUOTE_NOT_FOUND"
SCHEMA_PARSE_FAILED = "SCHEMA_PARSE_FAILED"
NO_EXTRACTED_DATA = "NO_EXTRACTED_DATA"
PIPELINE_EXCEPTION = "PIPELINE_EXCEPTION"
UNCLASSIFIED_ERROR = "UNCLASSIFIED_ERROR"


def make_failure(
    stage: str,
    code: str,
    category: str,
    message: str,
    details: Optional[dict[str, Any]] = None,
) -> FailureInfo:
    """Build a ``FailureInfo`` record (``details`` carries optional extras)."""
    return FailureInfo(
        stage=stage,
        code=code,
        category=category,
        message=message,
        details=details,
    )


def fallback_failure(error: Optional[str]) -> FailureInfo:
    """
    Failure record for a state that has an ``error`` but no ``failure``.

    Used by the persistence layer so a DLQ row never has empty stage/code
    columns, even for failures raised by code that predates this vocabulary.
    """
    return make_failure(
        stage=STAGE_PIPELINE,
        code=UNCLASSIFIED_ERROR,
        category=CAT_UNKNOWN,
        message=error or "Unknown error",
    )
