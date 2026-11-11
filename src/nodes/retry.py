"""
Shared retry bookkeeping for the self-correction loops.

The pipeline has two independent loops, each with its own budget so a
flaky classification cannot eat into the extraction budget (and vice versa):

  * ``classify`` loop — Classifier ↔ Classification Validator
    counter ``classify_retry_count`` / budget ``classify_max_retries``
  * ``extract`` loop  — Extractor ↔ Validator
    counter ``retry_count`` / budget ``max_retries``

In both, the counter means **retries actually performed**: 0 on the first
pass, incremented each time a failed attempt is sent back.  A budget of N
therefore allows N + 1 attempts in total before the document is flagged for
the Dead Letter Queue.

Every node records a failed attempt through ``register_failure`` /
``register_classification_failure`` so the paths cannot drift apart.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import ValidationError

from src.config import settings
from src.schemas.events import CorrectionLog, FailureInfo
from src.schemas.state import PipelineState

LOOP_CLASSIFY = "classify"
LOOP_EXTRACT = "extract"


def summarise_validation_error(exc: ValidationError, tag: str = "SCHEMA") -> str:
    """Compact, LLM-readable description of what broke a Pydantic schema."""
    issues = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        bad = repr(err.get("input"))
        if len(bad) > 80:
            bad = bad[:77] + "..."
        issues.append(f"[{tag}] field='{loc}': {err['msg']} (value={bad})")
    return "; ".join(issues)


def latest_log(state: PipelineState, loop: str) -> Optional[CorrectionLog]:
    """Most recent correction log of the given loop (logs without a loop are extract)."""
    for log in reversed(state.get("correction_logs", [])):
        if log.get("loop", LOOP_EXTRACT) == loop:
            return log
    return None


def _register(
    state: PipelineState,
    *,
    loop: str,
    count_key: str,
    budget_key: str,
    default_budget: int,
    error_summary: str,
    report: str,
    exhausted_message: str,
    failure: Optional[FailureInfo],
) -> dict[str, Any]:
    retries_used = state.get(count_key, 0)
    budget = state.get(budget_key, default_budget)

    logs: list[CorrectionLog] = list(state.get("correction_logs", []))
    same_loop = sum(1 for log in logs if log.get("loop", LOOP_EXTRACT) == loop)
    logs.append(
        {
            "cycle": same_loop + 1,
            "error_summary": error_summary,
            "raw_feedback": report,
            "loop": loop,
        }
    )

    update: dict[str, Any] = {"correction_logs": logs}
    if retries_used >= budget:
        update[count_key] = retries_used
        update["error"] = exhausted_message
        if failure is not None:
            update["failure"] = failure
    else:
        update[count_key] = retries_used + 1
    return update


def register_failure(
    state: PipelineState,
    *,
    error_summary: str,
    report: str,
    exhausted_message: str,
    failure: Optional[FailureInfo] = None,
) -> dict[str, Any]:
    """
    Record one failed *extraction* attempt and decide whether a retry is allowed.

    Returns the state update to merge:
      - ``correction_logs`` with a new entry (``cycle`` is the 1-indexed
        number of the failed attempt within the extract loop);
      - ``retry_count`` + 1 while budget remains;
      - once the budget is spent, ``retry_count`` stays at ``max_retries``,
        ``error`` is set to ``exhausted_message`` and ``failure`` (if given)
        is recorded (→ DLQ).
    """
    return _register(
        state,
        loop=LOOP_EXTRACT,
        count_key="retry_count",
        budget_key="max_retries",
        default_budget=3,
        error_summary=error_summary,
        report=report,
        exhausted_message=exhausted_message,
        failure=failure,
    )


def register_classification_failure(
    state: PipelineState,
    *,
    error_summary: str,
    report: str,
    exhausted_message: str,
    failure: Optional[FailureInfo] = None,
) -> dict[str, Any]:
    """Same as ``register_failure`` but for the classify loop and its own budget."""
    return _register(
        state,
        loop=LOOP_CLASSIFY,
        count_key="classify_retry_count",
        budget_key="classify_max_retries",
        default_budget=settings.classify_max_retries,
        error_summary=error_summary,
        report=report,
        exhausted_message=exhausted_message,
        failure=failure,
    )
