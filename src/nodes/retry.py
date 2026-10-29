"""
Shared retry bookkeeping for the self-correction loop.

``retry_count`` means **retries actually performed**: 0 on the first pass,
incremented each time a failed attempt is sent back to the Extractor.
``max_retries`` is the retry budget, so a document gets ``max_retries + 1``
extraction attempts in total before it is flagged for the Dead Letter Queue.

Both the Validator (rule failures) and the Extractor (LLM output that does
not parse into the schema) record a failed attempt through
``register_failure`` so the two paths cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from src.state.pipeline_state import CorrectionLog, PipelineState


def register_failure(
    state: PipelineState,
    *,
    error_summary: str,
    report: str,
    exhausted_message: str,
) -> dict[str, Any]:
    """
    Record one failed attempt and decide whether a retry is still allowed.

    Returns the state update to merge:
      - ``correction_logs`` with a new entry (``cycle`` is the 1-indexed
        number of the failed attempt);
      - ``retry_count`` + 1 while budget remains;
      - once the budget is spent, ``retry_count`` stays at ``max_retries``
        and ``error`` is set to ``exhausted_message`` (→ DLQ).
    """
    retries_used = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    logs: list[CorrectionLog] = list(state.get("correction_logs", []))
    logs.append(
        {
            "cycle": len(logs) + 1,
            "error_summary": error_summary,
            "raw_feedback": report,
        }
    )

    update: dict[str, Any] = {"correction_logs": logs}
    if retries_used >= max_retries:
        update["retry_count"] = retries_used
        update["error"] = exhausted_message
    else:
        update["retry_count"] = retries_used + 1
    return update
