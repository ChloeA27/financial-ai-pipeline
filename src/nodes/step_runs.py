"""
Builders for ``StepRun`` audit records (one per node execution).

Outcomes:
  passed   — the step succeeded and the document continues (or is stored)
  rejected — the step's output failed a check; a retry follows
  fatal    — the document stops at this step (``error_code`` says why; this
             includes the expected out-of-scope / unclear endings)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from src.llm.client import LLMCallInfo
from src.schemas.events import StepRun
from src.schemas.state import PipelineState

OUTCOME_PASSED = "passed"
OUTCOME_REJECTED = "rejected"
OUTCOME_FATAL = "fatal"


def current_attempt(state: PipelineState, loop: str) -> int:
    """1-indexed attempt number of the step about to run in ``loop``."""
    key = "classify_retry_count" if loop == "classify" else "retry_count"
    return state.get(key, 0) + 1


def outcome_of(update: dict[str, Any]) -> str:
    """rejected while a retry is still allowed, fatal once ``error`` is set."""
    return OUTCOME_FATAL if update.get("error") else OUTCOME_REJECTED


def make_step_run(
    state: PipelineState,
    *,
    stage: str,
    loop: str,
    attempt: int,
    outcome: str,
    output: Any = None,
    failure: Optional[dict[str, Any]] = None,
    llm_call: Optional[LLMCallInfo] = None,
    error_code: Optional[str] = None,
    error_category: Optional[str] = None,
) -> StepRun:
    """
    Assemble one audit record.

    ``failure`` (a FailureInfo) fills ``error_code`` / ``error_category``
    unless they are given explicitly (used for retryable rejections, which
    have no FailureInfo).
    """
    call = llm_call or {}
    return StepRun(
        run_id=state.get("run_id", ""),
        source_hash=state.get("source_hash", ""),
        file_path=state.get("file_path", ""),
        stage=stage,
        loop=loop,
        attempt=attempt,
        outcome=outcome,
        error_code=error_code or (failure or {}).get("code"),
        error_category=error_category or (failure or {}).get("category"),
        output=output,
        model_name=call.get("model_name"),
        prompt_version=call.get("prompt_version"),
        raw_response=call.get("raw_response"),
        prompt_tokens=call.get("prompt_tokens"),
        completion_tokens=call.get("completion_tokens"),
        total_tokens=call.get("total_tokens"),
        latency_ms=call.get("latency_ms"),
        started_at=datetime.now(timezone.utc).isoformat(),
    )
