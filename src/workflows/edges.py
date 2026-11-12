"""
Conditional edges — pure routing functions.

Each router reads the state a node just returned and names the next node.
Routers do no work and call no LLM. The nodes do the work, and the decision of
where to go next lives here only. ``graph.py`` connects routers to nodes.

Signals the routers read (all set by nodes):

  * ``error``                  fatal: the document stops (the node also set ``failure``)
  * ``classification_passed``  False = the classification was rejected, True = accepted
  * ``validation_passed``      False = the extraction was rejected, True = accepted
  * ``*_retry_count`` / ``*_max_retries``  retries already performed / allowed
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END

from src.config import settings
from src.schemas.state import PipelineState

# ── Classify loop ────────────────────────────────────────────────────


def route_after_classifier(
    state: PipelineState,
) -> Literal["classification_validator", "classifier", "__end__"]:
    """
    Decide where to go after the Classifier Node finishes.

    - fatal error (LLM failure, or classify budget spent on unparsable output) → END
    - ``classification_passed == False`` → the output could not even be parsed;
      the Classifier already logged the cycle, so skip the validator and
      retry (or end) via the shared classify router.
    - otherwise → Classification Validator
    """
    if state.get("error"):
        return END
    if state.get("classification_passed") is False:
        return route_after_classification_validation(state)
    return "classification_validator"


def route_after_classification_validation(
    state: PipelineState,
) -> Literal["extractor", "classifier", "__end__"]:
    """
    Decide where to go after the Classification Validator.

    - error set (budget spent, or terminal label none / unclear) → END
    - ``classification_passed == True`` → Extractor
    - rejected and ``classify_retry_count <= classify_max_retries`` → Classifier
      (the counter is retries already *performed*, as in the extract loop)
    """
    if state.get("error"):
        return END
    if state.get("classification_passed") is True:
        return "extractor"

    retry_count = state.get("classify_retry_count", 0)
    max_retries = state.get("classify_max_retries", settings.classify_max_retries)
    if retry_count <= max_retries:
        return "classifier"

    # Defensive: budget overrun without an error flag
    return END


# ── Extract loop ─────────────────────────────────────────────────────


def route_after_extraction(
    state: PipelineState,
) -> Literal["extraction_validator", "extractor", "__end__"]:
    """
    Decide where to go after the Extractor Node finishes.

    - fatal error (LLM/transport failure, or retries exhausted on schema
      parse failures)                        → END
    - ``validation_passed == False``         → the LLM output could not even be
      parsed into the schema; the Extractor already logged the correction
      cycle, so skip the Extraction Validator and retry (or end) via the shared router.
    - otherwise                              → Extraction Validator
    """
    if state.get("error"):
        return END
    if state.get("validation_passed") is False:
        route = route_after_extraction_validation(state)
        return END if route == "error" else route
    return "extraction_validator"


def route_after_extraction_validation(
    state: PipelineState,
) -> Literal["extractor", "__end__", "error"]:
    """
    Decide where to go after the Extraction Validator Node finishes.

    - validation_passed == True  → END (document is good, output)
    - validation_passed == False & retry_count <= max_retries
                                → Extractor Node (self-correction loop)
    - has fatal error           → END (abort; this includes "retry budget
                                  spent", which the failing node flags itself)

    ``retry_count`` is the number of retries already *performed*, so a
    count equal to ``max_retries`` means the last allowed retry is still
    about to run; only a count above it is out of budget.
    """
    # Fatal error → abort
    if state.get("error"):
        return "error"

    validation_passed = state.get("validation_passed")
    if validation_passed is True:
        return END  # All good — exit

    # Validation failed → check retry budget
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count <= max_retries:
        return "extractor"  # One more attempt

    # Defensive: budget overrun without an error flag
    return "error"
