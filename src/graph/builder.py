"""
Graph Builder — constructs the LangGraph state machine for the financial
extraction pipeline with two self-correction loops.

Topology::

    Reader
      │
      ▼
    Classifier  ◄──────────────┐   classify loop (own retry budget):
      │                        │   output unparsable / evidence quote not in
      ▼                        │   the source → retry with feedback
    Classification Validator ──┘
      │  label M&A / Dividend → continue
      │  label none / unclear → END (DLQ: UNSUPPORTED_DOC_TYPE / CLASSIFICATION_UNCLEAR)
      │  budget spent         → END (DLQ: stage classify / classify_validate)
      ▼
    Extractor  ◄───────────────┐   extract loop (own retry budget):
      │                        │   schema-parse failures skip the Validator,
      ▼                        │   rule failures come back from it
    Validator ─────────────────┘
      │  passed       → END (stored)
      │  budget spent → END (DLQ: stage extract / validate)

Only the Classifier and the Extractor call an LLM; the Reader and both
validators are plain code.  Every fatal exit carries a ``failure`` record
(stage, code, category) that the DLQ stores.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from src.nodes.reader import reader_node
from src.nodes.classifier import classifier_node
from src.nodes.classification_validator import classification_validator_node
from src.nodes.extractor import extractor_node
from src.nodes.validator import validator_node
from src.config import settings
from src.state.pipeline_state import PipelineState


# ── Conditional edges: self-correction routers ──


def _route_after_classifier(
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
        return _route_after_classification_validation(state)
    return "classification_validator"


def _route_after_classification_validation(
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


def _route_after_extraction(
    state: PipelineState,
) -> Literal["validator", "extractor", "__end__"]:
    """
    Decide where to go after the Extractor Node finishes.

    - fatal error (LLM/transport failure, or retries exhausted on schema
      parse failures)                        → END
    - ``validation_passed == False``         → the LLM output could not even be
      parsed into the schema; the Extractor already logged the correction
      cycle, so skip the Validator and retry (or end) via the shared router.
    - otherwise                              → Validator
    """
    if state.get("error"):
        return END
    if state.get("validation_passed") is False:
        route = _route_after_validation(state)
        return END if route == "error" else route
    return "validator"


def _route_after_validation(
    state: PipelineState,
) -> Literal["extractor", "__end__", "error"]:
    """
    Decide where to go after the Validator Node finishes.

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


# ── Graph construction ──


def build_pipeline() -> StateGraph:
    """
    Build and return the compiled LangGraph pipeline.

    Returns:
        A compiled ``StateGraph`` whose ``state_schema`` is ``PipelineState``.

    Usage::

        graph = build_pipeline()
        final_state = await graph.ainvoke(initial_state)
    """
    # 1. Define the graph with PipelineState as the schema
    workflow = StateGraph(state_schema=PipelineState)

    # 2. Register nodes
    workflow.add_node("reader", reader_node)
    workflow.add_node("classifier", classifier_node)
    workflow.add_node("classification_validator", classification_validator_node)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("validator", validator_node)

    # 3. Define edges (linear forward flow)
    workflow.set_entry_point("reader")
    workflow.add_edge("reader", "classifier")

    # 4. Conditional edges: two self-correction loops
    #    (a) classify loop: unparsable output / bad evidence → retry the Classifier
    workflow.add_conditional_edges(
        source="classifier",
        path=_route_after_classifier,
        path_map={
            "classification_validator": "classification_validator",
            "classifier": "classifier",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        source="classification_validator",
        path=_route_after_classification_validation,
        path_map={
            "extractor": "extractor",
            "classifier": "classifier",
            END: END,
        },
    )
    #    (b) extract loop
    #      - after extraction: schema-parse failures retry without the Validator
    workflow.add_conditional_edges(
        source="extractor",
        path=_route_after_extraction,
        path_map={
            "validator": "validator",
            "extractor": "extractor",
            END: END,
        },
    )
    #      - after validation: rule failures retry via the Extractor
    workflow.add_conditional_edges(
        source="validator",
        path=_route_after_validation,
        path_map={
            "extractor": "extractor",
            END: END,
            "error": END,
        },
    )

    # 5. Compile
    graph = workflow.compile()
    return graph


# ── Convenience runner ──


async def run_pipeline(file_path: str) -> PipelineState:
    """
    High-level entry point: read a file and run it through the full pipeline.

    Args:
        file_path: Path to a raw text announcement file.

    Returns:
        Final PipelineState after the graph finishes.

    Usage::

        result = await run_pipeline("raw_data/sample/ma_microsoft_activision.txt")
        print(result["validation_passed"])
        print(result["extracted_data"])
    """
    from src.nodes.reader import read_single_file

    # Bootstrap initial state
    initial_state = await read_single_file(file_path)

    # Compile graph
    graph = build_pipeline()

    # Execute
    final_state: PipelineState = await graph.ainvoke(initial_state)
    return final_state
