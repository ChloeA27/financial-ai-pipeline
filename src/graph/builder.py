"""
Graph Builder — constructs the LangGraph state machine for the financial
extraction pipeline with a self-correction loop.

Topology:

               ┌──────────┐
               │  Reader   │
               │   Node    │
               └────┬─────┘
                    │
                    ▼
               ┌──────────┐
               │Classifier│
               │   Node   │
               └────┬─────┘
                    │
                    ▼
               ┌──────────┐
               │ Extractor│
               │   Node   │
               └────┬─────┘
                    │
                    ▼
               ┌──────────┐
               │ Validator│
               │   Node   │
               └────┬─────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
     ┌──────────┐        ┌──────────┐
     │  PASS    │        │  FAIL    │
     │ (output) │        │(retry ≤  │
     └──────────┘        │ max?)    │
                          └────┬─────┘
                               │
                      ┌────────┴────────┐
                      ▼                 ▼
                 ┌──────────┐     ┌──────────┐
                 │ Extractor│     │   ERROR  │
                 │ (retry)  │     │ (abort)  │
                 └──────────┘     └──────────┘
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from src.nodes.reader import reader_node
from src.nodes.classifier import classifier_node
from src.nodes.extractor import extractor_node
from src.nodes.validator import validator_node
from src.state.pipeline_state import PipelineState


# ── Conditional edges: self-correction routers ──


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
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("validator", validator_node)

    # 3. Define edges (linear forward flow)
    workflow.set_entry_point("reader")
    workflow.add_edge("reader", "classifier")
    workflow.add_edge("classifier", "extractor")

    # 4. Conditional edges: self-correction loop
    #    (a) after extraction: schema-parse failures retry without the Validator
    workflow.add_conditional_edges(
        source="extractor",
        path=_route_after_extraction,
        path_map={
            "validator": "validator",
            "extractor": "extractor",
            END: END,
        },
    )
    #    (b) after validation: rule failures retry via the Extractor
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
