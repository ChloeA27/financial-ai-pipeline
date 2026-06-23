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


# ── Conditional edge: self-correction router ──


def _route_after_validation(
    state: PipelineState,
) -> Literal["extractor", "__end__", "error"]:
    """
    Decide where to go after the Validator Node finishes.

    - validation_passed == True  → END (document is good, output)
    - validation_passed == False & retry_count < max_retries
                                → Extractor Node (self-correction loop)
    - validation_passed == False & retry_count >= max_retries
                                → END (flagged as error, exhausted retries)
    - has fatal error           → END (abort)
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

    if retry_count < max_retries:
        return "extractor"  # One more attempt

    # Exhausted all retries
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
    workflow.add_edge("extractor", "validator")

    # 4. Conditional edge: self-correction loop
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
