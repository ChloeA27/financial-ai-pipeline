"""
Graph wiring — connects nodes (``src/nodes``) with routers (``edges.py``).

No routing decisions and no work here: which node runs next is decided by the
routers in ``edges.py``, and what a node does lives in its own module.

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

from langgraph.graph import END, StateGraph

from src.nodes.classification_validator import classification_validator_node
from src.nodes.classifier import classifier_node
from src.nodes.extractor import extractor_node
from src.nodes.reader import reader_node
from src.nodes.validator import validator_node
from src.schemas.state import PipelineState
from src.workflows.edges import (
    route_after_classification_validation,
    route_after_classifier,
    route_after_extraction,
    route_after_validation,
)


def _add_classification_stage(workflow: StateGraph, on_accept: str) -> None:
    """
    Reader → Classifier ⇄ Classification Validator, shared by both graphs.

    ``on_accept`` is where an accepted M&A / Dividend label goes: the Extractor
    in the full pipeline, END in the classification-only graph.
    """
    workflow.add_node("reader", reader_node)
    workflow.add_node("classifier", classifier_node)
    workflow.add_node("classification_validator", classification_validator_node)

    workflow.set_entry_point("reader")
    workflow.add_edge("reader", "classifier")

    # Classify loop: unparsable output / bad evidence → retry the Classifier
    workflow.add_conditional_edges(
        source="classifier",
        path=route_after_classifier,
        path_map={
            "classification_validator": "classification_validator",
            "classifier": "classifier",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        source="classification_validator",
        path=route_after_classification_validation,
        path_map={"extractor": on_accept, "classifier": "classifier", END: END},
    )


def build_pipeline():
    """
    Build and return the compiled LangGraph pipeline.

    Returns:
        A compiled graph whose ``state_schema`` is ``PipelineState``.

    Usage::

        graph = build_pipeline()
        final_state = await graph.ainvoke(initial_state)
    """
    workflow = StateGraph(state_schema=PipelineState)
    _add_classification_stage(workflow, on_accept="extractor")

    workflow.add_node("extractor", extractor_node)
    workflow.add_node("validator", validator_node)

    # Extract loop
    #   after extraction: schema-parse failures retry without the Validator
    workflow.add_conditional_edges(
        source="extractor",
        path=route_after_extraction,
        path_map={
            "validator": "validator",
            "extractor": "extractor",
            END: END,
        },
    )
    #   after validation: rule failures retry via the Extractor
    workflow.add_conditional_edges(
        source="validator",
        path=route_after_validation,
        path_map={
            "extractor": "extractor",
            END: END,
            "error": END,
        },
    )
    return workflow.compile()


def build_classification_graph():
    """
    The classification stage alone: Reader → Classifier ⇄ Classification
    Validator → END.  Same nodes and the same routers as the full pipeline
    (an accepted M&A / Dividend label ends the run instead of moving on to
    the Extractor), so E1 evaluates the real classification code path.
    """
    workflow = StateGraph(state_schema=PipelineState)
    _add_classification_stage(workflow, on_accept=END)
    return workflow.compile()


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

    initial_state = await read_single_file(file_path)
    graph = build_pipeline()
    final_state: PipelineState = await graph.ainvoke(initial_state)
    return final_state
