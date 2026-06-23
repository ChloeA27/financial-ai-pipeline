"""
RAG Repository — write validated extraction results to ChromaDB vector store.

Called from ``ExtractionRepository._upsert_extraction()`` after a successful
SCD Type 2 write — ensures only validated data enters the semantic search index.

Storage schema (per document):

    - id:          doc_id (aligned with SQLite doc_id)
    - document:    JSON-serialised business fields (no metadata, no raw_text)
    - metadata:    {doc_id, doc_type, file_path, pipeline_version, created_at}
    - collection:  extractions_{doc_type_lower}  (per-type isolation)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from src.rag.chroma_client import get_or_create_collection


def _now() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _strip_pipeline_fields(data: dict[str, Any]) -> dict[str, Any]:
    """
    Remove pipeline-controlled fields before storing in vector store.

    We only want business data in the embedding document — metadata,
    raw_text_snippet, and doc_id are pipeline artifacts that would
    confuse the LLM when retrieved as few-shot context.
    """
    skip_keys = {"doc_id", "metadata", "raw_text_snippet"}
    return {k: v for k, v in data.items() if k not in skip_keys}


async def save_to_vector_store(extracted_data: dict[str, Any]) -> None:
    """
    Save a validated extraction result to the per-doc-type ChromaDB collection.

    Args:
        extracted_data: The serialised extraction dict (as stored in
            ``PipelineState.extracted_data``). Must contain ``doc_id``,
            ``metadata.doc_type`` and business fields.
    """
    if not extracted_data:
        logger.warning("⏭️  save_to_vector_store: extracted_data is empty → skipping")
        return

    doc_id = extracted_data.get("doc_id", "")
    if not doc_id:
        logger.warning("⏭️  save_to_vector_store: missing doc_id → skipping")
        return

    metadata = extracted_data.get("metadata", {}) or {}
    doc_type = metadata.get("doc_type", "")

    if not doc_type:
        logger.warning(
            "⏭️  save_to_vector_store: missing doc_type in metadata → skipping"
        )
        return

    # ── Build the document text (business fields only) ──
    business_data = _strip_pipeline_fields(extracted_data)
    document_text = json.dumps(business_data, ensure_ascii=False, indent=2)

    # ── ChromaDB metadata ──
    chroma_metadata = {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "file_path": metadata.get("source_path", ""),
        "pipeline_version": metadata.get("pipeline_version", ""),
        "created_at": _now(),
    }

    # ── Write to per-type collection (upsert to handle re-runs) ──
    try:
        collection = get_or_create_collection(doc_type)
        collection.upsert(
            ids=[doc_id],
            documents=[document_text],
            metadatas=[chroma_metadata],
        )
        logger.info(
            "🧠  [Vector Store] upserted doc_id={} → collection='{}' ({} docs total)",
            doc_id,
            collection.name,
            collection.count(),
        )
    except Exception as exc:
        # Non-fatal — vector store failure should not crash pipeline
        logger.error("⚠️  save_to_vector_store failed for doc_id={}: {}", doc_id, exc)
