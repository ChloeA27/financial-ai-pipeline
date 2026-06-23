"""
RAG Retriever — fetch semantically similar historical extractions as few-shot
context for the LLM.

Usage::

    from src.rag.retriever import retrieve_context

    context = await retrieve_context(doc_type="Dividend", query_text=raw_content[:1000])
    if context:
        prompt += context

The retriever:
    1. Resolves the per-doc-type ChromaDB collection
    2. Queries with semantic similarity (Top-K)
    3. Formats results into a structured few-shot string
    4. Returns empty string if no historical data exists

Design note:
    ``retrieve_context`` is an ``async def`` function because ``collection.query()``
    is a synchronous ChromaDB call that will block the event loop if called
    directly from the async ``extractor_node``. We use
    ``asyncio.get_event_loop().run_in_executor()`` to offload it to a thread
    pool, keeping the pipeline event loop responsive.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from src.config import settings
from src.rag.chroma_client import get_or_create_collection, collection_count


def _format_extraction(idx: int, doc_text: str, metadata: dict[str, Any]) -> str:
    """
    Format a single extraction as a structured few-shot example block.

    Each block includes the extraction type header + the JSON fields,
    making it easy for the LLM to pattern-match.
    """
    return (
        f"─── Reference Extraction #{idx} ───\n"
        f"Type: {metadata.get('doc_type', 'Unknown')}\n"
        f"{doc_text}\n"
    )


async def retrieve_context(
    doc_type: str,
    query_text: str,
    top_k: int | None = None,
) -> str:
    """
    Retrieve semantically similar extractions as few-shot context.

    Args:
        doc_type: Document type to search within (e.g. ``"M&A"``, ``"Dividend"``).
        query_text: The raw text of the current document to find similar examples.
        top_k: Number of results to return (default: ``settings.rag_top_k``).

    Returns:
        A formatted string of reference extractions, or empty string if
        no historical data exists or an error occurs.
    """
    top_k = top_k or settings.rag_top_k

    # ── Graceful degradation: if no data, return empty ──
    # Note: collection_count() calls get_or_create_collection() which raises
    # ValueError for unknown doc_types. We catch Exception broadly so that
    # unknown doc_types also return "" gracefully (see test_retrieve_unknown_type).
    try:
        count = collection_count(doc_type)
    except Exception:
        count = 0

    if count == 0:
        logger.debug(
            "🔍  [RAG Retriever] collection '{}' is empty → no context",
            doc_type,
        )
        return ""

    # ── Query ChromaDB (async via thread pool — ChromaDB is sync) ──
    try:
        collection = get_or_create_collection(doc_type)
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: collection.query(
                query_texts=[query_text[:1000]],
                n_results=min(top_k, count),
            ),
        )
    except Exception as exc:
        logger.warning("⚠️  [RAG Retriever] query failed: {}", exc)
        return ""

    # ── Parse results ──
    if not results or not results.get("ids") or not results["ids"][0]:
        return ""

    documents = results["documents"][0] if results.get("documents") else []
    metadatas = results["metadatas"][0] if results.get("metadatas") else []
    distances = results["distances"][0] if results.get("distances") else []

    # ── Format blocks ──
    blocks: list[str] = []
    for i, (doc_text, meta) in enumerate(zip(documents, metadatas), start=1):
        block = _format_extraction(i, doc_text, meta)
        blocks.append(block)

    if not blocks:
        return ""

    # Distances are always available in query results
    avg_distance = sum(distances) / len(distances) if distances else 0.0
    logger.debug(
        "🔍  [RAG Retriever] {} | top-{} returned (avg_dist={:.4f})",
        doc_type,
        len(blocks),
        avg_distance,
    )

    header = (
        f"\n\n─── RETRIEVED HISTORICAL EXTRACTIONS ({doc_type}) ───\n"
        "Below are previously validated extractions for the same document "
        "type. Use them as format and content references.\n\n"
    )
    return header + "\n".join(blocks)
