"""
ChromaDB client management — singleton persistent client per process.

Each document type gets its own collection to keep vector spaces isolated:

    extractions_manda        — M&A extraction results
    extractions_dividend     — Dividend extraction results

Usage::

    from src.rag.chroma_client import get_chroma_client, get_or_create_collection

    client = get_chroma_client()
    collection = get_or_create_collection("extractions_manda")
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from loguru import logger

from src.config import settings
from src.rag.embedder import get_embedding_function


# ── Per-doc-type collection names ──

COLLECTION_MAP: dict[str, str] = {
    "M&A": "extractions_manda",
    "Dividend": "extractions_dividend",
}

_COLLECTION_NAMES: set[str] = set(COLLECTION_MAP.values())

# Reverse lookup: collection name → doc_type
_COLLECTION_TO_DOC_TYPE: dict[str, str] = {v: k for k, v in COLLECTION_MAP.items()}


# Singleton client — process-level; reset for testing via ``override_chroma_path()``.
_client: PersistentClient | None = None
_embedding_function = None
# Override: when set (e.g. from test fixtures), next get_chroma_client() call will
# re-initialise the singleton at this path instead of settings.chromadb_path.
_override_path: str | None = None


def override_chroma_path(tmp_path: str) -> None:
    """
    Override the ChromaDB storage path for test isolation.

    Call *before* ``get_chroma_client()`` is invoked. Resets the singleton
    so the next call creates a fresh client pointing to ``tmp_path``.

    Usage in tests::

        override_chroma_path(str(tmp_path))
        client = get_chroma_client()   # ← uses tmp_path, not settings.chromadb_path

    To reset back to the default from settings::

        override_chroma_path(None)
    """
    global _client, _embedding_function, _override_path
    _client = None
    _embedding_function = None
    _override_path = tmp_path


def get_chroma_client() -> PersistentClient:
    """
    Return the singleton ChromaDB PersistentClient.

    The database path resolution order:
      1. ``override_chroma_path()`` if set (for test isolation)
      2. ``settings.chromadb_path`` (default ``data/chromadb``)

    All collections share the same embedding function (``all-MiniLM-L6-v2``).
    """
    global _client, _embedding_function
    if _client is None:
        db_path = Path(_override_path or settings.chromadb_path)
        db_path.mkdir(parents=True, exist_ok=True)

        _embedding_function = get_embedding_function()

        _client = chromadb.PersistentClient(path=str(db_path))
        logger.info("🗄️  ChromaDB client initialised at '{}'", db_path)

    return _client


def get_or_create_collection(doc_type: str) -> Collection:
    """
    Get or create a per-doc-type ChromaDB collection.

    Args:
        doc_type: One of ``"M&A"``, ``"Dividend"``, or any key in ``COLLECTION_MAP``.

    Returns:
        A ChromaDB ``Collection`` object with the pre-configured embedding function.

    Raises:
        ValueError: If ``doc_type`` is not registered in ``COLLECTION_MAP``.
    """
    collection_name = COLLECTION_MAP.get(doc_type)
    if collection_name is None:
        raise ValueError(
            f"No ChromaDB collection registered for doc_type='{doc_type}'. "
            f"Available: {list(COLLECTION_MAP.keys())}"
        )

    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=_embedding_function,
    )
    return collection


def list_collections() -> list[str]:
    """List all existing ChromaDB collection names (for debugging / dashboard)."""
    client = get_chroma_client()
    return [c.name for c in client.list_collections()]


def collection_count(doc_type: str) -> int:
    """Return the number of documents in a given doc_type collection."""
    collection = get_or_create_collection(doc_type)
    return collection.count()
