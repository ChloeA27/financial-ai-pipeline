"""
Embedding function — wraps ``sentence-transformers`` for ChromaDB compatibility.

Uses ``all-MiniLM-L6-v2`` (384-dimensional, ~10 ms per document on CPU).

This module provides:

- ``get_embedding_function()`` → a ``chromadb.EmbeddingFunction`` subclass
  with full ChromaDB compatibility (has ``.name()`` method).

- ``embed_text()`` → direct embedding for debugging / standalone use.

Usage::

    from src.rag.embedder import get_embedding_function

    ef = get_embedding_function()
    collection = client.create_collection("my_collection", embedding_function=ef)
"""

from __future__ import annotations

from chromadb import EmbeddingFunction
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger

from src.config import settings


# Lazy singleton: ChromaDB's SentenceTransformerEmbeddingFunction already wraps
# sentence-transformers internally and provides the .name() method ChromaDB 1.5+ needs.
#
# Thread/process-safety note: This is a process-level singleton (module global).
# Unlike ChromaDB client state, the embedding function is **pure** (stateless) —
# it maps str → list[float] with no side effects. Resetting it across tests would
# be redundant; the loaded model is read-only and cached by sentence-transformers.
_EF: SentenceTransformerEmbeddingFunction | None = None


def get_embedding_function() -> EmbeddingFunction:
    """
    Return a singleton ChromaDB-compatible embedding function.

    The returned object is a ``SentenceTransformerEmbeddingFunction``
    (from ``chromadb.utils.embedding_functions``) which has all the
    required ChromaDB protocol methods (``.name()``, ``__call__``, etc.).
    """
    global _EF
    if _EF is None:
        model_name = settings.rag_embedding_model
        logger.info("🔤  Loading embedding function '{}' …", model_name)
        _EF = SentenceTransformerEmbeddingFunction(model_name=model_name)
        logger.success("✅  Embedding function '{}' loaded (dim=384)", model_name)
    return _EF


def embed_text(text: str) -> list[float]:
    """
    Embed a single text string into a 384-dimensional vector.

    Useful for debugging or direct similarity computation outside ChromaDB.
    """
    ef = get_embedding_function()
    result = ef([text])
    return result[0]
