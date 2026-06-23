"""
Tests for the RAG module — embedding, ChromaDB persistence, and retrieval.

Each test gets its own tmp_path ChromaDB directory via ``override_chroma_path``,
ensuring zero cross-test contamination and no writes to the real ``data/chromadb/``.
"""

from __future__ import annotations

from typing import Any

import numpy
import pytest

from src.rag.chroma_client import (
    COLLECTION_MAP,
    get_chroma_client,
    get_or_create_collection,
    collection_count,
    list_collections,
    override_chroma_path,
)
from src.rag.embedder import get_embedding_function, embed_text
from src.rag.repository import save_to_vector_store
from src.rag.retriever import retrieve_context


# ════════════════════════════════════════════════════════════════
#  Fixtures
# ════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _chroma_tmp_path(tmp_path: pytest.TempPathFactory) -> None:
    """
    Auto-fixture: gives every test its own ChromaDB directory.

    Calls ``override_chroma_path(str(tmp_path))`` before each test, which
    resets the singleton client so that ``get_chroma_client()`` points to
    the temporary directory instead of ``settings.chromadb_path``.

    This mirrors the ``tmp_db`` pattern in ``test_storage.py`` for SQLite
    isolation — no test writes to the real ``data/chromadb/``.
    """
    override_chroma_path(str(tmp_path))


@pytest.fixture
def sample_manda_vector_data() -> dict[str, Any]:
    """A validated M&A extraction dict ready for vector store."""
    return {
        "doc_id": "test-manda-001",
        "acquirer": "Microsoft Corporation",
        "target": "Activision Blizzard, Inc.",
        "total_value_usd": 68700000000.0,
        "stake_percentage": 100.0,
        "requires_shareholder_approval": True,
        "payment_method": "Cash",
        "announcement_date": "2022-01-18",
        "expected_close_date": "2023-06-30",
        "metadata": {
            "doc_type": "M&A",
            "source_path": "/fake/path/ma_test.txt",
            "pipeline_version": "1.0.0",
            "model_name": "test-model",
            "processed_at": "2026-06-23T00:00:00",
            "retry_count": 0,
        },
        "raw_text_snippet": "Microsoft Corp. today announced...",
    }


@pytest.fixture
def sample_dividend_vector_data() -> dict[str, Any]:
    """A validated Dividend extraction dict ready for vector store."""
    return {
        "doc_id": "test-dividend-001",
        "ticker": "AAPL",
        "declaration_date": "2026-04-30",
        "dividend_cash_amount": 0.52,
        "currency": "USD",
        "record_date": "2026-05-12",
        "ex_dividend_date": "2026-05-08",
        "payment_date": "2026-05-21",
        "dividend_type": "Regular Cash",
        "frequency": "Quarterly",
        "metadata": {
            "doc_type": "Dividend",
            "source_path": "/fake/path/dividend_test.txt",
            "pipeline_version": "1.0.0",
            "model_name": "test-model",
            "processed_at": "2026-06-23T00:00:00",
            "retry_count": 0,
        },
        "raw_text_snippet": "Apple Inc. today announced a quarterly dividend...",
    }


# ════════════════════════════════════════════════════════════════
#  Tests: Embedder
# ════════════════════════════════════════════════════════════════


class TestEmbedder:
    """SentenceTransformer embedding function correctness."""

    def test_embedding_dimension(self) -> None:
        """all-MiniLM-L6-v2 produces 384-dimensional vectors."""
        vector = embed_text("Hello, world!")
        assert len(vector) == 384
        assert all(isinstance(v, (float, int, numpy.floating)) for v in vector)

    def test_embedding_function_callable(self) -> None:
        """get_embedding_function returns a callable that works with ChromaDB."""
        fn = get_embedding_function()
        result = fn(["test text"])
        assert len(result) == 1
        assert len(result[0]) == 384

    def test_semantic_similarity(self) -> None:
        """Similar texts produce similar vectors (cosine > 0.5)."""
        vec_a = embed_text("Apple declared a dividend")
        vec_b = embed_text("Apple board announces dividend")
        vec_c = embed_text("The weather is nice today")

        # Cosine similarity
        def cosine_sim(v1: list[float], v2: list[float]) -> float:
            dot = sum(a * b for a, b in zip(v1, v2))
            n1 = sum(a * a for a in v1) ** 0.5
            n2 = sum(b * b for b in v2) ** 0.5
            return dot / (n1 * n2)

        sim_ab = cosine_sim(vec_a, vec_b)
        sim_ac = cosine_sim(vec_a, vec_c)

        assert sim_ab > 0.5, f"Similar texts should have high cosine sim, got {sim_ab}"
        assert sim_ac < sim_ab, "Unrelated texts should be less similar than related ones"


# ════════════════════════════════════════════════════════════════
#  Tests: ChromaDB Client
# ════════════════════════════════════════════════════════════════


class TestChromaClient:
    """ChromaDB client and collection management."""

    def test_collection_map_coverage(self) -> None:
        """COLLECTION_MAP covers the known doc_types."""
        assert "M&A" in COLLECTION_MAP
        assert "Dividend" in COLLECTION_MAP
        assert COLLECTION_MAP["M&A"] == "extractions_manda"
        assert COLLECTION_MAP["Dividend"] == "extractions_dividend"

    def test_get_or_create_collection(self) -> None:
        """Getting a collection creates it with the correct name."""
        col = get_or_create_collection("M&A")
        assert col.name == "extractions_manda"

        col2 = get_or_create_collection("M&A")
        assert col2.name == col.name  # idempotent

    def test_get_or_create_unknown_type(self) -> None:
        """Unknown doc_type raises ValueError."""
        with pytest.raises(ValueError, match="No ChromaDB collection registered"):
            get_or_create_collection("Unknown")

    def test_collection_count_empty(self) -> None:
        """Empty collection reports count 0."""
        assert collection_count("M&A") == 0

    def test_per_type_isolation(self) -> None:
        """M&A and Dividend live in separate collections."""
        col_manda = get_or_create_collection("M&A")
        col_div = get_or_create_collection("Dividend")

        col_manda.add(
            ids=["doc1"], documents=["M&A test"], metadatas=[{"doc_type": "M&A"}]
        )

        assert col_manda.count() == 1
        assert col_div.count() == 0

    def test_list_collections(self) -> None:
        """list_collections shows created collections."""
        get_or_create_collection("M&A")
        get_or_create_collection("Dividend")
        names = list_collections()
        assert "extractions_manda" in names
        assert "extractions_dividend" in names


# ════════════════════════════════════════════════════════════════
#  Tests: RAG Repository (save_to_vector_store)
# ════════════════════════════════════════════════════════════════


class TestRagRepository:
    """Writing validated extractions to vector store."""

    async def test_save_manda_to_vector_store(
        self, sample_manda_vector_data: dict[str, Any]
    ) -> None:
        """Saving a validated M&A extraction writes to the correct collection."""
        await save_to_vector_store(sample_manda_vector_data)

        col = get_or_create_collection("M&A")
        assert col.count() == 1

        # Verify the document contains business fields (not pipeline fields)
        results = col.get(ids=["test-manda-001"])
        assert results["documents"] is not None
        doc_text = results["documents"][0]
        assert "Microsoft Corporation" in doc_text
        assert "68700000000" in doc_text
        assert "raw_text_snippet" not in doc_text  # stripped

    async def test_save_dividend_to_vector_store(
        self, sample_dividend_vector_data: dict[str, Any]
    ) -> None:
        """Saving a validated Dividend extraction writes to the correct collection."""
        await save_to_vector_store(sample_dividend_vector_data)

        col = get_or_create_collection("Dividend")
        assert col.count() == 1

        results = col.get(ids=["test-dividend-001"])
        assert results["documents"] is not None
        doc_text = results["documents"][0]
        assert "AAPL" in doc_text
        assert "0.52" in doc_text
        assert "raw_text_snippet" not in doc_text  # stripped

    async def test_save_empty_data(self) -> None:
        """Empty data is skipped gracefully."""
        await save_to_vector_store({})  # should not raise
        assert collection_count("M&A") == 0

    async def test_save_without_metadata(self) -> None:
        """Data without metadata is skipped gracefully."""
        await save_to_vector_store({"doc_id": "no-meta"})  # should not raise
        assert collection_count("M&A") == 0

    async def test_save_idempotent(
        self, sample_manda_vector_data: dict[str, Any]
    ) -> None:
        """Same doc_id upserts (idempotent — no duplicate)."""
        await save_to_vector_store(sample_manda_vector_data)
        await save_to_vector_store(sample_manda_vector_data)

        col = get_or_create_collection("M&A")
        assert col.count() == 1  # upsert, not append


# ════════════════════════════════════════════════════════════════
#  Tests: RAG Retriever (retrieve_context)
# ════════════════════════════════════════════════════════════════


class TestRagRetriever:
    """Semantic retrieval from vector store."""

    async def test_retrieve_empty_returns_empty_string(self) -> None:
        """No data in collection returns empty string (graceful degradation)."""
        result = await retrieve_context("M&A", "test query")
        assert result == ""

    async def test_retrieve_returns_formatted_context(
        self, sample_manda_vector_data: dict[str, Any]
    ) -> None:
        """After saving data, retrieval returns a formatted few-shot string."""
        await save_to_vector_store(sample_manda_vector_data)

        result = await retrieve_context("M&A", "Microsoft acquisition Activision")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Reference Extraction" in result
        assert "Microsoft Corporation" in result

    async def test_retrieve_top_k_respected(
        self, sample_manda_vector_data: dict[str, Any]
    ) -> None:
        """top_k parameter limits the number of returned examples."""
        # Add 2 docs
        data2 = dict(sample_manda_vector_data)
        data2["doc_id"] = "test-manda-002"
        data2["target"] = "Another Target"

        await save_to_vector_store(sample_manda_vector_data)
        await save_to_vector_store(data2)

        result = await retrieve_context("M&A", "acquisition", top_k=1)
        assert result.count("Reference Extraction") == 1

    async def test_no_cross_type_contamination(
        self,
        sample_manda_vector_data: dict[str, Any],
        sample_dividend_vector_data: dict[str, Any],
    ) -> None:
        """M&A query does not return Dividend results."""
        await save_to_vector_store(sample_manda_vector_data)
        await save_to_vector_store(sample_dividend_vector_data)

        manda_result = await retrieve_context("M&A", "dividend payment quarterly")
        dividend_result = await retrieve_context(
            "Dividend", "merger acquisition"
        )

        # M&A collection shouldn't have dividend content
        if manda_result:
            assert "Reference Extraction" in manda_result

        if dividend_result:
            assert "Reference Extraction" in dividend_result

        # Cross-check: M&A retrieval should not contain dividend keywords
        assert "AAPL" not in manda_result

    async def test_retrieve_unknown_type(self) -> None:
        """
        Unknown doc_type returns empty string (graceful degradation).

        This works because ``collection_count("Unknown")`` raises ValueError
        (no collection registered), the broad ``except Exception`` catches it,
        sets count=0, and the function returns "".
        """
        result = await retrieve_context("Unknown", "test")
        assert result == ""
