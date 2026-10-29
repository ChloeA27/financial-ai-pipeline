"""
Tests for the RAG module — embedding, ChromaDB persistence, and retrieval.

Every test gets its own tmp_path ChromaDB directory and an offline fake embedding
function from the autouse fixture in ``conftest.py`` (zero cross-test contamination,
no writes to the real ``data/chromadb/``). ``TestEmbedder`` is marked
``real_embedder`` and uses the actual sentence-transformers model instead.
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
from src.rag.repository import delete_from_vector_store, save_to_vector_store
from src.rag.retriever import retrieve_context


# ════════════════════════════════════════════════════════════════
#  Fixtures
# ════════════════════════════════════════════════════════════════


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


@pytest.mark.real_embedder
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
        assert (
            sim_ac < sim_ab
        ), "Unrelated texts should be less similar than related ones"


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
        dividend_result = await retrieve_context("Dividend", "merger acquisition")

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


class TestRetrieverSelfExclusion:
    """A document must not be retrieved as its own few-shot example."""

    @staticmethod
    def _doc(doc_id: str, path: str, acquirer: str) -> dict[str, Any]:
        return {
            "doc_id": doc_id,
            "acquirer": acquirer,
            "target": "Target Corp",
            "metadata": {"doc_type": "M&A", "source_path": path},
        }

    async def test_same_file_is_excluded(self) -> None:
        await save_to_vector_store(self._doc("a", "/data/a.txt", "AlphaAcquirer"))
        await save_to_vector_store(self._doc("b", "/data/b.txt", "BetaAcquirer"))

        ctx = await retrieve_context(
            "M&A", "acquisition", exclude_file_path="/data/a.txt"
        )
        assert "BetaAcquirer" in ctx
        assert "AlphaAcquirer" not in ctx

    async def test_only_own_file_indexed_returns_empty(self) -> None:
        await save_to_vector_store(self._doc("a", "/data/a.txt", "AlphaAcquirer"))
        ctx = await retrieve_context(
            "M&A", "acquisition", exclude_file_path="/data/a.txt"
        )
        assert ctx == ""

    async def test_without_exclusion_everything_is_eligible(self) -> None:
        await save_to_vector_store(self._doc("a", "/data/a.txt", "AlphaAcquirer"))
        ctx = await retrieve_context("M&A", "acquisition")
        assert "AlphaAcquirer" in ctx

    async def test_exclusion_still_fills_top_k_from_other_files(self) -> None:
        await save_to_vector_store(self._doc("a", "/data/a.txt", "AlphaAcquirer"))
        for i in range(3):
            await save_to_vector_store(
                self._doc(f"o{i}", f"/data/o{i}.txt", f"Other{i}Acquirer")
            )
        ctx = await retrieve_context(
            "M&A", "acquisition", top_k=3, exclude_file_path="/data/a.txt"
        )
        assert ctx.count("Reference Extraction") == 3
        assert "AlphaAcquirer" not in ctx

    async def test_extractor_excludes_its_own_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """extractor_node must pass its file_path through to the retriever."""
        from src.nodes import extractor as extractor_module

        seen: dict[str, Any] = {}

        async def spy(**kwargs: Any) -> str:
            seen.update(kwargs)
            return ""

        class _Client:
            async def generate_structured(self, **_: Any) -> Any:
                from src.schemas.extraction.manda import MandaExtraction

                return MandaExtraction(acquirer="A", target="B")

            async def close(self) -> None:
                return None

        monkeypatch.setattr(extractor_module, "retrieve_context", spy)
        monkeypatch.setattr(extractor_module, "LLMClient", _Client)

        await extractor_module.extractor_node(
            {
                "file_path": "/data/a.txt",
                "raw_content": "some text",
                "doc_type": "M&A",
                "correction_logs": [],
                "retry_count": 0,
            }
        )
        assert seen["exclude_file_path"] == "/data/a.txt"


class TestDeleteFromVectorStore:
    async def test_deletes_only_the_given_doc(
        self, sample_manda_vector_data: dict[str, Any]
    ) -> None:
        await save_to_vector_store(sample_manda_vector_data)
        await save_to_vector_store({**sample_manda_vector_data, "doc_id": "keep"})
        await delete_from_vector_store("test-manda-001", "M&A")
        assert get_or_create_collection("M&A").get()["ids"] == ["keep"]

    async def test_missing_id_or_unknown_type_is_harmless(self) -> None:
        await delete_from_vector_store("nope", "M&A")
        await delete_from_vector_store("nope", "Unknown")
        await delete_from_vector_store("", "M&A")
        await delete_from_vector_store("x", "")
