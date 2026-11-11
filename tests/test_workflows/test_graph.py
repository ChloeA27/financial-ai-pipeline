"""
Tests for the LangGraph graph builder — state construction, conditional
edge routing (self-correction loop), and the high-level run_pipeline().
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

from src.workflows.edges import (
    route_after_classification_validation,
    route_after_classifier,
    route_after_extraction,
    route_after_extraction_validation,
)
from src.workflows.graph import build_pipeline, run_pipeline


# ═══════════════════════════════════════════════════════════════════
#  Conditional edge router
# ═══════════════════════════════════════════════════════════════════


class TestRouteAfterValidation:
    """Unit tests for the route_after_extraction_validation conditional edge function."""

    def test_passed_goes_to_end(self) -> None:
        state: dict[str, Any] = {
            "validation_passed": True,
            "retry_count": 0,
            "max_retries": 3,
            "error": None,
        }
        result = route_after_extraction_validation(state)
        assert result == "__end__"

    def test_failed_still_has_retries(self) -> None:
        """If retry_count < max_retries, route to extractor."""
        state = {
            "validation_passed": False,
            "retry_count": 1,
            "max_retries": 3,
            "error": None,
        }
        result = route_after_extraction_validation(state)
        assert result == "extractor"

    def test_last_allowed_retry_still_runs(self) -> None:
        """retry_count counts retries performed; == max means the last one is due."""
        state = {
            "validation_passed": False,
            "retry_count": 3,
            "max_retries": 3,
            "error": None,
        }
        assert route_after_extraction_validation(state) == "extractor"

    def test_budget_overrun_without_error_flag_routes_to_error(self) -> None:
        state = {
            "validation_passed": False,
            "retry_count": 4,
            "max_retries": 3,
            "error": None,
        }
        assert route_after_extraction_validation(state) == "error"

    def test_exhausted_flag_set_by_node_routes_to_error(self) -> None:
        state = {
            "validation_passed": False,
            "retry_count": 3,
            "max_retries": 3,
            "error": "extraction_validator_node: max retries (3) exhausted.",
        }
        assert route_after_extraction_validation(state) == "error"

    def test_fatal_error_routes_to_error(self) -> None:
        """Fatal error takes precedence over validation status."""
        state = {
            "validation_passed": True,
            "retry_count": 0,
            "max_retries": 3,
            "error": "Something went wrong",
        }
        result = route_after_extraction_validation(state)
        assert result == "error"

    def test_max_retries_at_boundary(self) -> None:
        """retry_count == 0 is still within budget."""
        state = {
            "validation_passed": False,
            "retry_count": 0,
            "max_retries": 3,
            "error": None,
        }
        result = route_after_extraction_validation(state)
        assert result == "extractor"

    def test_max_retries_one_below_limit(self) -> None:
        """retry_count == 2 when max=3 → still eligible."""
        state = {
            "validation_passed": False,
            "retry_count": 2,
            "max_retries": 3,
            "error": None,
        }
        result = route_after_extraction_validation(state)
        assert result == "extractor"

    def test_validation_none_with_no_error(self) -> None:
        """If validation_passed is None (not yet run), what happens?"""
        state = {
            "validation_passed": None,
            "retry_count": 0,
            "max_retries": 3,
            "error": None,
        }
        # validation_passed is None → treated as falsy → check retries
        result = route_after_extraction_validation(state)
        assert result == "extractor"

    def test_validation_passed_is_false_with_retries_left(self) -> None:
        """Explicit False should be treated as failed."""
        state = {
            "validation_passed": False,
            "retry_count": 0,
            "max_retries": 3,
            "error": None,
        }
        result = route_after_extraction_validation(state)
        assert result == "extractor"


# ═══════════════════════════════════════════════════════════════════
#  Post-extraction router (schema-parse failures skip the Validator)
# ═══════════════════════════════════════════════════════════════════


class TestRouteAfterExtraction:
    def test_normal_success_goes_to_validator(self) -> None:
        state = {"validation_passed": None, "retry_count": 0, "max_retries": 3}
        assert route_after_extraction(state) == "extraction_validator"

    def test_parse_failure_with_budget_retries_extractor(self) -> None:
        state = {"validation_passed": False, "retry_count": 1, "max_retries": 3}
        assert route_after_extraction(state) == "extractor"

    def test_error_ends(self) -> None:
        state = {"validation_passed": False, "retry_count": 3, "error": "boom"}
        assert route_after_extraction(state) == "__end__"

    def test_budget_overrun_without_error_flag_still_ends(self) -> None:
        state = {"validation_passed": False, "retry_count": 4, "max_retries": 3}
        assert route_after_extraction(state) == "__end__"


# ═══════════════════════════════════════════════════════════════════
#  Classify-loop routers
# ═══════════════════════════════════════════════════════════════════


class TestRouteAfterClassifier:
    def test_normal_success_goes_to_the_classification_validator(self) -> None:
        state = {"classification_passed": None, "classify_retry_count": 0}
        assert route_after_classifier(state) == "classification_validator"

    def test_parse_failure_with_budget_retries_the_classifier(self) -> None:
        state = {
            "classification_passed": False,
            "classify_retry_count": 1,
            "classify_max_retries": 2,
        }
        assert route_after_classifier(state) == "classifier"

    def test_error_ends(self) -> None:
        state = {"classification_passed": False, "error": "boom"}
        assert route_after_classifier(state) == "__end__"

    def test_budget_overrun_without_error_flag_still_ends(self) -> None:
        state = {
            "classification_passed": False,
            "classify_retry_count": 3,
            "classify_max_retries": 2,
        }
        assert route_after_classifier(state) == "__end__"


class TestRouteAfterClassificationValidation:
    def test_accepted_goes_to_extractor(self) -> None:
        assert (
            route_after_classification_validation({"classification_passed": True})
            == "extractor"
        )

    def test_rejected_with_budget_goes_back_to_classifier(self) -> None:
        state = {
            "classification_passed": False,
            "classify_retry_count": 2,
            "classify_max_retries": 2,
        }
        assert route_after_classification_validation(state) == "classifier"

    def test_error_wins_over_acceptance(self) -> None:
        """Terminal labels (none / unclear) are accepted *and* carry an error."""
        state = {"classification_passed": True, "error": "not a supported type"}
        assert route_after_classification_validation(state) == "__end__"

    def test_exhausted_budget_ends(self) -> None:
        state = {"classification_passed": False, "error": "max retries exhausted"}
        assert route_after_classification_validation(state) == "__end__"


# ═══════════════════════════════════════════════════════════════════
#  Whole graph with a scripted LLM (real parsing, no network)
# ═══════════════════════════════════════════════════════════════════


class TestGraphSelfCorrection:
    @pytest.fixture
    def scripted(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        from tests.test_nodes.test_extractor import FakeOpenAI
        from src.nodes import extractor as extractor_module

        FakeOpenAI.queue = []
        FakeOpenAI.calls = []
        monkeypatch.setattr("src.llm.client.AsyncOpenAI", FakeOpenAI)

        async def _no_rag(**_: Any) -> str:
            return ""

        monkeypatch.setattr(extractor_module, "retrieve_context", _no_rag)
        src = tmp_path / "ma.txt"
        src.write_text("Acme to acquire Widgets Inc. for $1B in cash.")
        return FakeOpenAI, str(src)

    CLASSIFY = (
        '{"doc_type": "M&A", "evidence_quote": "Acme to acquire Widgets Inc. for $1B",'
        ' "confidence": 0.9, "reasoning": "deal"}'
    )
    GOOD = (
        '{"acquirer": "Acme", "target": "Widgets Inc.", "total_value_usd": 1e9,'
        ' "payment_method": "Cash"}'
    )
    BAD_ENUM = '{"acquirer": "Acme", "target": "W", "payment_method": "cash"}'

    async def test_bad_enum_then_good_recovers(self, scripted) -> None:
        fake, path = scripted
        fake.queue = [self.CLASSIFY, self.BAD_ENUM, self.GOOD]

        result = await run_pipeline(path)

        assert result["validation_passed"] is True
        assert result.get("error") is None
        assert result["retry_count"] == 1
        assert len(result["correction_logs"]) == 1
        assert "payment_method" in result["correction_logs"][0]["error_summary"]
        assert result["extracted_data"]["acquirer"] == "Acme"

    async def test_always_bad_exhausts_to_dlq_state(self, scripted) -> None:
        fake, path = scripted
        fake.queue = [self.CLASSIFY] + [self.BAD_ENUM] * 4

        result = await run_pipeline(path)

        assert result["validation_passed"] is False
        assert "max retries (3) exhausted" in result["error"]
        # 1 first attempt + max_retries (3) retries, all failed
        assert result["retry_count"] == 3
        assert [l["cycle"] for l in result["correction_logs"]] == [1, 2, 3, 4]
        assert fake.queue == []  # no extraction attempt beyond the budget

    async def test_parse_failure_then_validator_failure_then_good(
        self, scripted
    ) -> None:
        fake, path = scripted
        no_acquirer = '{"target": "W", "total_value_usd": 5, "payment_method": "Cash"}'
        fake.queue = [self.CLASSIFY, self.BAD_ENUM, no_acquirer, self.GOOD]

        result = await run_pipeline(path)

        assert result["validation_passed"] is True
        assert result["retry_count"] == 2
        assert [l["cycle"] for l in result["correction_logs"]] == [1, 2]

    async def test_max_retries_zero_means_single_attempt(self, scripted) -> None:
        fake, path = scripted
        fake.queue = [self.CLASSIFY, self.BAD_ENUM, self.GOOD]  # GOOD never used
        from src.nodes.reader import read_single_file

        state = await read_single_file(path)
        state["max_retries"] = 0
        result = await build_pipeline().ainvoke(state)

        assert result["validation_passed"] is False
        assert result["retry_count"] == 0
        assert len(result["correction_logs"]) == 1
        assert fake.queue == [self.GOOD]

    async def test_validator_failures_get_full_budget(self, scripted) -> None:
        """Rule failures (not just parse failures) also get max_retries retries."""
        fake, path = scripted
        no_acquirer = '{"target": "W", "total_value_usd": 5, "payment_method": "Cash"}'
        fake.queue = [self.CLASSIFY] + [no_acquirer] * 4

        result = await run_pipeline(path)

        assert result["validation_passed"] is False
        assert result["retry_count"] == 3
        assert len(result["correction_logs"]) == 4
        assert "extraction_validator_node: max retries (3) exhausted" in result["error"]
        assert fake.queue == []

    # ── classify loop ──

    BAD_QUOTE = (
        '{"doc_type": "M&A", "evidence_quote": "agreed to buy Widgets outright",'
        ' "confidence": 0.9, "reasoning": "deal"}'
    )
    BAD_LABEL = (
        '{"doc_type": "Management_Change", "evidence_quote": "",'
        ' "confidence": 0.9, "reasoning": "x"}'
    )
    NONE = '{"doc_type": "none", "confidence": 0.8, "reasoning": "earnings"}'
    UNCLEAR = '{"doc_type": "unclear", "confidence": 0.4, "reasoning": "mixed"}'

    async def test_bad_quote_is_retried_with_feedback_then_recovers(
        self, scripted
    ) -> None:
        fake, path = scripted
        fake.queue = [self.BAD_QUOTE, self.CLASSIFY, self.GOOD]

        result = await run_pipeline(path)

        assert result["validation_passed"] is True
        assert result["classify_retry_count"] == 1
        assert result["retry_count"] == 0  # extraction budget untouched
        [log] = result["correction_logs"]
        assert log["loop"] == "classify"
        assert "[REFERENCE]" in log["error_summary"]
        # the second classifier prompt carries the feedback
        second_classifier_prompt = fake.calls[1][-1]["content"]
        assert "PREVIOUS CLASSIFICATION FEEDBACK" in second_classifier_prompt
        assert fake.queue == []

    async def test_unparsable_label_is_retried_then_recovers(self, scripted) -> None:
        fake, path = scripted
        fake.queue = [self.BAD_LABEL, self.CLASSIFY, self.GOOD]

        result = await run_pipeline(path)

        assert result["validation_passed"] is True
        assert result["classify_retry_count"] == 1
        assert "[SCHEMA]" in result["correction_logs"][0]["error_summary"]

    async def test_classification_that_never_validates_stops_before_extraction(
        self, scripted
    ) -> None:
        fake, path = scripted
        fake.queue = [self.BAD_QUOTE] * 3 + [self.GOOD]  # 1 attempt + 2 retries

        result = await run_pipeline(path)

        assert result["doc_type"] is None
        assert "classification_validator: max retries (2) exhausted" in result["error"]
        failure = result["failure"]
        assert (failure["stage"], failure["code"], failure["category"]) == (
            "classify_validate",
            "EVIDENCE_QUOTE_NOT_FOUND",
            "reference",
        )
        assert len(fake.calls) == 3  # three classifier calls, no extraction call
        assert fake.queue == [self.GOOD]
        assert result["extracted_data"] is None

    async def test_unparsable_classifier_output_that_never_recovers(
        self, scripted
    ) -> None:
        fake, path = scripted
        fake.queue = [self.BAD_LABEL] * 3 + [self.GOOD]

        result = await run_pipeline(path)

        assert result["failure"]["stage"] == "classify"
        assert result["failure"]["code"] == "CLASSIFICATION_PARSE_FAILED"
        assert fake.queue == [self.GOOD]

    async def test_none_label_is_out_of_scope_and_skips_extraction(
        self, scripted
    ) -> None:
        fake, path = scripted
        fake.queue = [self.NONE, self.GOOD]

        result = await run_pipeline(path)

        assert result["doc_type"] == "none"
        assert result["failure"]["code"] == "UNSUPPORTED_DOC_TYPE"
        assert result["failure"]["category"] == "out_of_scope"
        assert result["retry_count"] == 0 and result["classify_retry_count"] == 0
        assert fake.queue == [self.GOOD]  # the extractor never ran

    async def test_unclear_label_needs_a_human_and_skips_extraction(
        self, scripted
    ) -> None:
        fake, path = scripted
        fake.queue = [self.UNCLEAR, self.GOOD]

        result = await run_pipeline(path)

        assert result["failure"]["code"] == "CLASSIFICATION_UNCLEAR"
        assert result["failure"]["category"] == "ambiguous"
        assert fake.queue == [self.GOOD]

    async def test_the_two_budgets_are_independent(self, scripted) -> None:
        fake, path = scripted
        # classification needs both its retries, then extraction exhausts its own
        fake.queue = [self.BAD_QUOTE, self.BAD_QUOTE, self.CLASSIFY] + [
            self.BAD_ENUM
        ] * 4

        result = await run_pipeline(path)

        assert result["classify_retry_count"] == 2
        assert result["retry_count"] == 3
        by_loop = {
            loop: [l["cycle"] for l in result["correction_logs"] if l["loop"] == loop]
            for loop in ("classify", "extract")
        }
        assert by_loop == {"classify": [1, 2], "extract": [1, 2, 3, 4]}
        assert result["failure"]["stage"] == "extract"
        assert result["failure"]["code"] == "SCHEMA_PARSE_FAILED"

    async def test_validator_exhaustion_is_attributed_to_the_rule(
        self, scripted
    ) -> None:
        fake, path = scripted
        no_acquirer = '{"target": "W", "total_value_usd": 5, "payment_method": "Cash"}'
        fake.queue = [self.CLASSIFY] + [no_acquirer] * 4

        result = await run_pipeline(path)

        failure = result["failure"]
        assert (failure["stage"], failure["code"], failure["category"]) == (
            "validate",
            "ACQUIRER.NON_EMPTY",
            "missing_info",
        )

    def test_graph_compiles(self) -> None:
        assert build_pipeline() is not None


# ═══════════════════════════════════════════════════════════════════
#  run_pipeline (integration-light, files exist on disk)
# ═══════════════════════════════════════════════════════════════════


class TestRunPipeline:
    """High-level pipeline execution with real file I/O."""

    async def test_missing_file_raises(self) -> None:
        """run_pipeline with a non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            await run_pipeline("/tmp/nonexistent_file_12345.txt")

    @pytest.mark.integration
    async def test_pipeline_state_structure(self) -> None:
        """Running on a real file should return a well-structured state."""
        import os

        sample_path = "evals/datasets/golden/inputs/test_ma_pass.txt"
        if not os.path.exists(sample_path):
            pytest.skip(f"Sample file not found: {sample_path}")

        result = await run_pipeline(sample_path)

        # The state must be a dict-like PipelineState
        assert isinstance(result, dict)

        # Required keys must be present
        assert "file_path" in result
        assert "raw_content" in result
        assert "doc_type" in result
        assert "extracted_data" in result
        assert "validation_passed" in result
        assert "retry_count" in result

        # file_path should be the resolved absolute path
        assert "test_ma_pass.txt" in result.get("file_path", "")

        # raw_content should be non-empty
        assert len(result.get("raw_content", "")) > 0

    @pytest.mark.integration
    async def test_pipeline_routes_all_nodes(self) -> None:
        """The pipeline should execute all nodes (reader → classifier → extractor → validator)."""
        import os

        sample_path = "evals/datasets/golden/inputs/test_ma_pass.txt"
        if not os.path.exists(sample_path):
            pytest.skip("Sample file not found")

        result = await run_pipeline(sample_path)

        # doc_type should be classified
        assert result.get("doc_type") is not None
        assert result["doc_type"] in ("M&A", "Dividend", "none", "unclear")

        # extracted_data should be populated (even if extraction failed)
        assert (
            result.get("extracted_data") is not None or result.get("error") is not None
        )

        # validation should have run
        assert (
            result.get("validation_passed") is not None
            or result.get("error") is not None
        )

    @pytest.mark.integration
    async def test_pipeline_idempotent_state_keys(self) -> None:
        """Running twice on the same file should produce the same key structure."""
        import os

        sample_path = "evals/datasets/golden/inputs/test_ma_pass.txt"
        if not os.path.exists(sample_path):
            pytest.skip("Sample file not found")

        result1 = await run_pipeline(sample_path)
        result2 = await run_pipeline(sample_path)

        # Both runs should have the same set of top-level keys
        keys1 = set(result1.keys())
        keys2 = set(result2.keys())
        # Core keys that must always be present
        for key in (
            "file_path",
            "raw_content",
            "doc_type",
            "extracted_data",
            "validation_passed",
            "retry_count",
        ):
            assert key in keys1
            assert key in keys2


# ═══════════════════════════════════════════════════════════════════
#  Edge case: file with no extractable data
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases for pipeline routing."""

    @pytest.mark.integration
    async def test_fail_loop_file_hits_dlq(self) -> None:
        """test_ma_fail_loop.txt should exhaust retries and hit DLQ path."""
        import os

        fail_path = "evals/datasets/golden/inputs/test_ma_fail_loop.txt"
        if not os.path.exists(fail_path):
            pytest.skip("Fail-loop sample file not found")

        result = await run_pipeline(fail_path)

        # The document is an analyst note (speculative, no concrete deal)
        # so it should either fail validation or have an error
        assert (
            result.get("validation_passed") is not True
            or result.get("error") is not None
        )
