"""A final LLM exception says who must act: config, input or transient."""

from __future__ import annotations

import httpx
import openai
import pytest

from src.llm.errors import classify_llm_exception


def _status_error(cls: type, status: int) -> Exception:
    request = httpx.Request("POST", "https://api.example.test/chat/completions")
    response = httpx.Response(status, request=request)
    return cls("boom", response=response, body=None)


@pytest.mark.parametrize(
    "cls,status",
    [
        (openai.AuthenticationError, 401),
        (openai.PermissionDeniedError, 403),
        (openai.NotFoundError, 404),
    ],
)
def test_credentials_permissions_or_model_are_config_errors(cls, status) -> None:
    assert classify_llm_exception(_status_error(cls, status)) == (
        "LLM_CONFIG_ERROR",
        "config_error",
    )


def test_a_rejected_request_is_an_input_error() -> None:
    exc = _status_error(openai.BadRequestError, 400)
    assert classify_llm_exception(exc) == ("LLM_REQUEST_REJECTED", "input_error")


@pytest.mark.parametrize(
    "exc",
    [
        openai.APIConnectionError(request=httpx.Request("POST", "https://x.test")),
        _status_error(openai.RateLimitError, 429),
        RuntimeError("anything else"),
    ],
)
def test_everything_else_is_transient(exc) -> None:
    assert classify_llm_exception(exc) == ("LLM_CALL_FAILED", "transient")


def test_client_passes_explicit_retry_and_timeout(monkeypatch) -> None:
    seen = {}

    class Capture:
        def __init__(self, **kwargs) -> None:
            seen.update(kwargs)

    monkeypatch.setattr("src.llm.client.AsyncOpenAI", Capture)
    from src.config import settings
    from src.llm.client import LLMClient

    LLMClient()
    assert seen["max_retries"] == settings.llm_max_retries == 2
    assert seen["timeout"] == settings.llm_timeout_seconds == 60.0
