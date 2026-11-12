"""LLMClient records what each call returned and cost, before parsing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import pytest
from pydantic import BaseModel, ValidationError

from src.llm.client import LLMClient, prompt_fingerprint


class _Answer(BaseModel):
    value: int


class UsageFakeOpenAI:
    """Fake ``AsyncOpenAI`` whose responses carry token usage."""

    replies: list[str] = []
    usage: Optional[SimpleNamespace] = None
    fail: bool = False
    requests: list[dict[str, Any]] = []

    def __init__(self, **_: Any) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        UsageFakeOpenAI.requests.append(kwargs)
        if UsageFakeOpenAI.fail:
            raise ConnectionError("network down")
        content = UsageFakeOpenAI.replies.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=UsageFakeOpenAI.usage,
        )

    async def close(self) -> None:
        return None


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> type[UsageFakeOpenAI]:
    UsageFakeOpenAI.replies = []
    UsageFakeOpenAI.usage = SimpleNamespace(
        prompt_tokens=120, completion_tokens=30, total_tokens=150
    )
    UsageFakeOpenAI.fail = False
    UsageFakeOpenAI.requests = []
    monkeypatch.setattr("src.llm.client.AsyncOpenAI", UsageFakeOpenAI)
    return UsageFakeOpenAI


async def test_usage_raw_text_and_latency_are_recorded(fake) -> None:
    fake.replies = ['{"value": 7}']
    client = LLMClient()
    result = await client.generate_structured(
        system_prompt="sys", user_prompt="u", response_model=_Answer
    )

    assert result.value == 7
    call = client.last_call
    assert call["raw_response"] == '{"value": 7}'
    assert (call["prompt_tokens"], call["completion_tokens"], call["total_tokens"]) == (
        120,
        30,
        150,
    )
    assert call["latency_ms"] >= 0
    assert call["prompt_version"].startswith("sha256:")


async def test_raw_reply_survives_a_parse_failure(fake) -> None:
    fake.replies = ["I think the value is seven"]
    client = LLMClient()
    with pytest.raises(ValidationError):
        await client.generate_structured(
            system_prompt="sys", user_prompt="u", response_model=_Answer
        )
    assert client.last_call["raw_response"] == "I think the value is seven"


async def test_transport_failure_leaves_no_stale_call_info(fake) -> None:
    fake.replies = ['{"value": 1}']
    client = LLMClient()
    await client.generate_structured(
        system_prompt="sys", user_prompt="u", response_model=_Answer
    )
    fake.fail = True
    with pytest.raises(ConnectionError):
        await client.generate_structured(
            system_prompt="sys", user_prompt="u", response_model=_Answer
        )
    assert client.last_call is None


async def test_missing_usage_is_tolerated(fake) -> None:
    fake.usage = None
    fake.replies = ['{"value": 1}']
    client = LLMClient()
    await client.generate_structured(
        system_prompt="sys", user_prompt="u", response_model=_Answer
    )
    assert client.last_call["total_tokens"] is None


async def test_prompt_version_tracks_prompt_and_schema(fake) -> None:
    class _Other(BaseModel):
        value: int
        extra: str

    versions = []
    for system_prompt, model in (("sys", _Answer), ("sys", _Answer), ("sys2", _Answer)):
        fake.replies = ['{"value": 1}']
        client = LLMClient()
        await client.generate_structured(
            system_prompt=system_prompt,
            user_prompt="different every time",
            response_model=model,
        )
        versions.append(client.last_call["prompt_version"])
    fake.replies = ['{"value": 1, "extra": "x"}']
    client = LLMClient()
    await client.generate_structured(
        system_prompt="sys", user_prompt="u", response_model=_Other
    )
    versions.append(client.last_call["prompt_version"])

    assert versions[0] == versions[1]  # same prompt + schema, user text irrelevant
    assert versions[0] != versions[2]  # prompt changed
    assert versions[0] != versions[3]  # schema changed


async def test_thinking_mode_is_sent_only_when_set(fake) -> None:
    fake.replies = ['{"value": 1}', '{"value": 2}']
    client = LLMClient()
    await client.generate_structured(
        system_prompt="sys", user_prompt="u", response_model=_Answer, thinking=""
    )
    await client.generate_structured(
        system_prompt="sys",
        user_prompt="u",
        response_model=_Answer,
        thinking="disabled",
    )
    assert "extra_body" not in fake.requests[0]  # other providers: nothing extra
    assert fake.requests[1]["extra_body"] == {"thinking": {"type": "disabled"}}


async def test_reasoning_tokens_are_recorded_when_reported(fake) -> None:
    fake.usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=1024,
        total_tokens=1124,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=990),
    )
    fake.replies = ['{"value": 1}']
    client = LLMClient()
    await client.generate_structured(
        system_prompt="sys", user_prompt="u", response_model=_Answer
    )
    assert client.last_call["reasoning_tokens"] == 990


def test_fingerprint_is_short_and_stable() -> None:
    assert prompt_fingerprint("abc") == prompt_fingerprint("abc")
    assert len(prompt_fingerprint("abc")) == len("sha256:") + 12
