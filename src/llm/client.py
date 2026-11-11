"""
LLM client — OpenAI-compatible wrapper for DeepSeek / Ollama / any OpenAI proxy.

DeepSeek does NOT support the ``response_format`` parameter, so we use
regular text generation + manual JSON parsing with Pydantic validation.

Every call records ``last_call`` (raw reply text, token usage, latency, and a
hash of the exact system message) *before* the reply is parsed, so the audit
trail keeps what the model said even when its output fails validation.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Optional

from openai import AsyncOpenAI
from typing_extensions import TypedDict

from src.config import settings


class LLMCallInfo(TypedDict):
    """What one chat-completion call cost and returned (before parsing)."""

    model_name: str
    prompt_version: str
    """``sha256:<12 hex>`` of the full system message actually sent."""
    raw_response: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    """Hidden reasoning tokens inside ``completion_tokens`` (thinking models)."""
    total_tokens: Optional[int]
    latency_ms: int


def prompt_fingerprint(system_message: str) -> str:
    """Short, stable id of a system message: changes whenever the prompt or schema does."""
    return "sha256:" + hashlib.sha256(system_message.encode("utf-8")).hexdigest()[:12]


class LLMClient:
    """Thin async wrapper around DeepSeek (OpenAI-compatible) API."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        self._model = settings.deepseek_model
        self.last_call: Optional[LLMCallInfo] = None
        """Info about the most recent call; ``None`` if it never got a reply."""

    async def _chat(
        self,
        system_message: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        thinking: str = "",
    ) -> str:
        """
        Send one chat completion, record ``last_call`` and return the raw text.

        ``thinking`` ("enabled" / "disabled") is sent as DeepSeek's
        ``{"thinking": {"type": ...}}`` request field; "" sends nothing, so
        other OpenAI-compatible providers are unaffected.
        """
        self.last_call = None
        started = time.perf_counter()
        extra: dict[str, Any] = {}
        if thinking:
            extra["extra_body"] = {"thinking": {"type": thinking}}
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
        raw_text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        self.last_call = LLMCallInfo(
            model_name=self._model,
            prompt_version=prompt_fingerprint(system_message),
            raw_response=raw_text,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            reasoning_tokens=getattr(details, "reasoning_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        return raw_text

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        """
        Send a chat completion request and return the text content.

        Args:
            system_prompt: System-level instruction for the LLM.
            user_prompt: The user message / input data.
            temperature: Sampling temperature (low = deterministic).
            max_tokens: Maximum tokens in the response.

        Returns:
            The LLM's response text content.
        """
        return await self._chat(
            system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens
        )

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type,
        *,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        thinking: str = "",
    ) -> object:
        """
        Request structured JSON output and parse into a Pydantic model.

        Works with any OpenAI-compatible API, including DeepSeek which
        does NOT support the ``response_format`` parameter.

        Strategy:
        1. Request plain-text completion with strong JSON instructions.
        2. Extract the JSON block from the response (handles markdown fences).
        3. Parse & validate with Pydantic ``model_validate_json()``.

        Args:
            system_prompt: System-level instruction.
            user_prompt: The user message.
            response_model: A Pydantic BaseModel subclass to parse into.
            temperature: Sampling temperature (ignored by thinking models).
            max_tokens: Maximum tokens in the response, hidden reasoning included.
            thinking: "enabled" / "disabled" / "" (see ``_chat``).

        Returns:
            An instance of *response_model*.
        """
        # Augment system prompt with JSON schema hint
        schema_hint = _build_schema_hint(response_model)

        system_message = (
            f"{system_prompt}\n\n"
            "You MUST respond with valid JSON only, no markdown fences, "
            "no code blocks, no explanations. The JSON must conform to "
            f"this schema:\n\n{schema_hint}"
        )
        raw_text = await self._chat(
            system_message,
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
        )

        # Clean the response — strip markdown JSON fences if present
        cleaned = _extract_json(raw_text)

        # Parse via Pydantic
        return response_model.model_validate_json(cleaned)

    async def close(self) -> None:
        """Release the underlying HTTP session."""
        await self._client.close()


# ── Helpers ──


def _build_schema_hint(model: type) -> str:
    """
    Produce a minimal JSON schema snippet from a Pydantic model.

    Uses ``model_json_schema()`` which is the recommended Pydantic v2 API.
    """
    schema = model.model_json_schema()
    # Only keep the properties section for brevity
    properties = schema.get("properties", {})
    return json.dumps(properties, indent=2)


def _extract_json(text: str) -> str:
    """
    Extract a JSON object from a string that may contain markdown fences
    or surrounding explanation text.

    Handles:
      - ```json ... ```
      - ``` ... ```
      - Plain JSON at the start/end of text
    """
    # Try to find a JSON code block first
    patterns = [
        r"```(?:json)?\s*\n?([\s\S]*?)```",  # ```json ... ```
        r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}",  # bare {...} balanced
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidate = match.group(1) if match.lastindex else match.group(0)
            candidate = candidate.strip()
            try:
                json.loads(candidate)  # validate it's parseable
                return candidate
            except json.JSONDecodeError:
                continue

    # Last resort: return the whole thing and let Pydantic raise
    return text.strip()
