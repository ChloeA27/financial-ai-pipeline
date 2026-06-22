"""
LLM client — OpenAI-compatible wrapper for DeepSeek / Ollama / any OpenAI proxy.

DeepSeek does NOT support the ``response_format`` parameter, so we use
regular text generation + manual JSON parsing with Pydantic validation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from openai import AsyncOpenAI

from src.config import settings


class LLMClient:
    """Thin async wrapper around DeepSeek (OpenAI-compatible) API."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        self._model = settings.deepseek_model

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
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type,
        *,
        temperature: float = 0.1,
        max_tokens: int = 4096,
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
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            An instance of *response_model*.
        """
        # Augment system prompt with JSON schema hint
        schema_hint = _build_schema_hint(response_model)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\n\n"
                        "You MUST respond with valid JSON only, no markdown fences, "
                        "no code blocks, no explanations. The JSON must conform to "
                        f"this schema:\n\n{schema_hint}"
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        raw_text = response.choices[0].message.content or ""

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
