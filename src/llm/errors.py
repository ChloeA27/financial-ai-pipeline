"""
Which failure an exception from the LLM provider is.

The OpenAI SDK has already retried what can be retried (connection errors,
timeouts, 408/409/429/5xx; ``llm_max_retries`` with exponential backoff) before
an exception reaches a node.  What arrives here is final, so the only question is
who must act:

    401 / 403 / 404   config_error  our key, permissions or model name are wrong;
                                    every document will fail the same way
    400 / 422         input_error   the provider rejected this request (often:
                                    the input is too long for the model)
    anything else     transient     the provider was unreachable or overloaded
"""

from __future__ import annotations

import openai

from src.schemas.failure import (
    CAT_CONFIG_ERROR,
    CAT_INPUT_ERROR,
    CAT_TRANSIENT,
    LLM_CALL_FAILED,
    LLM_CONFIG_ERROR,
    LLM_REQUEST_REJECTED,
)

_CONFIG = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
)
_REJECTED = (openai.BadRequestError, openai.UnprocessableEntityError)


def classify_llm_exception(exc: BaseException) -> tuple[str, str]:
    """(code, category) of a final LLM-call exception."""
    if isinstance(exc, _CONFIG):
        return LLM_CONFIG_ERROR, CAT_CONFIG_ERROR
    if isinstance(exc, _REJECTED):
        return LLM_REQUEST_REJECTED, CAT_INPUT_ERROR
    return LLM_CALL_FAILED, CAT_TRANSIENT
