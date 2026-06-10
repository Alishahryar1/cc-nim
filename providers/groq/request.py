"""Request builder for Groq (OpenAI-compatible chat completions).

See Groq docs: https://console.groq.com/docs/openai — ``messages[].name`` and
unsupported token fields yield 400; ``max_completion_tokens`` is preferred over
deprecated ``max_tokens``.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.exceptions import InvalidRequestError

_GROQ_UNSUPPORTED_TOP_KEYS = frozenset({"logprobs", "logit_bias", "top_logprobs"})


def _strip_message_names(messages: Any) -> None:
    """Remove ``name`` from each chat message (Groq rejects ``messages[].name``)."""
    if not isinstance(messages, list):
        return
    for msg in messages:
        if isinstance(msg, dict):
            msg.pop("name", None)


def _strip_unsupported_body_keys(body: dict[str, Any]) -> None:
    for key in _GROQ_UNSUPPORTED_TOP_KEYS:
        body.pop(key, None)


def _normalize_max_completion_tokens(body: dict[str, Any]) -> None:
    if "max_completion_tokens" in body:
        body.pop("max_tokens", None)
        return
    if "max_tokens" in body and body["max_tokens"] is not None:
        body["max_completion_tokens"] = body.pop("max_tokens")


def _cap_max_completion_tokens(body: dict[str, Any], max_tokens: int) -> None:
    """Cap max_completion_tokens to the configured limit (Groq API constraint)."""
    if "max_completion_tokens" in body:
        if body["max_completion_tokens"] is not None:
            body["max_completion_tokens"] = min(
                body["max_completion_tokens"], max_tokens
            )


def _normalize_n_candidates(body: dict[str, Any]) -> None:
    """Groq only supports ``n`` = 1; coerce if present."""
    if body.get("n") is None:
        return
    body["n"] = 1


def build_request_body(
    request_data: Any, *, thinking_enabled: bool, max_tokens: int = 32768
) -> dict:
    """Build OpenAI-format request body from an Anthropic request for Groq.
    
    Args:
        request_data: The incoming Anthropic request data
        thinking_enabled: Whether to enable thinking mode
        max_tokens: Maximum tokens cap for Groq (default 32768)
    """
    logger.debug(
        "GROQ_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    try:
        body = build_base_request_body(
            request_data,
            reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
            if thinking_enabled
            else ReasoningReplayMode.DISABLED,
        )
    except OpenAIConversionError as exc:
        raise InvalidRequestError(str(exc)) from exc

    request_extra = getattr(request_data, "extra_body", None)
    if isinstance(request_extra, dict) and request_extra:
        merged = dict(request_extra)
        body["extra_body"] = merged

    _strip_message_names(body.get("messages"))
    _strip_unsupported_body_keys(body)
    _normalize_max_completion_tokens(body)
    _cap_max_completion_tokens(body, max_tokens)
    _normalize_n_candidates(body)

    logger.debug(
        "GROQ_REQUEST: conversion done model={} msgs={} tools={} max_tokens={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
        body.get("max_completion_tokens"),
    )
    return body
