"""Native Anthropic Messages request builder for OpenRouter."""

from __future__ import annotations

from typing import Any

from loguru import logger

from config.constants import (
    ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS as OPENROUTER_DEFAULT_MAX_TOKENS,
)
from core.anthropic.native_messages_request import (
    OpenRouterExtraBodyError,
    build_openrouter_native_request_body,
)
from providers.exceptions import InvalidRequestError


def build_request_body(
    request_data: Any,
    *,
    thinking_enabled: bool,
    max_tokens_cap: int | None = None,
) -> dict:
    """Build an Anthropic-format request body for OpenRouter's messages API."""
    logger.debug(
        "OPENROUTER_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    try:
        body = build_openrouter_native_request_body(
            request_data,
            thinking_enabled=thinking_enabled,
            default_max_tokens=OPENROUTER_DEFAULT_MAX_TOKENS,
        )
    except OpenRouterExtraBodyError as exc:
        raise InvalidRequestError(str(exc)) from exc

    # OpenRouter-specific max_tokens: cap against OPENROUTER_MAX_TOKENS
    if max_tokens_cap is not None and isinstance(body.get("max_tokens"), int):
        body["max_tokens"] = min(body["max_tokens"], max_tokens_cap)

    logger.debug(
        "OPENROUTER_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
