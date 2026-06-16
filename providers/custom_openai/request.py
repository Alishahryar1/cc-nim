"""
Request body builder for the Custom OpenAI-compatible provider.

Converts free-claude-code's internal Anthropic-format request objects into
OpenAI /v1/chat/completions format so they can be sent to any OpenAI-compatible
upstream (e.g. freellmapi). The core conversion work is delegated to
``core.anthropic.build_base_request_body``, with additional handling for:
  - Reasoning/thinking replay (persisting thinking blocks from prior responses).
  - Extra body passthrough (allows downstream caller to inject custom fields).
  - Max-token field normalization (OpenAI uses ``max_completion_tokens``).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.exceptions import InvalidRequestError


def _normalize_max_completion_tokens(body: dict[str, Any]) -> None:
    """Normalise max-token fields from Anthropic to OpenAI convention.

    OpenAI /v1/chat/completions uses ``max_completion_tokens`` (not ``max_tokens``).
    This ensures the upstream receives the correct field name regardless of what
    the calling code provides.
    """
    if "max_completion_tokens" in body:
        body.pop("max_tokens", None)
        return
    if "max_tokens" in body and body["max_tokens"] is not None:
        body["max_completion_tokens"] = body.pop("max_tokens")


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build OpenAI-format request body from an Anthropic request for Custom OpenAI.

    This is the primary conversion entrypoint. It:
      1. Delegates to ``build_base_request_body`` for the standard Anthropic→OpenAI
         message/tool/system conversion.
      2. When thinking is enabled, keeps ``reasoning_content`` blocks so upstream
         services that support extended thinking can leverage them.
      3. Passes through any ``extra_body`` from the original request (e.g. service-
         specific parameters like temperature presets).
      4. Normalises max-token naming to ``max_completion_tokens``.
    """
    logger.debug(
        "CUSTOM_OPENAI_REQUEST: conversion start model={} msgs={}",
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
        body["extra_body"] = dict(request_extra)

    _normalize_max_completion_tokens(body)

    logger.debug(
        "CUSTOM_OPENAI_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
