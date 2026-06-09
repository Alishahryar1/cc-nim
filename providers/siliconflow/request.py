"""Request builder for SiliconFlow (OpenAI-compatible chat completions).

SiliconFlow API: https://docs.siliconflow.com/en/api-reference/chat-completions/chat-completions
- Standard OpenAI chat completions endpoint at ``/v1/chat/completions``.
- Supports ``enable_thinking`` for thinking-capable models (e.g. Qwen, DeepSeek-R1).
- Uses ``max_tokens`` (not ``max_completion_tokens``).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.exceptions import InvalidRequestError


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build OpenAI-format request body from an Anthropic request for SiliconFlow."""
    logger.debug(
        "SILICONFLOW_REQUEST: conversion start model={} msgs={}",
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

    extra: dict[str, Any] = {}
    request_extra = getattr(request_data, "extra_body", None)
    if isinstance(request_extra, dict) and request_extra:
        extra.update(request_extra)

    if thinking_enabled:
        extra["enable_thinking"] = True

    if extra:
        body["extra_body"] = extra

    logger.debug(
        "SILICONFLOW_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
