"""Request builder for Hugging Face Inference Providers (OpenAI-compatible).

See https://huggingface.co/docs/inference-providers — the router exposes the
standard OpenAI chat-completions surface. Model ids are Hub repo ids (e.g.
``meta-llama/Llama-3.3-70B-Instruct``) with an optional ``:provider`` suffix
to pin a specific inference provider.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.exceptions import InvalidRequestError

# Router 400 shape: "`max_tokens` must be less than or equal to `32768`, ..."
# The limit varies per model (bounded by its context window), so it is parsed
# from the error instead of hardcoded.
_MAX_TOKENS_LIMIT_RE = re.compile(
    r"max_tokens.*?less than or equal to\D*(\d+)", re.IGNORECASE | re.DOTALL
)


def clone_body_with_clamped_max_tokens(error_text: str, body: dict) -> dict | None:
    """Return a retry body with ``max_tokens`` clamped to the router's limit, or None."""
    match = _MAX_TOKENS_LIMIT_RE.search(error_text)
    if match is None:
        return None
    limit = int(match.group(1))
    current = body.get("max_tokens")
    if not isinstance(current, int) or current <= limit:
        return None
    retry_body = dict(body)
    retry_body["max_tokens"] = limit
    return retry_body


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build OpenAI-format request body from an Anthropic request for Hugging Face."""
    logger.debug(
        "HUGGINGFACE_REQUEST: conversion start model={} msgs={}",
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

    logger.debug(
        "HUGGINGFACE_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
