"""Native Anthropic Messages request builder for Z.ai."""

from __future__ import annotations

from typing import Any

from loguru import logger

from config.constants import ZAI_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.native_messages_request import (
    build_base_native_anthropic_request_body,
)
from providers.exceptions import InvalidRequestError


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build JSON for Z.ai Anthropic-compat ``POST …/messages``."""
    logger.debug(
        "ZAI_REQUEST: native build model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    body = build_base_native_anthropic_request_body(
        request_data,
        default_max_tokens=ZAI_DEFAULT_MAX_OUTPUT_TOKENS,
        thinking_enabled=thinking_enabled,
    )
    extra = getattr(request_data, "extra_body", None)
    if extra:
        raise InvalidRequestError(
            "Z.ai native Messages API does not support extra_body on requests."
        )
    body["stream"] = True

    logger.debug(
        "ZAI_REQUEST: build done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
