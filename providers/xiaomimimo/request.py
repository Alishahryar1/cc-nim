"""Native Anthropic Messages request builder for Xiaomi MiMo."""

from __future__ import annotations

from typing import Any

from loguru import logger

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.native_messages_request import (
    OpenRouterExtraBodyError,
    build_base_native_anthropic_request_body,
    validate_openrouter_extra_body,
)
from providers.exceptions import InvalidRequestError


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build JSON for Xiaomi MiMo Anthropic-compat ``POST …/messages``."""
    logger.debug(
        "XIAOMIMIMO_REQUEST: native build model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    body = build_base_native_anthropic_request_body(
        request_data,
        default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        thinking_enabled=thinking_enabled,
    )

    extra = getattr(request_data, "extra_body", None)
    if isinstance(extra, dict) and extra:
        try:
            validate_openrouter_extra_body(extra)
        except OpenRouterExtraBodyError as exc:
            raise InvalidRequestError(str(exc)) from exc
        body.update(extra)

    body["stream"] = True

    logger.debug(
        "XIAOMIMIMO_REQUEST: build done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
