"""Exact token counting via the Anthropic HTTP API."""

from typing import Any

import httpx
from loguru import logger

from free_claude_code.application.execution import AnthropicTokenCountUnavailable
from free_claude_code.core.anthropic.models import Message, SystemContent, Tool
from free_claude_code.core.diagnostics import (
    exception_cause_types,
    safe_exception_message,
)

__all__ = ["AnthropicTokenCountUnavailable", "count_tokens_via_anthropic_api"]

_COUNT_TOKENS_URL = "https://api.anthropic.com/v1/messages/count_tokens"
_ANTHROPIC_VERSION = "2023-06-01"


def count_tokens_via_anthropic_api(
    *,
    api_key: str,
    model: str,
    messages: list[Message],
    system: str | list[SystemContent] | None,
    tools: list[Tool] | None,
    timeout: float,
    proxy: str = "",
) -> int:
    """Return an exact input-token count from Anthropic."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            message.model_dump(mode="json", exclude_none=True) for message in messages
        ],
    }
    if system is not None:
        payload["system"] = (
            [block.model_dump(mode="json", exclude_none=True) for block in system]
            if isinstance(system, list)
            else system
        )
    if tools:
        payload["tools"] = [
            tool.model_dump(mode="json", exclude_none=True) for tool in tools
        ]

    try:
        with httpx.Client(proxy=proxy.strip() or None, timeout=timeout) as client:
            response = client.post(
                _COUNT_TOKENS_URL,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": _ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
        input_tokens = result["input_tokens"]
        if not isinstance(input_tokens, int) or isinstance(input_tokens, bool):
            raise TypeError("Anthropic input_tokens must be an integer")
        return input_tokens
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        logger.debug(
            "Anthropic count_tokens API unavailable, causes={}",
            exception_cause_types(exc),
        )
        raise AnthropicTokenCountUnavailable(safe_exception_message(exc)) from exc
