"""Exact token counting via the real Anthropic API.

Owns Anthropic SDK failure classification per the failure-ownership
convention: callers only ever see ``AnthropicTokenCountUnavailable`` or an
``int`` result, never a raw SDK/HTTP exception.
"""

import anthropic
from loguru import logger

from free_claude_code.core.anthropic.models import Message, SystemContent, Tool
from free_claude_code.core.diagnostics import (
    exception_cause_types,
    safe_exception_message,
)


class AnthropicTokenCountUnavailable(Exception):
    """Raised when the real Anthropic count_tokens API could not be reached."""


def count_tokens_via_anthropic_api(
    *,
    api_key: str,
    model: str,
    messages: list[Message],
    system: str | list[SystemContent] | None,
    tools: list[Tool] | None,
    timeout: float,
) -> int:
    """Return an exact input token count from the real Anthropic API."""
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    messages_payload = [m.model_dump(mode="json", exclude_none=True) for m in messages]
    system_payload = (
        [s.model_dump(mode="json", exclude_none=True) for s in system]
        if isinstance(system, list)
        else system
    )
    tools_payload = (
        [t.model_dump(mode="json", exclude_none=True) for t in tools] if tools else None
    )

    try:
        result = client.messages.count_tokens(
            model=model,
            messages=messages_payload,
            system=system_payload if system_payload is not None else anthropic.omit,
            tools=tools_payload if tools_payload is not None else anthropic.omit,
        )
    except anthropic.APIError as exc:
        logger.debug(
            "Anthropic count_tokens API unavailable, causes={}",
            exception_cause_types(exc),
        )
        raise AnthropicTokenCountUnavailable(safe_exception_message(exc)) from exc
    return result.input_tokens
