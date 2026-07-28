"""OpenRouter-format structured reasoning replay and stream conversion."""

import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from free_claude_code.core.anthropic import (
    is_synthetic_openai_tool_turn_boundary,
)
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.streaming import AnthropicStreamLedger
from free_claude_code.core.reasoning import ReasoningPolicy


def apply_reasoning_details_replay(
    body: dict[str, Any], request: MessagesRequest, _policy: ReasoningPolicy
) -> None:
    """Replay opaque reasoning details on their converted assistant messages."""
    assistant_details = _assistant_reasoning_details(request.messages)
    if not assistant_details:
        return
    messages = body.get("messages")
    if not isinstance(messages, list):
        return

    cursor = 0
    for details in assistant_details:
        for index in range(cursor, len(messages)):
            message = messages[index]
            if (
                not isinstance(message, dict)
                or message.get("role") != "assistant"
                or is_synthetic_openai_tool_turn_boundary(message)
            ):
                continue
            existing = message.get("reasoning_details")
            if isinstance(existing, list):
                existing.extend(details)
            else:
                message["reasoning_details"] = list(details)
            cursor = index + 1
            break


def iter_reasoning_detail_events(
    delta: Any,
    ledger: AnthropicStreamLedger,
    *,
    native_reasoning: str | None,
) -> Iterator[str]:
    """Convert structured reasoning details without duplicating native text."""
    details = _field(delta, "reasoning_details")
    if details is None:
        extra = _field(delta, "model_extra")
        if isinstance(extra, Mapping):
            details = extra.get("reasoning_details")
    if not _is_sequence(details):
        return

    for detail in details:
        preserved = _preserved_reasoning_detail(detail)
        if preserved:
            yield from ledger.close_content_blocks()
            index = ledger.blocks.allocate_index()
            yield ledger.content_block_start(index, "redacted_thinking", data=preserved)
            yield ledger.content_block_stop(index)
            continue
        if native_reasoning:
            continue
        text = _reasoning_detail_text(detail)
        if not text:
            continue
        yield from ledger.ensure_thinking_block()
        yield ledger.emit_thinking_delta(text)


def _assistant_reasoning_details(messages: Any) -> list[list[dict[str, Any]]]:
    if not _is_sequence(messages):
        return []
    result: list[list[dict[str, Any]]] = []
    for message in messages:
        if _field(message, "role") != "assistant":
            continue
        details = _redacted_reasoning_details(_field(message, "content"))
        if details:
            result.append(details)
    return result


def _redacted_reasoning_details(content: Any) -> list[dict[str, Any]]:
    if not _is_sequence(content):
        return []
    details: list[dict[str, Any]] = []
    for block in content:
        if _field(block, "type") != "redacted_thinking":
            continue
        data = _field(block, "data")
        if not isinstance(data, str) or not data:
            continue
        parsed = _json_payload(data)
        if isinstance(parsed, list):
            details.extend(item for item in parsed if isinstance(item, dict))
        elif isinstance(parsed, dict):
            details.append(parsed)
        else:
            details.append({"type": "reasoning.encrypted", "data": data})
    return details


def _reasoning_detail_text(detail: Any) -> str | None:
    kind = str(_field(detail, "type") or "").lower()
    if "encrypted" in kind or "redacted" in kind:
        return None
    for key in ("text", "content", "reasoning"):
        value = _field(detail, key)
        if isinstance(value, str) and value:
            return value
    return None


def _preserved_reasoning_detail(detail: Any) -> str | None:
    if not isinstance(detail, Mapping):
        return None
    kind = str(_field(detail, "type") or "").lower()
    if (
        "encrypted" in kind
        or "redacted" in kind
        or "summary" in kind
        or _reasoning_detail_text(detail) is None
    ):
        return json.dumps(dict(detail), separators=(",", ":"))
    return None


def _json_payload(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _field(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )
