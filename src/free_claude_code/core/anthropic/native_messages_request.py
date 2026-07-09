"""Native Anthropic Messages request body construction (JSON-ready dicts)."""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

_REQUEST_FIELDS = (
    "model",
    "messages",
    "system",
    "max_tokens",
    "stop_sequences",
    "stream",
    "temperature",
    "top_p",
    "top_k",
    "metadata",
    "tools",
    "tool_choice",
    "thinking",
    "context_management",
    "output_config",
    "mcp_servers",
    "extra_body",
)


def _serialize_value(value: Any) -> Any:
    """Convert Pydantic models and lightweight objects into JSON-ready values."""
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {
            key: _serialize_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_serialize_value(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "__dict__"):
        return {
            key: _serialize_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_") and item is not None
        }
    return value


def _dump_request_fields(request_data: Any) -> dict[str, Any]:
    """Extract the public Anthropic request fields."""
    if isinstance(request_data, BaseModel):
        raw = request_data.model_dump(exclude_none=True)
        return {
            field: raw[field]
            for field in _REQUEST_FIELDS
            if field in raw and raw[field] is not None
        }

    dump = getattr(request_data, "model_dump", None)
    if callable(dump):
        raw = dump(exclude_none=True)
        if isinstance(raw, dict):
            return {
                field: raw[field]
                for field in _REQUEST_FIELDS
                if field in raw and raw[field] is not None
            }

    dumped: dict[str, Any] = {}
    for field in _REQUEST_FIELDS:
        value = getattr(request_data, field, None)
        if value is not None:
            dumped[field] = _serialize_value(value)
    return dumped


def dump_raw_messages_request(request_data: Any) -> dict[str, Any]:
    """Public JSON-ready dict of Anthropic public request fields (for native adapters)."""
    return _dump_request_fields(request_data)


def sanitize_native_messages_thinking_policy(
    messages: Any, *, thinking_enabled: bool
) -> Any:
    """Filter assistant message thinking blocks for upstream native Anthropic JSON.

    When ``thinking_enabled`` is false, remove ``thinking`` and ``redacted_thinking``
    history so disabled policy is not undermined by prior turns.

    When true, keep ``redacted_thinking`` and signed ``thinking``; remove only
    unsigned plain ``thinking`` blocks (not replayable).
    """
    if not isinstance(messages, list):
        return messages

    sanitized_messages: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized_messages.append(message)
            continue

        if message.get("role") != "assistant":
            sanitized_messages.append(message)
            continue

        content = message.get("content")
        if not isinstance(content, list):
            sanitized_messages.append(message)
            continue

        if not thinking_enabled:
            sanitized_content = [
                block
                for block in content
                if not (
                    isinstance(block, dict)
                    and block.get("type") in ("thinking", "redacted_thinking")
                )
            ]
        else:
            sanitized_content = [
                block
                for block in content
                if not (
                    isinstance(block, dict)
                    and block.get("type") == "thinking"
                    and not isinstance(block.get("signature"), str)
                )
            ]

        sanitized_message = dict(message)
        sanitized_message["content"] = sanitized_content or ""
        sanitized_messages.append(sanitized_message)

    return sanitized_messages


def _fold_blocks_into_tool_result(
    target: dict[str, Any], *, leading: list[Any], trailing: list[Any]
) -> None:
    """Fold non-tool_result blocks into a tool_result's content in place,
    preserving order: ``leading`` blocks go before the existing content,
    ``trailing`` after. Keeps a plain-string content as a joined string when
    only text is involved; otherwise normalizes to list form so non-text
    blocks (e.g. images) survive."""
    content = target.get("content")

    def is_text(block: Any) -> bool:
        return (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )

    if isinstance(content, str) and all(is_text(b) for b in (*leading, *trailing)):
        parts = [b["text"] for b in leading]
        if content:
            parts.append(content)
        parts.extend(b["text"] for b in trailing)
        target["content"] = "\n\n".join(part for part in parts if part)
        return

    normalized: list[Any] = list(leading)
    if isinstance(content, str):
        if content:
            normalized.append({"type": "text", "text": content})
    elif isinstance(content, list):
        normalized.extend(content)
    normalized.extend(trailing)
    target["content"] = normalized


def sanitize_tool_result_user_messages(messages: Any) -> Any:
    """Make user turns that carry a tool_result contain only tool_results.

    Some local chat templates (Mistral/devstral via llama.cpp) fail to render
    a user turn that mixes a ``tool_result`` block with any other block — e.g.
    when Claude Code appends a reminder ``text`` block to the same turn as a
    tool result. Real Anthropic accepts the mixed turn; these templates raise
    "roles must alternate ... except for tool calls and results". Fold the
    non-tool_result blocks into an adjacent tool_result so the turn is
    tool_result-only, which every template renders and which the model reads
    identically.

    Content block order is part of the prompt, so folding preserves it: blocks
    that appear *before* a tool_result are folded into that tool_result (ahead
    of its content); blocks after the last tool_result are folded into it
    (after its content). A leading ``text`` block therefore stays before the
    tool output rather than being moved behind it.
    """
    if not isinstance(messages, list):
        return messages

    def is_tool_result(block: Any) -> bool:
        return isinstance(block, dict) and block.get("type") == "tool_result"

    sanitized_messages: list[Any] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            sanitized_messages.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            sanitized_messages.append(message)
            continue
        if not any(is_tool_result(b) for b in content) or all(
            is_tool_result(b) for b in content
        ):
            # No tool_result to fold into, or already tool_result-only.
            sanitized_messages.append(message)
            continue

        new_results: list[Any] = []
        pending: list[Any] = []  # non-tool_result blocks seen since the last one
        for block in content:
            if is_tool_result(block):
                folded = dict(block)
                if pending:
                    _fold_blocks_into_tool_result(folded, leading=pending, trailing=[])
                    pending = []
                new_results.append(folded)
            else:
                pending.append(block)
        if pending:
            # Blocks after the last tool_result attach to it, after its content.
            _fold_blocks_into_tool_result(new_results[-1], leading=[], trailing=pending)

        sanitized_message = dict(message)
        sanitized_message["content"] = new_results
        sanitized_messages.append(sanitized_message)

    return sanitized_messages


def build_base_native_anthropic_request_body(
    request: Any,
    *,
    default_max_tokens: int,
    thinking_enabled: bool,
) -> dict[str, Any]:
    """Serialize a Pydantic messages request to a generic native Anthropic body."""
    body = dump_raw_messages_request(request)

    body.pop("extra_body", None)

    if "thinking" in body:
        thinking_cfg = body.pop("thinking")
        if thinking_enabled and isinstance(thinking_cfg, dict):
            thinking_payload: dict[str, Any] = {"type": "enabled"}
            budget_tokens = thinking_cfg.get("budget_tokens")
            if isinstance(budget_tokens, int):
                thinking_payload["budget_tokens"] = budget_tokens
            body["thinking"] = thinking_payload

    if "max_tokens" not in body:
        body["max_tokens"] = default_max_tokens

    if "messages" in body:
        body["messages"] = sanitize_native_messages_thinking_policy(
            body["messages"],
            thinking_enabled=thinking_enabled,
        )
        body["messages"] = sanitize_tool_result_user_messages(body["messages"])

    return body
