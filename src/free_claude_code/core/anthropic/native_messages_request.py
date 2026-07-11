"""Native Anthropic Messages request body construction (JSON-ready dicts)."""

from typing import Any

from .models import MessagesRequest

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


def _dump_request_fields(request_data: MessagesRequest) -> dict[str, Any]:
    """Extract the public Anthropic request fields."""
    raw = request_data.model_dump(exclude_none=True)
    return {
        field: raw[field]
        for field in _REQUEST_FIELDS
        if field in raw and raw[field] is not None
    }


def dump_raw_messages_request(request_data: MessagesRequest) -> dict[str, Any]:
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


def build_base_native_anthropic_request_body(
    request: MessagesRequest,
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

    return body
