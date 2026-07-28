"""Shared Anthropic request serialization helpers."""

import json
from typing import Any

from free_claude_code.config.settings import get_settings
from free_claude_code.core.token_saver import TokenSaver

from .models import MessagesRequest

_MESSAGES_REQUEST_FIELDS = (
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


def dump_messages_request(request: MessagesRequest) -> dict[str, Any]:
    """Return JSON-ready public Messages fields without FCC routing state."""
    raw = request.model_dump(exclude_none=True)
    return {
        field: raw[field]
        for field in _MESSAGES_REQUEST_FIELDS
        if field in raw and raw[field] is not None
    }


def serialize_tool_result_content(
    content: Any,
    *,
    tool_name: str = "",
    is_error: bool = False,
) -> str:
    """Serialize Anthropic ``tool_result.content`` into provider-safe text."""
    if content is None:
        return ""
    if isinstance(content, str):
        serialized = content
    elif isinstance(content, dict):
        serialized = json.dumps(content, ensure_ascii=False)
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict):
                parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        serialized = "\n".join(parts)
    else:
        serialized = str(content)

    settings = get_settings()
    if settings.token_saver_mode != "none":
        saver = TokenSaver.singleton(settings.token_saver_mode)
        serialized = saver.save_result(
            serialized, tool_name=tool_name, is_error=is_error
        )
    return serialized
