"""Reasoning and thinking conversion helpers for OpenAI Responses."""

from collections.abc import Mapping
from typing import Any


def reasoning_text_from_item(item: Mapping[str, Any]) -> str | None:
    content_parts = _text_parts_from_items(
        item.get("content"), item_type="reasoning_text"
    )
    if content_parts:
        return "\n".join(content_parts)
    summary_parts = _text_parts_from_items(
        item.get("summary"), item_type="summary_text"
    )
    if summary_parts:
        return "\n".join(summary_parts)
    return None


def encrypted_reasoning_from_item(item: Mapping[str, Any]) -> str | None:
    """Return opaque reasoning content without interpreting it."""

    return _optional_str(item.get("encrypted_content"))


def combine_reasoning(existing: str | None, addition: str | None) -> str | None:
    if addition is None:
        return existing
    if existing is None:
        return addition
    if existing == "":
        return addition
    if addition == "":
        return existing
    return f"{existing}\n{addition}"


def responses_reasoning_to_output_config(value: Any) -> dict[str, Any] | None:
    """Preserve the client's named effort for application-level resolution."""
    if not isinstance(value, Mapping):
        return None
    effort = value.get("effort")
    if isinstance(effort, str) and effort.strip():
        return {"effort": effort.strip().lower()}
    return None


def _text_parts_from_items(value: Any, *, item_type: str) -> list[str]:
    if not isinstance(value, list):
        return []
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict) and item.get("type") == item_type:
            text = _optional_str(item.get("text"))
            if text is not None:
                parts.append(text)
    return parts


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None
