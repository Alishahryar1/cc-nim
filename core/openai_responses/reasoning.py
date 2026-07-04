"""Reasoning and thinking conversion helpers for OpenAI Responses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .tools import optional_str


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


def combine_reasoning(existing: str | None, addition: str | None) -> str | None:
    if not addition:
        return existing
    if not existing:
        return addition
    return f"{existing}\n{addition}"


def responses_reasoning_to_thinking(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    if value.get("effort") == "none":
        return {"type": "disabled", "enabled": False}
    if any(item is not None for item in value.values()):
        return {"type": "enabled", "enabled": True}
    return None


def _text_parts_from_items(value: Any, *, item_type: str) -> list[str]:
    if not isinstance(value, list):
        return []
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict) and item.get("type") == item_type:
            text = optional_str(item.get("text"))
            if text:
                parts.append(text)
    return parts
