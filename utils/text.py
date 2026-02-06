"""Shared text extraction utilities."""

from typing import Any


def extract_text_from_content(content: Any) -> str:
    """Extract concatenated text from message content (str or list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            text = getattr(block, "text", "")
            if text and isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""
