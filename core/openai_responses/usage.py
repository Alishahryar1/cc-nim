"""Usage helpers for OpenAI Responses payloads."""

from __future__ import annotations

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODER = None

_DISALLOWED_SPECIAL: tuple[str, ...] = ()


def estimate_text_tokens(text: str) -> int:
    """Return a best-effort token estimate for Responses usage details."""
    if not text:
        return 0
    if _ENCODER is not None:
        return len(_ENCODER.encode(text, disallowed_special=_DISALLOWED_SPECIAL))
    return max(1, len(text) // 4)
