"""Process-wide best-effort plain-text token estimation."""

from typing import Protocol

import tiktoken
from loguru import logger

_DISALLOWED_SPECIAL: tuple[str, ...] = ()


class _TokenEncoder(Protocol):
    def encode(
        self, text: str, *, disallowed_special: tuple[str, ...]
    ) -> list[int]: ...


def _load_encoder() -> _TokenEncoder | None:
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:
        logger.warning(
            "cl100k_base token encoder unavailable ({}); using approximate token estimates",
            type(exc).__name__,
        )
        return None


_ENCODER = _load_encoder()


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens for plain text using the shared process-wide encoder."""
    if not text:
        return 0
    if _ENCODER is not None:
        return len(_ENCODER.encode(text, disallowed_special=_DISALLOWED_SPECIAL))
    return max(1, len(text) // 4)
