"""Usage helpers for OpenAI Responses payloads."""

from typing import Protocol

_DISALLOWED_SPECIAL: tuple[str, ...] = ()


class _TokenEncoder(Protocol):
    def encode(
        self, text: str, *, disallowed_special: tuple[str, ...]
    ) -> list[int]: ...


def _load_encoder() -> _TokenEncoder | None:
    try:
        import tiktoken
    except ImportError:
        return None

    try:
        return tiktoken.get_encoding("cl100k_base")
    except ValueError:
        return None


_ENCODER = _load_encoder()


def estimate_text_tokens(text: str) -> int:
    """Return a best-effort token estimate for Responses usage details."""
    if not text:
        return 0
    if _ENCODER is not None:
        return len(_ENCODER.encode(text, disallowed_special=_DISALLOWED_SPECIAL))
    return max(1, len(text) // 4)
