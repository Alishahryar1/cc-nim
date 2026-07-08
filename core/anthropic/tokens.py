"""Token estimation for Anthropic-compatible requests."""

from __future__ import annotations

import json
import unicodedata

import tiktoken
from loguru import logger

from .content import get_block_attr

_DISALLOWED_SPECIAL: tuple[str, ...] = ()


class _LazyEncoder:
    """Lazy initialisation of tiktoken encoder with graceful fallback."""

    __slots__ = ("_encoder", "_fallback_warned")

    def __init__(self) -> None:
        self._encoder: tiktoken.Encoding | None = None
        self._fallback_warned: bool = False

    def _init(self) -> tiktoken.Encoding | None:
        if self._encoder is not None:
            return self._encoder
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
            return self._encoder
        except Exception as exc:  # noqa: BLE001
            if not self._fallback_warned:
                logger.warning(
                    "tiktoken cl100k_base encoding unavailable ({}). "
                    "Using approximate whitespace-based fallback for token counting. "
                    "Install a stable network connection or pre-download tiktoken files "
                    "to improve accuracy.",
                    exc,
                )
                self._fallback_warned = True
        return None

    def encode(self, text: str, *, disallowed_special: tuple[str, ...]) -> list[int]:
        encoder = self._init()
        if encoder is not None:
            return encoder.encode(text, disallowed_special=disallowed_special)
        # Approximate fallback: token ≈ 1 per 4 chars, but count CJK/code points
        # as individual rough tokens for better approximation.
        tokens = _fallback_tokenize(text)
        return tokens


def _fallback_tokenize(text: str) -> list[int]:
    """Rough heuristic tokenisation when tiktoken is unavailable."""
    # Split on whitespace and count each chunk as roughly one token per 4 chars.
    # This is intentionally coarse—good enough for request-size heuristics.
    if not text:
        return []
    result: list[int] = []
    # Use a simple sliding window: every ~4 non-whitespace characters ≈ 1 token
    parts: list[str] = []
    for ch in text:
        if ch.isspace():
            if parts:
                result.append(hash("".join(parts)) & 0x7FFFFFFF)
                parts = []
        else:
            # Approximate: CJK characters and emojis are often single tokens
            cat = unicodedata.category(ch)
            if cat.startswith("C") or cat.startswith("Lo"):
                # Likely a single token (control, CJK, symbol, etc.)
                if parts:
                    result.append(hash("".join(parts)) & 0x7FFFFFFF)
                    parts = []
                result.append(hash(ch) & 0x7FFFFFFF)
            else:
                parts.append(ch)
                if len(parts) >= 4:
                    result.append(hash("".join(parts)) & 0x7FFFFFFF)
                    parts = []
    if parts:
        result.append(hash("".join(parts)) & 0x7FFFFFFF)
    return result


_LAZY_ENCODER = _LazyEncoder()


def _count_text_tokens(text: str) -> int:
    return len(_LAZY_ENCODER.encode(text, disallowed_special=_DISALLOWED_SPECIAL))


def get_token_count(
    messages: list,
    system: str | list | None = None,
    tools: list | None = None,
) -> int:
    """Estimate token count for a request."""
    total_tokens = 0

    if system:
        if isinstance(system, str):
            total_tokens += _count_text_tokens(system)
        elif isinstance(system, list):
            for block in system:
                text = get_block_attr(block, "text", "")
                if text:
                    total_tokens += _count_text_tokens(str(text))
        total_tokens += 4

    for msg in messages:
        if isinstance(msg.content, str):
            total_tokens += _count_text_tokens(msg.content)
        elif isinstance(msg.content, list):
            for block in msg.content:
                b_type = get_block_attr(block, "type") or None

                if b_type == "text":
                    text = get_block_attr(block, "text", "")
                    total_tokens += _count_text_tokens(str(text))
                elif b_type == "thinking":
                    thinking = get_block_attr(block, "thinking", "")
                    total_tokens += _count_text_tokens(str(thinking))
                elif b_type == "tool_use":
                    name = get_block_attr(block, "name", "")
                    inp = get_block_attr(block, "input", {})
                    block_id = get_block_attr(block, "id", "")
                    total_tokens += _count_text_tokens(str(name))
                    total_tokens += _count_text_tokens(json.dumps(inp))
                    total_tokens += _count_text_tokens(str(block_id))
                    total_tokens += 15
                elif b_type == "image":
                    source = get_block_attr(block, "source")
                    if isinstance(source, dict):
                        data = source.get("data") or source.get("base64") or ""
                        if data:
                            total_tokens += max(85, len(data) // 3000)
                        else:
                            total_tokens += 765
                    else:
                        total_tokens += 765
                elif b_type == "tool_result":
                    content = get_block_attr(block, "content", "")
                    tool_use_id = get_block_attr(block, "tool_use_id", "")
                    if isinstance(content, str):
                        total_tokens += _count_text_tokens(content)
                    else:
                        total_tokens += _count_text_tokens(json.dumps(content))
                    total_tokens += _count_text_tokens(str(tool_use_id))
                    total_tokens += 8
                elif b_type in (
                    "server_tool_use",
                    "web_search_tool_result",
                    "web_fetch_tool_result",
                ):
                    if hasattr(block, "model_dump"):
                        blob: object = block.model_dump()
                    else:
                        blob = block
                    try:
                        total_tokens += _count_text_tokens(
                            json.dumps(blob, default=str, ensure_ascii=False)
                        )
                    except (TypeError, ValueError, OverflowError) as e:
                        logger.debug(
                            "Block encode fallback b_type={} err={}", b_type, e
                        )
                        total_tokens += _count_text_tokens(str(blob))
                    total_tokens += 12
                else:
                    logger.debug(
                        "Unexpected block type %r, falling back to json/str encoding",
                        b_type,
                    )
                    try:
                        total_tokens += _count_text_tokens(json.dumps(block))
                    except (TypeError, ValueError):
                        total_tokens += _count_text_tokens(str(block))

    if tools:
        for tool in tools:
            tool_str = (
                tool.name + (tool.description or "") + json.dumps(tool.input_schema)
            )
            total_tokens += _count_text_tokens(tool_str)

    total_tokens += len(messages) * 4
    if tools:
        total_tokens += len(tools) * 5

    return max(1, total_tokens)
