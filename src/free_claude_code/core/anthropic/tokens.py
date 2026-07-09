"""Token estimation for Anthropic-compatible requests."""

import json

import tiktoken
from loguru import logger

from .content import get_block_attr

ENCODER = tiktoken.get_encoding("cl100k_base")

_DISALLOWED_SPECIAL: tuple[str, ...] = ()


def _count_text_tokens(text: str) -> int:
    return len(ENCODER.encode(text, disallowed_special=_DISALLOWED_SPECIAL))


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
                    except TypeError, ValueError:
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
