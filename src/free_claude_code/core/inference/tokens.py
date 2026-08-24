"""Token estimation over the canonical inference request."""

import json
from collections.abc import Mapping, Sequence

from free_claude_code.core.json_types import JsonValue
from free_claude_code.core.token_estimation import estimate_text_tokens

from .request import (
    Base64MediaSource,
    DocumentContent,
    FunctionTool,
    ImageContent,
    InferenceRequest,
    InstructionItem,
    InstructionPlacement,
    MessageItem,
    ReasoningItem,
    RefusalContent,
    TextContent,
    ToolCallItem,
    ToolResultItem,
    UrlMediaSource,
    thaw_json,
)


def get_inference_token_count(request: InferenceRequest) -> int:
    """Estimate input tokens from the one canonical transcript interpretation."""

    total = 0
    has_top_level_instruction = False
    for item in request.items:
        if isinstance(item, InstructionItem):
            total += estimate_text_tokens(item.text)
            has_top_level_instruction |= (
                item.placement is InstructionPlacement.TOP_LEVEL
            )
        elif isinstance(item, MessageItem):
            total += sum(_content_tokens(part) for part in item.content)
        elif isinstance(item, ReasoningItem):
            total += estimate_text_tokens(item.reasoning)
        elif isinstance(item, ToolCallItem):
            total += estimate_text_tokens(item.name)
            total += estimate_text_tokens(_json_text(item.input))
            total += estimate_text_tokens(item.call_id)
            total += 15
        elif isinstance(item, ToolResultItem):
            total += estimate_text_tokens(_tool_result_text(item.content))
            total += estimate_text_tokens(item.call_id)
            total += 8

    if has_top_level_instruction:
        total += 4
    total += request.message_count * 4

    for tool in request.tools:
        if isinstance(tool, FunctionTool):
            schema = _json_text(tool.input_schema)
        else:
            schema = " ".join(
                value
                for value in (
                    tool.format.type.value,
                    tool.format.syntax,
                    tool.format.definition,
                )
                if value
            )
        total += estimate_text_tokens(tool.name + (tool.description or "") + schema)
    if request.tools:
        total += len(request.tools) * 5
    return max(1, total)


def _content_tokens(content: object) -> int:
    if isinstance(content, TextContent):
        return estimate_text_tokens(content.text)
    if isinstance(content, RefusalContent):
        return estimate_text_tokens(content.refusal)
    if isinstance(content, ImageContent):
        return _media_tokens(content.source)
    if isinstance(content, DocumentContent):
        source = content.source
        if isinstance(source, Base64MediaSource):
            return max(85, len(source.data) // 3000)
        if isinstance(source, UrlMediaSource):
            return 765
        return estimate_text_tokens(source.file_id)
    raise TypeError(f"unsupported canonical content: {type(content).__name__}")


def _media_tokens(source: UrlMediaSource | Base64MediaSource) -> int:
    if isinstance(source, Base64MediaSource):
        return max(85, len(source.data) // 3000)
    return 765


def _tool_result_text(value: JsonValue) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return _json_text(value)
    if isinstance(value, Sequence):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping) and item.get("type") == "text":
                text = item.get("text")
                parts.append(text if isinstance(text, str) else str(text or ""))
            elif isinstance(item, Mapping | Sequence) and not isinstance(item, str):
                parts.append(_json_text(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)


def _json_text(value: JsonValue | Mapping[str, JsonValue]) -> str:
    return json.dumps(
        thaw_json(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )
