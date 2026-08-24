"""Encode canonical inference requests for an OpenAI Responses upstream."""

import json
from collections.abc import Mapping, Sequence

from free_claude_code.core.inference import (
    Base64MediaSource,
    CustomTool,
    DocumentContent,
    FunctionTool,
    ImageContent,
    InferenceItem,
    InferenceRequest,
    InstructionItem,
    MessageItem,
    MessageRole,
    ReasoningItem,
    RefusalContent,
    ReplayAttachment,
    ReplayCompatibilityScope,
    TextContent,
    ToolCallItem,
    ToolCallKind,
    ToolChoiceMode,
    ToolResultItem,
    UrlMediaSource,
    replay_payload_text,
    thaw_json,
    thaw_json_object,
)
from free_claude_code.core.json_types import JsonValue
from free_claude_code.core.reasoning import ReasoningControl, ReasoningPolicy
from free_claude_code.providers.openai_compat import OpenAIToolNameCodec


class ResponsesRequestEncodingError(ValueError):
    """Canonical semantics cannot be represented by Responses without loss."""


def build_responses_request_body(
    request: InferenceRequest,
    *,
    provider_model: str,
    reasoning: ReasoningPolicy,
    tool_names: OpenAIToolNameCodec,
    replay_scope: ReplayCompatibilityScope,
) -> dict[str, object]:
    """Build one stateless Responses request from canonical values."""

    _validate_request(request)
    instructions: list[str] = []
    input_items: list[dict[str, object]] = []
    for items in _turns(request.items):
        for item in items:
            if isinstance(item, InstructionItem):
                _require_no_cache_control(item.cache_control, "instruction")
                instructions.append(item.text)
            elif isinstance(item, MessageItem):
                input_items.append(_message_item(item))
            elif isinstance(item, ReasoningItem):
                input_items.append(_reasoning_item(item, replay_scope=replay_scope))
            elif isinstance(item, ToolCallItem):
                input_items.append(
                    _tool_call_item(
                        item,
                        tool_names=tool_names,
                        replay_scope=replay_scope,
                    )
                )
            elif isinstance(item, ToolResultItem):
                input_items.append(_tool_result_item(item))

    if not input_items:
        raise ResponsesRequestEncodingError(
            "OpenAI Responses requires at least one conversational input item"
        )

    body: dict[str, object] = {
        "model": provider_model,
        "input": input_items,
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }
    if instructions:
        body["instructions"] = "\n\n".join(instructions)
    if request.max_output_tokens is not None:
        body["max_output_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.metadata is not None:
        body["metadata"] = thaw_json_object(request.metadata)
    if request.tools:
        body["tools"] = [
            _tool_definition(tool, tool_names=tool_names) for tool in request.tools
        ]
    if request.tool_choice is not None:
        body["tool_choice"] = _tool_choice(request, tool_names=tool_names)
    elif request.tools:
        body["tool_choice"] = "auto"
    if request.parallel_tool_calls is not None:
        body["parallel_tool_calls"] = request.parallel_tool_calls
    if config := _reasoning_config(reasoning):
        body["reasoning"] = config
    return body


def _validate_request(request: InferenceRequest) -> None:
    unsupported: list[str] = []
    if request.stop_sequences:
        unsupported.append("stop_sequences")
    if request.top_k is not None:
        unsupported.append("top_k")
    if request.extensions:
        unsupported.append("extensions")
    if unsupported:
        raise ResponsesRequestEncodingError(
            f"OpenAI Responses cannot represent these canonical fields: {unsupported}"
        )


def _turns(items: tuple[InferenceItem, ...]) -> list[list[InferenceItem]]:
    turns: list[list[InferenceItem]] = []
    turn_indexes: dict[str, int] = {}
    top_level: list[InferenceItem] = []
    for item in items:
        turn_id = item.turn_id
        if turn_id is None:
            top_level.append(item)
            continue
        index = turn_indexes.get(turn_id)
        if index is None:
            index = len(turns)
            turn_indexes[turn_id] = index
            turns.append([])
        turns[index].append(item)
    return ([top_level] if top_level else []) + turns


def _message_item(item: MessageItem) -> dict[str, object]:
    content: list[dict[str, object]] = []
    for part in item.content:
        if isinstance(part, TextContent):
            _require_no_cache_control(part.cache_control, "message text")
            content.append(
                {
                    "type": (
                        "input_text" if item.role is MessageRole.USER else "output_text"
                    ),
                    "text": part.text,
                }
            )
        elif isinstance(part, RefusalContent):
            if item.role is MessageRole.USER:
                content.append({"type": "input_text", "text": part.refusal})
            else:
                content.append({"type": "refusal", "refusal": part.refusal})
        elif isinstance(part, ImageContent):
            if item.role is not MessageRole.USER:
                raise ResponsesRequestEncodingError(
                    "Responses cannot replay assistant image content"
                )
            _require_no_cache_control(part.cache_control, "image")
            content.append({"type": "input_image", "image_url": _image_url(part)})
        elif isinstance(part, DocumentContent):
            raise ResponsesRequestEncodingError(
                "Responses transport does not support canonical document content"
            )
    return {"type": "message", "role": item.role.value, "content": content}


def _image_url(content: ImageContent) -> str:
    if isinstance(content.source, UrlMediaSource):
        return content.source.url
    if isinstance(content.source, Base64MediaSource):
        return f"data:{content.source.media_type};base64,{content.source.data}"
    raise ResponsesRequestEncodingError("unsupported canonical image source")


def _reasoning_item(
    item: ReasoningItem,
    *,
    replay_scope: ReplayCompatibilityScope,
) -> dict[str, object]:
    result: dict[str, object] = {
        "type": "reasoning",
        "summary": (
            [{"type": "summary_text", "text": item.reasoning}] if item.reasoning else []
        ),
    }
    matching = [
        artifact
        for artifact in item.artifacts
        if artifact.attachment is ReplayAttachment.REASONING
        and artifact.scope == replay_scope
    ]
    if matching:
        if len(matching) != 1:
            raise ResponsesRequestEncodingError(
                "Responses encrypted reasoning accepts one scoped replay artifact"
            )
        result["encrypted_content"] = replay_payload_text(matching[0])
    return result


def _tool_call_item(
    item: ToolCallItem,
    *,
    tool_names: OpenAIToolNameCodec,
    replay_scope: ReplayCompatibilityScope,
) -> dict[str, object]:
    if any(
        artifact.scope == replay_scope
        and artifact.attachment is ReplayAttachment.TOOL_CALL
        for artifact in item.artifacts
    ):
        raise ResponsesRequestEncodingError(
            "Responses has no standard tool-call replay carrier"
        )
    name = tool_names.encode(
        item.name,
        kind=item.kind,
        namespace=item.namespace,
    )
    if item.kind is ToolCallKind.CUSTOM:
        custom_input = thaw_json(item.input)
        return {
            "type": "custom_tool_call",
            "call_id": item.call_id,
            "name": name,
            "input": (
                custom_input
                if isinstance(custom_input, str)
                else json.dumps(custom_input, ensure_ascii=False, separators=(",", ":"))
            ),
        }
    return {
        "type": "function_call",
        "call_id": item.call_id,
        "name": name,
        "arguments": json.dumps(
            thaw_json(item.input), ensure_ascii=False, separators=(",", ":")
        ),
    }


def _tool_result_item(item: ToolResultItem) -> dict[str, object]:
    return {
        "type": "function_call_output",
        "call_id": item.call_id,
        "output": _serialize_tool_result(item.content),
    }


def _serialize_tool_result(content: JsonValue) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        return json.dumps(thaw_json(content), ensure_ascii=False)
    if isinstance(content, Sequence):
        parts: list[str] = []
        for value in content:
            if isinstance(value, Mapping) and value.get("type") == "text":
                text = value.get("text")
                parts.append(text if isinstance(text, str) else str(text or ""))
            elif isinstance(value, Mapping | Sequence) and not isinstance(value, str):
                parts.append(json.dumps(thaw_json(value), ensure_ascii=False))
            else:
                parts.append(str(value))
        return "\n".join(parts)
    return str(content)


def _tool_definition(
    tool: FunctionTool | CustomTool,
    *,
    tool_names: OpenAIToolNameCodec,
) -> dict[str, object]:
    kind = (
        ToolCallKind.CUSTOM if isinstance(tool, CustomTool) else ToolCallKind.FUNCTION
    )
    name = tool_names.encode(tool.name, kind=kind, namespace=tool.namespace)
    _require_no_cache_control(tool.cache_control, "tool definition")
    if isinstance(tool, FunctionTool):
        return {
            "type": "function",
            "name": name,
            "description": tool.description,
            "parameters": thaw_json_object(tool.input_schema),
            "strict": tool.strict,
        }
    result: dict[str, object] = {
        "type": "custom",
        "name": name,
        "description": tool.description,
        "format": {"type": tool.format.type.value},
    }
    custom_format = result["format"]
    if isinstance(custom_format, dict):
        if tool.format.syntax is not None:
            custom_format["syntax"] = tool.format.syntax
        if tool.format.definition is not None:
            custom_format["definition"] = tool.format.definition
    return result


def _tool_choice(
    request: InferenceRequest,
    *,
    tool_names: OpenAIToolNameCodec,
) -> object:
    choice = request.tool_choice
    if choice is None or choice.mode is ToolChoiceMode.AUTO:
        return "auto"
    if choice.mode is ToolChoiceMode.NONE:
        return "none"
    if choice.mode is ToolChoiceMode.REQUIRED:
        return "required"
    if choice.name is None or choice.kind is None:
        raise ResponsesRequestEncodingError("specific tool choice has no identity")
    return {
        "type": "custom" if choice.kind is ToolCallKind.CUSTOM else "function",
        "name": tool_names.encode(
            choice.name,
            kind=choice.kind,
            namespace=choice.namespace,
        ),
    }


def _reasoning_config(reasoning: ReasoningPolicy) -> dict[str, str]:
    if reasoning.control is ReasoningControl.OFF:
        return {"effort": "none"}
    if reasoning.effort is not None:
        return {"effort": reasoning.effort.value, "summary": "auto"}
    if reasoning.requests_reasoning:
        return {"summary": "auto"}
    return {}


def _require_no_cache_control(value: object, path: str) -> None:
    if value is not None:
        raise ResponsesRequestEncodingError(
            f"Responses cannot represent {path} cache_control"
        )
