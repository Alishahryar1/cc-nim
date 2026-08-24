"""Encode canonical inference requests as OpenAI Chat Completions bodies."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

from free_claude_code.core.inference import (
    Base64MediaSource,
    CacheControl,
    CustomTool,
    DocumentContent,
    FunctionTool,
    ImageContent,
    InferenceItem,
    InferenceRequest,
    InstructionItem,
    InstructionPlacement,
    MessageItem,
    MessageRole,
    ReasoningItem,
    RefusalContent,
    ReplayArtifact,
    ReplayArtifactKind,
    ReplayCompatibilityScope,
    TextContent,
    ToolCallItem,
    ToolCallKind,
    ToolChoiceMode,
    ToolResultItem,
    UrlMediaSource,
    thaw_json,
    thaw_json_object,
)
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.providers.openai_compat import OpenAIToolNameCodec


class OpenAIConversionError(ValueError):
    """Canonical content cannot be represented by OpenAI Chat without loss."""


class ReasoningReplayMode(StrEnum):
    """How visible assistant reasoning is replayed to a Chat provider."""

    DISABLED = "disabled"
    THINK_TAGS = "think_tags"
    REASONING_CONTENT = "reasoning_content"
    REASONING = "reasoning"


@dataclass(slots=True)
class _PlainSegment:
    messages: list[JsonObject]


@dataclass(slots=True)
class _ToolTurnSegment:
    assistant_message: JsonObject
    required_tool_ids: list[str]
    deferred_items: list[InferenceItem] = field(default_factory=list)
    reasoning_replay: ReasoningReplayMode = ReasoningReplayMode.THINK_TAGS
    replay_scope: ReplayCompatibilityScope | None = None
    assistant_emitted: bool = False


type _TranscriptSegment = _PlainSegment | _ToolTurnSegment


class _SyntheticOpenAIToolTurnBoundary(dict[str, JsonValue]):
    __slots__ = ()


def is_synthetic_openai_tool_turn_boundary(message: object) -> bool:
    """Return whether FCC inserted a harmless assistant tool-turn boundary."""

    return isinstance(message, _SyntheticOpenAIToolTurnBoundary)


class _OpenAIChatHistoryLedger:
    """Emit tool calls before their results without reordering later content."""

    def __init__(
        self,
        *,
        tool_names: OpenAIToolNameCodec,
    ) -> None:
        self._tool_names = tool_names
        self._output: list[JsonObject] = []
        self._segments: list[_TranscriptSegment] = []
        self._tool_results: dict[str, JsonObject] = {}

    def add_plain(self, messages: list[JsonObject]) -> None:
        if messages:
            self._segments.append(_PlainSegment(messages))
            self._drain_ready_segments()

    def add_tool_turn(self, segment: _ToolTurnSegment) -> None:
        self._segments.append(segment)
        self._drain_ready_segments()

    def add_user_turn(self, items: list[InferenceItem]) -> None:
        visible: list[MessageItem] = []

        def flush_visible() -> None:
            if visible:
                self.add_plain(_user_messages(visible))
                visible.clear()

        for item in items:
            if isinstance(item, MessageItem):
                visible.append(item)
            elif isinstance(item, ToolResultItem):
                flush_visible()
                self._record_tool_result(item)
            else:
                raise OpenAIConversionError(
                    f"user turn contains unsupported {type(item).__name__}"
                )
        flush_visible()
        self._drain_ready_segments()

    def finish(self) -> list[JsonObject]:
        self._drain_ready_segments()
        missing = self._missing_required_tool_ids()
        if missing:
            raise OpenAIConversionError(
                "OpenAI Chat cannot replay incomplete tool history; missing "
                f"tool results for call IDs: {missing}"
            )
        while self._segments:
            segment = self._segments.pop(0)
            if isinstance(segment, _PlainSegment):
                self._output.extend(segment.messages)
            else:
                self._emit_tool_turn(segment)
        return _coalesce_user_messages(_close_tool_result_turns(self._output))

    def _record_tool_result(self, item: ToolResultItem) -> None:
        message: JsonObject = {
            "role": "tool",
            "tool_call_id": item.call_id,
            "content": serialize_tool_result_content(item.content),
        }
        if self._has_pending_tool_id(item.call_id):
            self._tool_results[item.call_id] = message
        else:
            self.add_plain([message])

    def _drain_ready_segments(self) -> None:
        while self._segments:
            segment = self._segments[0]
            if isinstance(segment, _PlainSegment):
                self._output.extend(segment.messages)
                self._segments.pop(0)
                continue
            if not segment.assistant_emitted:
                self._output.append(segment.assistant_message)
                segment.assistant_emitted = True
            if any(
                call_id not in self._tool_results
                for call_id in segment.required_tool_ids
            ):
                break
            self._segments.pop(0)
            for call_id in segment.required_tool_ids:
                self._output.append(self._tool_results.pop(call_id))
            self._output.extend(
                _assistant_messages(
                    segment.deferred_items,
                    reasoning_replay=segment.reasoning_replay,
                    replay_scope=segment.replay_scope,
                )
            )

    def _emit_tool_turn(self, segment: _ToolTurnSegment) -> None:
        if not segment.assistant_emitted:
            self._output.append(segment.assistant_message)
        for call_id in segment.required_tool_ids:
            result = self._tool_results.pop(call_id, None)
            if result is not None:
                self._output.append(result)
        self._output.extend(
            _assistant_messages(
                segment.deferred_items,
                reasoning_replay=segment.reasoning_replay,
                replay_scope=segment.replay_scope,
            )
        )

    def _missing_required_tool_ids(self) -> list[str]:
        return [
            call_id
            for segment in self._segments
            if isinstance(segment, _ToolTurnSegment)
            for call_id in segment.required_tool_ids
            if call_id not in self._tool_results
        ]

    def _has_pending_tool_id(self, call_id: str) -> bool:
        return any(
            isinstance(segment, _ToolTurnSegment)
            and call_id in segment.required_tool_ids
            for segment in self._segments
        )


def build_base_request_body(
    request: InferenceRequest,
    *,
    provider_model: str,
    tool_names: OpenAIToolNameCodec,
    replay_scope: ReplayCompatibilityScope | None,
    default_max_tokens: int | None = None,
    reasoning_replay: ReasoningReplayMode = ReasoningReplayMode.THINK_TAGS,
) -> JsonObject:
    """Build common OpenAI Chat fields from canonical semantics."""

    turns = _turns(request.items)
    ledger = _OpenAIChatHistoryLedger(tool_names=tool_names)
    for turn_items in turns:
        if all(isinstance(item, InstructionItem) for item in turn_items):
            text = "\n\n".join(
                item.text for item in turn_items if isinstance(item, InstructionItem)
            )
            ledger.add_plain([{"role": "user", "content": text}])
        elif any(isinstance(item, ToolResultItem) for item in turn_items) or all(
            isinstance(item, MessageItem) and item.role is MessageRole.USER
            for item in turn_items
        ):
            ledger.add_user_turn(turn_items)
        else:
            segment = _assistant_segment(
                turn_items,
                reasoning_replay=reasoning_replay,
                replay_scope=replay_scope,
                tool_names=tool_names,
            )
            if isinstance(segment, _ToolTurnSegment):
                ledger.add_tool_turn(segment)
            else:
                ledger.add_plain(segment.messages)

    messages = ledger.finish()
    top_level = [
        item
        for item in request.items
        if isinstance(item, InstructionItem)
        and item.placement is InstructionPlacement.TOP_LEVEL
    ]
    if top_level:
        messages.insert(0, _system_message(top_level))

    body: JsonObject = {"model": provider_model, "messages": messages}
    max_tokens = request.max_output_tokens or default_max_tokens
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.stop_sequences:
        body["stop"] = list(request.stop_sequences)
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
    return body


def serialize_tool_result_content(content: JsonValue) -> str:
    """Serialize canonical tool output into provider-safe text."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        return json.dumps(thaw_json(content), ensure_ascii=False)
    if isinstance(content, Sequence):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text":
                text = item.get("text")
                parts.append(text if isinstance(text, str) else str(text or ""))
            elif isinstance(item, Mapping | Sequence) and not isinstance(item, str):
                parts.append(json.dumps(thaw_json(item), ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _turns(items: tuple[InferenceItem, ...]) -> list[list[InferenceItem]]:
    turns: list[list[InferenceItem]] = []
    indexes: dict[str, int] = {}
    for item in items:
        if isinstance(item, InstructionItem):
            if item.placement is InstructionPlacement.TOP_LEVEL:
                continue
            turn_id = item.turn_id
        else:
            turn_id = item.turn_id
        if turn_id is None:
            raise OpenAIConversionError("transcript item has no turn identity")
        index = indexes.get(turn_id)
        if index is None:
            index = len(turns)
            indexes[turn_id] = index
            turns.append([])
        turns[index].append(item)
    return turns


def _assistant_segment(
    items: list[InferenceItem],
    *,
    reasoning_replay: ReasoningReplayMode,
    replay_scope: ReplayCompatibilityScope | None,
    tool_names: OpenAIToolNameCodec,
) -> _TranscriptSegment:
    first_tool = next(
        (index for index, item in enumerate(items) if isinstance(item, ToolCallItem)),
        None,
    )
    if first_tool is None:
        return _PlainSegment(
            _assistant_messages(
                items,
                reasoning_replay=reasoning_replay,
                replay_scope=replay_scope,
            )
        )
    calls = [item for item in items if isinstance(item, ToolCallItem)]
    pre_items = items[:first_tool]
    message = _assistant_message(
        pre_items,
        reasoning_replay=reasoning_replay,
        replay_scope=replay_scope,
    )
    message["tool_calls"] = [
        _tool_call(item, tool_names=tool_names, replay_scope=replay_scope)
        for item in calls
    ]
    if message.get("content") == " ":
        message["content"] = ""
    if reasoning_replay is ReasoningReplayMode.REASONING_CONTENT:
        message.setdefault("reasoning_content", "")
    return _ToolTurnSegment(
        assistant_message=message,
        required_tool_ids=[item.call_id for item in calls],
        deferred_items=[
            item
            for index, item in enumerate(items)
            if index > first_tool and not isinstance(item, ToolCallItem)
        ],
        reasoning_replay=reasoning_replay,
        replay_scope=replay_scope,
    )


def _assistant_messages(
    items: list[InferenceItem],
    *,
    reasoning_replay: ReasoningReplayMode,
    replay_scope: ReplayCompatibilityScope | None,
) -> list[JsonObject]:
    if not items:
        return []
    return [
        _assistant_message(
            items,
            reasoning_replay=reasoning_replay,
            replay_scope=replay_scope,
        )
    ]


def _assistant_message(
    items: list[InferenceItem],
    *,
    reasoning_replay: ReasoningReplayMode,
    replay_scope: ReplayCompatibilityScope | None,
) -> JsonObject:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_details: list[JsonValue] = []
    for item in items:
        if isinstance(item, MessageItem):
            if item.role is not MessageRole.ASSISTANT:
                raise OpenAIConversionError("assistant turn contains user content")
            for part in item.content:
                if isinstance(part, TextContent):
                    text_parts.append(part.text)
                elif isinstance(part, RefusalContent):
                    text_parts.append(part.refusal)
                else:
                    raise OpenAIConversionError(
                        "OpenAI Chat cannot represent assistant media content"
                    )
        elif isinstance(item, ReasoningItem):
            if reasoning_replay is ReasoningReplayMode.THINK_TAGS and item.reasoning:
                text_parts.append(f"<think>\n{item.reasoning}\n</think>")
            elif (
                reasoning_replay is not ReasoningReplayMode.DISABLED and item.reasoning
            ):
                reasoning_parts.append(item.reasoning)
            reasoning_details.extend(
                _reasoning_details(item.artifacts, replay_scope=replay_scope)
            )
        else:
            raise OpenAIConversionError(
                f"assistant turn contains unsupported {type(item).__name__}"
            )
    message: JsonObject = {
        "role": "assistant",
        "content": "\n\n".join(text_parts) or " ",
    }
    if reasoning_parts and reasoning_replay in {
        ReasoningReplayMode.REASONING_CONTENT,
        ReasoningReplayMode.REASONING,
    }:
        message[reasoning_replay.value] = "\n".join(reasoning_parts)
    if reasoning_details:
        message["reasoning_details"] = [
            thaw_json(detail) for detail in reasoning_details
        ]
    return message


def _user_messages(items: list[MessageItem]) -> list[JsonObject]:
    content_parts: list[JsonObject] = []
    for item in items:
        if item.role is not MessageRole.USER:
            raise OpenAIConversionError("user turn contains assistant content")
        for part in item.content:
            if isinstance(part, TextContent):
                content: JsonObject = {"type": "text", "text": part.text}
                _add_cache_control(content, part.cache_control)
                content_parts.append(content)
            elif isinstance(part, ImageContent):
                content_parts.append(_image_part(part))
            elif isinstance(part, DocumentContent):
                raise OpenAIConversionError(
                    "OpenAI Chat cannot represent document content without data loss"
                )
            elif isinstance(part, RefusalContent):
                content_parts.append({"type": "text", "text": part.refusal})
    if all(
        part.get("type") == "text" and "cache_control" not in part
        for part in content_parts
    ):
        message_content: JsonValue = "\n".join(
            str(part.get("text", "")) for part in content_parts
        )
    else:
        message_content = content_parts
    return [{"role": "user", "content": message_content}]


def _system_message(items: list[InstructionItem]) -> JsonObject:
    if all(item.cache_control is None for item in items):
        return {
            "role": "system",
            "content": "\n\n".join(item.text for item in items).strip(),
        }
    parts: list[JsonObject] = []
    for item in items:
        part: JsonObject = {"type": "text", "text": item.text}
        _add_cache_control(part, item.cache_control)
        parts.append(part)
    return {"role": "system", "content": parts}


def _image_part(content: ImageContent) -> JsonObject:
    if isinstance(content.source, UrlMediaSource):
        url = content.source.url
    elif isinstance(content.source, Base64MediaSource):
        url = f"data:{content.source.media_type};base64,{content.source.data}"
    else:
        raise OpenAIConversionError("unsupported canonical image source")
    part: JsonObject = {"type": "image_url", "image_url": {"url": url}}
    _add_cache_control(part, content.cache_control)
    return part


def _tool_definition(
    tool: FunctionTool | CustomTool,
    *,
    tool_names: OpenAIToolNameCodec,
) -> JsonObject:
    kind = (
        ToolCallKind.CUSTOM if isinstance(tool, CustomTool) else ToolCallKind.FUNCTION
    )
    if isinstance(tool, FunctionTool):
        parameters = thaw_json_object(tool.input_schema)
        description = tool.description or ""
        strict = tool.strict
    else:
        parameters = {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Free-form input for the custom tool.",
                }
            },
            "required": ["input"],
        }
        format_parts = [f"Custom tool input format: {tool.format.type.value}."]
        if tool.format.syntax:
            format_parts.append(f"Syntax: {tool.format.syntax}.")
        if tool.format.definition:
            format_parts.append(tool.format.definition)
        description = "\n\n".join(
            part for part in (tool.description, *format_parts) if part
        )
        strict = False
    function: JsonObject = {
        "name": tool_names.encode(
            tool.name,
            kind=kind,
            namespace=tool.namespace,
        ),
        "description": description,
        "parameters": parameters,
    }
    if strict:
        function["strict"] = True
    definition: JsonObject = {"type": "function", "function": function}
    _add_cache_control(definition, tool.cache_control)
    return definition


def _tool_choice(
    request: InferenceRequest,
    *,
    tool_names: OpenAIToolNameCodec,
) -> JsonValue:
    choice = request.tool_choice
    if choice is None:
        return "auto"
    if choice.mode is ToolChoiceMode.AUTO:
        return "auto"
    if choice.mode is ToolChoiceMode.NONE:
        return "none"
    if choice.mode is ToolChoiceMode.REQUIRED:
        return "required"
    if choice.kind is None or choice.name is None:
        raise OpenAIConversionError("specific tool choice has no identity")
    return {
        "type": "function",
        "function": {
            "name": tool_names.encode(
                choice.name,
                kind=choice.kind,
                namespace=choice.namespace,
            )
        },
    }


def _tool_call(
    item: ToolCallItem,
    *,
    tool_names: OpenAIToolNameCodec,
    replay_scope: ReplayCompatibilityScope | None,
) -> JsonObject:
    if item.kind is ToolCallKind.CUSTOM:
        arguments_value: JsonValue = {"input": _custom_input_text(item.input)}
    else:
        arguments_value = item.input
    call: JsonObject = {
        "id": item.call_id,
        "type": "function",
        "function": {
            "name": tool_names.encode(
                item.name,
                kind=item.kind,
                namespace=item.namespace,
            ),
            "arguments": json.dumps(
                thaw_json(arguments_value),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }
    if extra_content := _tool_extra_content(
        item.artifacts,
        replay_scope=replay_scope,
    ):
        call["extra_content"] = extra_content
    return call


def _custom_input_text(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        raw = value.get("input")
        if isinstance(raw, str):
            return raw
    return json.dumps(thaw_json(value), ensure_ascii=False, separators=(",", ":"))


def _reasoning_details(
    artifacts: tuple[ReplayArtifact, ...],
    *,
    replay_scope: ReplayCompatibilityScope | None,
) -> list[JsonValue]:
    details: list[JsonValue] = []
    for artifact in artifacts:
        if (
            artifact.kind is not ReplayArtifactKind.REASONING_DETAILS
            or not _scope_matches(artifact, replay_scope)
        ):
            continue
        payload = artifact.payload
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                details.extend(cast(list[JsonValue], parsed))
            elif isinstance(parsed, dict):
                details.append(cast(JsonValue, parsed))
        elif isinstance(payload, Sequence) and not isinstance(payload, str):
            details.extend(payload)
        else:
            details.append(payload)
    return details


def _tool_extra_content(
    artifacts: tuple[ReplayArtifact, ...],
    *,
    replay_scope: ReplayCompatibilityScope | None,
) -> JsonObject:
    result: JsonObject = {}
    for artifact in artifacts:
        if not _scope_matches(artifact, replay_scope):
            continue
        if artifact.kind is ReplayArtifactKind.THOUGHT_SIGNATURE and isinstance(
            artifact.payload, str
        ):
            result["google"] = {"thought_signature": artifact.payload}
        elif artifact.kind is ReplayArtifactKind.TOOL_EXTRA_CONTENT and isinstance(
            artifact.payload, Mapping
        ):
            result.update(thaw_json_object(artifact.payload))
    return result


def _scope_matches(
    artifact: ReplayArtifact,
    replay_scope: ReplayCompatibilityScope | None,
) -> bool:
    return replay_scope is not None and artifact.scope == replay_scope


def _add_cache_control(target: JsonObject, cache_control: CacheControl | None) -> None:
    if cache_control is None:
        return
    value: dict[str, str] = {"type": cache_control.type.value}
    if cache_control.ttl is not None:
        value["ttl"] = cache_control.ttl.value
    target["cache_control"] = value


def _coalesce_user_messages(
    messages: list[JsonObject],
) -> list[JsonObject]:
    result: list[JsonObject] = []
    for message in messages:
        if (
            message.get("role") == "user"
            and result
            and result[-1].get("role") == "user"
        ):
            previous = result[-1]
            previous["content"] = _merge_user_content(
                previous.get("content"),
                message.get("content"),
            )
        else:
            result.append(message)
    return result


def _merge_user_content(first: JsonValue, second: JsonValue) -> JsonValue:
    if isinstance(first, str) and isinstance(second, str):
        return f"{first}\n\n{second}"
    first_parts = _user_content_parts(first)
    second_parts = _user_content_parts(second)
    if (
        first_parts
        and second_parts
        and first_parts[-1].get("type") == "text"
        and second_parts[0].get("type") == "text"
        and "cache_control" not in first_parts[-1]
        and "cache_control" not in second_parts[0]
    ):
        first_text = str(first_parts[-1].get("text", ""))
        second_text = str(second_parts.pop(0).get("text", ""))
        first_parts[-1]["text"] = f"{first_text}\n\n{second_text}"
    return [*first_parts, *second_parts]


def _user_content_parts(content: JsonValue) -> list[JsonObject]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, Sequence) and not isinstance(content, str):
        parts: list[JsonObject] = []
        for part in content:
            if not isinstance(part, Mapping):
                raise OpenAIConversionError("OpenAI Chat produced invalid user content")
            parts.append(thaw_json_object(part))
        return parts
    raise OpenAIConversionError("OpenAI Chat produced invalid user content")


def _close_tool_result_turns(
    messages: list[JsonObject],
) -> list[JsonObject]:
    result: list[JsonObject] = []
    for message in messages:
        if (
            message.get("role") == "user"
            and result
            and result[-1].get("role") == "tool"
        ):
            result.append(
                _SyntheticOpenAIToolTurnBoundary(role="assistant", content=" ")
            )
        result.append(message)
    return result
