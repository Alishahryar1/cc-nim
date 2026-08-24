"""Parse exact OpenAI-compatible function-tag tool envelopes."""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import cast

from free_claude_code.core.anthropic.tool_schema import (
    arguments_match_schema,
    coerce_text_argument,
)
from free_claude_code.core.inference import (
    FunctionTool,
    InferenceRequest,
    ToolChoiceMode,
    thaw_json_object,
)
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.providers.openai_compat import OpenAIToolNameCodec

_FUNCTION_TAG_BLOCK_START = "<tool_call>"
_FUNCTION_TAG_BLOCK_END = "</tool_call>"
_FUNCTION_TAG_START = "<function="
_FUNCTION_TAG_END = "</function>"
_PARAMETER_TAG_START = "<parameter="
_PARAMETER_TAG_END = "</parameter>"
_MAX_FUNCTION_TAG_CANDIDATE_CHARS = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _RawFunctionTagCall:
    name: str
    arguments: dict[str, str]


class _FunctionTagState(Enum):
    SEARCHING = 1
    CANDIDATE = 2
    DISABLED = 3
    FINISHED = 4


class FunctionTagToolParser:
    """Parse an exact terminal function-tag envelope into tool use."""

    def __init__(self, request: InferenceRequest):
        self._tool_names = OpenAIToolNameCodec.from_request(request)
        self._schemas = {
            tool.name: thaw_json_object(tool.input_schema)
            for tool in request.tools
            if isinstance(tool, FunctionTool)
        }
        tool_choice_disabled = (
            request.tool_choice is not None
            and request.tool_choice.mode is ToolChoiceMode.NONE
        )
        self._state = (
            _FunctionTagState.SEARCHING
            if self._schemas and not tool_choice_disabled
            else _FunctionTagState.DISABLED
        )
        self._parts: list[str] = []
        self._length = 0
        self._marker_tail = ""

    def feed(self, text: str) -> str:
        """Hold a possible reserved response and return text safe to expose."""
        if not text:
            return ""
        if self._state in {_FunctionTagState.DISABLED, _FunctionTagState.FINISHED}:
            return text

        if self._state is _FunctionTagState.CANDIDATE:
            self._parts.append(text)
            self._length += len(text)
            if self._length > _MAX_FUNCTION_TAG_CANDIDATE_CHARS:
                return self.disable()
            return ""

        candidate = "".join((self._marker_tail, text))
        marker_index = candidate.find(_FUNCTION_TAG_BLOCK_START)
        if marker_index >= 0:
            visible = candidate[:marker_index]
            control = candidate[marker_index:]
            self._state = _FunctionTagState.CANDIDATE
            self._marker_tail = ""
            self._parts.append(control)
            self._length = len(control)
            if self._length > _MAX_FUNCTION_TAG_CANDIDATE_CHARS:
                return "".join((visible, self.disable()))
            return visible

        held_length = _partial_function_tag_marker_suffix_length(candidate)
        if held_length:
            self._marker_tail = candidate[-held_length:]
            return candidate[:-held_length]
        self._marker_tail = ""
        return candidate

    def disable(self) -> str:
        """Disable textual recovery and release any held candidate unchanged."""
        if self._state in {_FunctionTagState.DISABLED, _FunctionTagState.FINISHED}:
            return ""
        self._state = _FunctionTagState.DISABLED
        text = "".join((self._marker_tail, *self._parts))
        self._marker_tail = ""
        self._parts.clear()
        self._length = 0
        return text

    def finish(self) -> tuple[str, tuple[JsonObject, ...]]:
        """Finalize one response atomically as visible text or validated tools."""
        if self._state in {_FunctionTagState.DISABLED, _FunctionTagState.FINISHED}:
            return "", ()
        if self._state is _FunctionTagState.SEARCHING:
            self._state = _FunctionTagState.FINISHED
            text = self._marker_tail
            self._marker_tail = ""
            return text, ()
        self._state = _FunctionTagState.FINISHED
        raw = "".join(self._parts)
        self._parts.clear()
        self._length = 0
        try:
            calls = _parse_function_tag_calls(raw)
            tool_uses = self._validated_tool_uses(calls)
        except ValueError:
            return raw, ()
        return "", tool_uses

    def _validated_tool_uses(
        self, calls: tuple[_RawFunctionTagCall, ...]
    ) -> tuple[JsonObject, ...]:
        tool_uses: list[JsonObject] = []
        for call in calls:
            name = self._tool_names.decode(call.name)
            schema = self._schemas.get(name)
            if schema is None:
                raise ValueError
            properties = schema.get("properties")
            property_schemas = properties if isinstance(properties, Mapping) else {}
            additional = schema.get("additionalProperties")
            arguments: JsonObject = {}
            for parameter_name, value in call.arguments.items():
                parameter_schema = property_schemas.get(parameter_name)
                if not isinstance(parameter_schema, Mapping):
                    parameter_schema = (
                        additional if isinstance(additional, Mapping) else {}
                    )
                arguments[parameter_name] = cast(
                    JsonValue,
                    coerce_text_argument(value, parameter_schema),
                )
            if not arguments_match_schema(arguments, schema):
                raise ValueError
            tool_use: JsonObject = {
                "type": "tool_use",
                "id": f"toolu_function_tag_{uuid.uuid4().hex[:8]}",
                "name": name,
                "input": arguments,
            }
            tool_uses.append(tool_use)
        return tuple(tool_uses)


def _parse_function_tag_calls(text: str) -> tuple[_RawFunctionTagCall, ...]:
    cursor = 0
    calls: list[_RawFunctionTagCall] = []
    while True:
        cursor = _skip_function_tag_whitespace(text, cursor)
        if cursor == len(text):
            break
        if not text.startswith(_FUNCTION_TAG_BLOCK_START, cursor):
            raise ValueError

        block_start = cursor + len(_FUNCTION_TAG_BLOCK_START)
        block_end = text.find(_FUNCTION_TAG_BLOCK_END, block_start)
        if block_end < 0:
            raise ValueError
        name, arguments = _parse_function_tag_block(text[block_start:block_end])
        calls.append(_RawFunctionTagCall(name=name, arguments=arguments))
        cursor = block_end + len(_FUNCTION_TAG_BLOCK_END)

    if not calls:
        raise ValueError
    return tuple(calls)


def _parse_function_tag_block(block: str) -> tuple[str, dict[str, str]]:
    cursor = _skip_function_tag_whitespace(block, 0)
    if not block.startswith(_FUNCTION_TAG_START, cursor):
        raise ValueError
    name_end = block.find(">", cursor + len(_FUNCTION_TAG_START))
    if name_end < 0:
        raise ValueError
    name = block[cursor + len(_FUNCTION_TAG_START) : name_end]
    if not _valid_function_tag_name(name):
        raise ValueError

    arguments: dict[str, str] = {}
    cursor = name_end + 1
    while True:
        cursor = _skip_function_tag_whitespace(block, cursor)
        if block.startswith(_FUNCTION_TAG_END, cursor):
            cursor += len(_FUNCTION_TAG_END)
            break
        if cursor == len(block) or not block.startswith(_PARAMETER_TAG_START, cursor):
            raise ValueError

        parameter_name_end = block.find(">", cursor + len(_PARAMETER_TAG_START))
        if parameter_name_end < 0:
            raise ValueError
        parameter_name = block[cursor + len(_PARAMETER_TAG_START) : parameter_name_end]
        if not _valid_function_tag_name(parameter_name) or parameter_name in arguments:
            raise ValueError

        value_start = parameter_name_end + 1
        value_end = block.find(_PARAMETER_TAG_END, value_start)
        if value_end < 0:
            raise ValueError
        arguments[parameter_name] = _unwrap_function_tag_newlines(
            block[value_start:value_end]
        )
        cursor = value_end + len(_PARAMETER_TAG_END)

    if block[cursor:].strip():
        raise ValueError
    return name, arguments


def _skip_function_tag_whitespace(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor


def _valid_function_tag_name(value: str) -> bool:
    return bool(value) and not any(
        character.isspace() or character in "<>" for character in value
    )


def _unwrap_function_tag_newlines(value: str) -> str:
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    if value.endswith("\r\n"):
        return value[:-2]
    if value.endswith("\n"):
        return value[:-1]
    return value


def _partial_function_tag_marker_suffix_length(text: str) -> int:
    max_length = min(len(text), len(_FUNCTION_TAG_BLOCK_START) - 1)
    for length in range(max_length, 0, -1):
        if _FUNCTION_TAG_BLOCK_START.startswith(text[-length:]):
            return length
    return 0
