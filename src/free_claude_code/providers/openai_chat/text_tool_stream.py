"""Normalize exact textual tool-call dialects on OpenAI-compatible streams."""

import json
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import SimpleNamespace
from typing import Any

from free_claude_code.providers.failure_policy import RetryableToolProtocolError
from free_claude_code.providers.http import maybe_await_aclose
from free_claude_code.providers.native_tool_schema import (
    arguments_match_schema,
    coerce_text_argument,
    openai_tool_schemas,
)

_TOOL_BLOCK_START = "<tool_call>"
_TOOL_BLOCK_END = "</tool_call>"
_FUNCTION_START = "<function="
_FUNCTION_END = "</function>"
_PARAMETER_START = "<parameter="
_PARAMETER_END = "</parameter>"
_MAX_TEXT_TOOL_BLOCK_CHARS = 4 * 1024 * 1024


class OpenAITextToolProtocolError(RetryableToolProtocolError):
    """An OpenAI-compatible endpoint leaked malformed textual tool markup."""


class _NormalizedTextToolStream(AsyncIterator[Any]):
    """Expose normalized chunks while retaining the raw stream's close contract."""

    def __init__(self, iterator: AsyncIterator[Any], source: Any) -> None:
        self._iterator = iterator
        self._source = source
        self._closed = False

    def __aiter__(self) -> _NormalizedTextToolStream:
        return self

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        return await anext(self._iterator)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await maybe_await_aclose(self._iterator)
        await maybe_await_aclose(self._source)


@dataclass(frozen=True, slots=True)
class _TextToolCall:
    index: int
    name: str
    arguments: dict[str, Any]


class _FramingMode(StrEnum):
    TEXT = "text"
    CONTROL = "control"
    FINISHED = "finished"


class _TextToolFramer:
    """Hold an exact textual control envelope without exposing it as text."""

    def __init__(self) -> None:
        self._mode = _FramingMode.TEXT
        self._text_tail = ""
        self._control_parts: list[str] = []
        self._control_length = 0

    @property
    def control(self) -> str | None:
        if not self._control_parts:
            return None
        return "".join(self._control_parts)

    def feed(self, text: str) -> str:
        if self._mode is _FramingMode.FINISHED:
            raise OpenAITextToolProtocolError(
                "OpenAI-compatible provider returned content after a terminal chunk."
            )
        if not text:
            return ""
        if self._mode is _FramingMode.CONTROL:
            self._append_control(text)
            return ""

        candidate = "".join((self._text_tail, text))
        marker_index = candidate.find(_TOOL_BLOCK_START)
        if marker_index >= 0:
            visible = candidate[:marker_index]
            self._text_tail = ""
            self._mode = _FramingMode.CONTROL
            self._append_control(candidate[marker_index:])
            return visible

        held_length = _partial_marker_suffix_length(candidate, _TOOL_BLOCK_START)
        if held_length:
            visible = candidate[:-held_length]
            self._text_tail = candidate[-held_length:]
            return visible

        self._text_tail = ""
        return candidate

    def finish(self) -> str:
        if self._mode is _FramingMode.FINISHED:
            return ""
        self._mode = _FramingMode.FINISHED
        if self._control_parts:
            return ""
        tail = self._text_tail
        self._text_tail = ""
        return tail

    def _append_control(self, text: str) -> None:
        if not text:
            return
        self._control_length += len(text)
        if self._control_length > _MAX_TEXT_TOOL_BLOCK_CHARS:
            raise OpenAITextToolProtocolError(
                "OpenAI-compatible provider returned oversized textual tool markup."
            )
        self._control_parts.append(text)


def normalize_openai_text_tool_stream(
    stream: Any,
    body: Mapping[str, Any],
) -> Any:
    """Convert exact textual tool envelopes into ordinary OpenAI tool deltas.

    This recognizer is enabled only when the request declared tools and only for
    the complete ``<tool_call><function=...>`` protocol signature. Ordinary text
    remains untouched, while structured upstream tool calls stay authoritative.
    """
    schemas = openai_tool_schemas(body)
    if not schemas:
        return stream

    async def _iter() -> AsyncIterator[Any]:
        content_framer = _TextToolFramer()
        reasoning_framer = _TextToolFramer()
        structured_tool_calls_seen = False
        finalized = False

        async for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                yield chunk
                continue

            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                yield chunk
                continue

            structured_calls = getattr(delta, "tool_calls", None)
            if _has_structured_tool_calls(structured_calls):
                structured_tool_calls_seen = True

            content = getattr(delta, "content", None)
            safe_content = (
                content_framer.feed(content) if isinstance(content, str) else content
            )
            reasoning = getattr(delta, "reasoning_content", None)
            safe_reasoning = (
                reasoning_framer.feed(reasoning)
                if isinstance(reasoning, str)
                else reasoning
            )

            finish_reason = getattr(choice, "finish_reason", None)
            generated_calls: tuple[_TextToolCall, ...] = ()
            if finish_reason is not None:
                safe_content, safe_reasoning, generated_calls = _finish_framers(
                    content_framer,
                    reasoning_framer,
                    content=safe_content,
                    reasoning_content=safe_reasoning,
                    schemas=schemas,
                    structured_tool_calls_seen=structured_tool_calls_seen,
                )
                finalized = True

            for normalized_chunk in _normalized_chunks(
                chunk,
                choice,
                delta,
                content=safe_content,
                reasoning_content=safe_reasoning,
                generated_calls=generated_calls,
                terminal=finish_reason is not None,
            ):
                yield normalized_chunk

        if not finalized:
            content, reasoning, generated_calls = _finish_framers(
                content_framer,
                reasoning_framer,
                content=None,
                reasoning_content=None,
                schemas=schemas,
                structured_tool_calls_seen=structured_tool_calls_seen,
            )
            if content or reasoning or generated_calls:
                for synthetic_chunk in _synthetic_chunks(
                    content=content,
                    reasoning_content=reasoning,
                    generated_calls=generated_calls,
                ):
                    yield synthetic_chunk

    return _NormalizedTextToolStream(_iter(), stream)


def _finish_framers(
    content_framer: _TextToolFramer,
    reasoning_framer: _TextToolFramer,
    *,
    content: Any,
    reasoning_content: Any,
    schemas: Mapping[str, dict[str, Any]],
    structured_tool_calls_seen: bool,
) -> tuple[Any, Any, tuple[_TextToolCall, ...]]:
    content = _join_optional_text(content, content_framer.finish())
    reasoning_content = _join_optional_text(
        reasoning_content,
        reasoning_framer.finish(),
    )
    if structured_tool_calls_seen:
        return content, reasoning_content, ()

    controls = tuple(
        control
        for control in (content_framer.control, reasoning_framer.control)
        if control is not None
    )
    if len(controls) > 1:
        raise OpenAITextToolProtocolError(
            "OpenAI-compatible provider returned textual tool calls in multiple "
            "output fields."
        )
    calls = _parse_control_sequence(controls[0], schemas) if controls else ()
    return content, reasoning_content, calls


def _parse_control_sequence(
    text: str,
    schemas: Mapping[str, dict[str, Any]],
) -> tuple[_TextToolCall, ...]:
    cursor = 0
    calls: list[_TextToolCall] = []
    while True:
        cursor = _skip_whitespace(text, cursor)
        if cursor == len(text):
            break
        if not text.startswith(_TOOL_BLOCK_START, cursor):
            if calls:
                raise OpenAITextToolProtocolError(
                    "OpenAI-compatible provider returned content after textual "
                    "tool markup."
                )
            raise OpenAITextToolProtocolError(
                "OpenAI-compatible provider returned malformed textual tool markup."
            )

        block_start = cursor + len(_TOOL_BLOCK_START)
        block_end = text.find(_TOOL_BLOCK_END, block_start)
        if block_end < 0:
            raise OpenAITextToolProtocolError(
                "OpenAI-compatible provider returned an incomplete textual tool call."
            )
        name, arguments = _parse_tool_block(text[block_start:block_end], schemas)
        calls.append(_TextToolCall(index=len(calls), name=name, arguments=arguments))
        cursor = block_end + len(_TOOL_BLOCK_END)

    if not calls:
        raise OpenAITextToolProtocolError(
            "OpenAI-compatible provider returned empty textual tool markup."
        )
    return tuple(calls)


def _parse_tool_block(
    block: str,
    schemas: Mapping[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    cursor = _skip_whitespace(block, 0)
    if not block.startswith(_FUNCTION_START, cursor):
        raise OpenAITextToolProtocolError(
            "OpenAI-compatible provider returned malformed textual function markup."
        )
    name_end = block.find(">", cursor + len(_FUNCTION_START))
    if name_end < 0:
        raise OpenAITextToolProtocolError(
            "OpenAI-compatible provider returned an incomplete textual function tag."
        )
    name = block[cursor + len(_FUNCTION_START) : name_end]
    if not _valid_tag_name(name):
        raise OpenAITextToolProtocolError(
            "OpenAI-compatible provider returned an invalid textual tool name."
        )
    schema = schemas.get(name)
    if schema is None:
        raise OpenAITextToolProtocolError(
            "OpenAI-compatible provider returned a call for an undeclared tool."
        )

    properties = schema.get("properties")
    property_schemas = properties if isinstance(properties, Mapping) else {}
    arguments: dict[str, Any] = {}
    cursor = name_end + 1
    while True:
        cursor = _skip_whitespace(block, cursor)
        if block.startswith(_FUNCTION_END, cursor):
            cursor += len(_FUNCTION_END)
            break
        if cursor == len(block):
            raise OpenAITextToolProtocolError(
                "OpenAI-compatible provider returned an incomplete textual function."
            )
        if not block.startswith(_PARAMETER_START, cursor):
            raise OpenAITextToolProtocolError(
                "OpenAI-compatible provider returned malformed textual parameters."
            )

        parameter_name_end = block.find(">", cursor + len(_PARAMETER_START))
        if parameter_name_end < 0:
            raise OpenAITextToolProtocolError(
                "OpenAI-compatible provider returned an incomplete parameter tag."
            )
        parameter_name = block[cursor + len(_PARAMETER_START) : parameter_name_end]
        if not _valid_tag_name(parameter_name):
            raise OpenAITextToolProtocolError(
                "OpenAI-compatible provider returned an invalid parameter name."
            )
        if parameter_name in arguments:
            raise OpenAITextToolProtocolError(
                "OpenAI-compatible provider returned a duplicate textual parameter."
            )

        value_start = parameter_name_end + 1
        value_end = block.find(_PARAMETER_END, value_start)
        if value_end < 0:
            raise OpenAITextToolProtocolError(
                "OpenAI-compatible provider returned an incomplete parameter value."
            )
        value = _unwrap_template_newlines(block[value_start:value_end])
        parameter_schema = property_schemas.get(parameter_name)
        if not isinstance(parameter_schema, Mapping):
            parameter_schema = {}
        try:
            arguments[parameter_name] = coerce_text_argument(
                value,
                parameter_schema,
            )
        except ValueError:
            raise OpenAITextToolProtocolError(
                "OpenAI-compatible provider returned a textual parameter with the "
                "wrong type."
            ) from None
        cursor = value_end + len(_PARAMETER_END)

    if block[cursor:].strip():
        raise OpenAITextToolProtocolError(
            "OpenAI-compatible provider returned content after a textual function."
        )
    if not arguments_match_schema(arguments, schema):
        raise OpenAITextToolProtocolError(
            f"OpenAI-compatible provider returned schema-invalid arguments for "
            f"tool {name!r}."
        )
    return name, arguments


def _normalized_chunks(
    source_chunk: Any,
    source_choice: Any,
    source_delta: Any,
    *,
    content: Any,
    reasoning_content: Any,
    generated_calls: tuple[_TextToolCall, ...],
    terminal: bool,
) -> list[Any]:
    source_content = getattr(source_delta, "content", None)
    source_reasoning = getattr(source_delta, "reasoning_content", None)
    structured_calls = getattr(source_delta, "tool_calls", None)
    source_finish_reason = getattr(source_choice, "finish_reason", None)
    source_usage = getattr(source_chunk, "usage", None)

    if (
        not generated_calls
        and content == source_content
        and reasoning_content == source_reasoning
    ):
        return [source_chunk]

    chunks: list[Any] = []
    has_source_payload = (
        bool(content)
        or bool(reasoning_content)
        or _has_structured_tool_calls(structured_calls)
    )
    if has_source_payload or not generated_calls:
        chunks.append(
            _chunk(
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=structured_calls,
            )
        )
    chunks.extend(_tool_call_chunk(call) for call in generated_calls)

    if not chunks:
        chunks.append(_chunk())
    if terminal:
        chunks[-1].choices[0].finish_reason = source_finish_reason
        chunks[-1].usage = source_usage
    return chunks


def _synthetic_chunks(
    *,
    content: str | None,
    reasoning_content: str | None,
    generated_calls: tuple[_TextToolCall, ...],
) -> list[Any]:
    chunks = (
        [_chunk(content=content, reasoning_content=reasoning_content)]
        if content or reasoning_content
        else []
    )
    chunks.extend(_tool_call_chunk(call) for call in generated_calls)
    return chunks


def _chunk(
    *,
    content: Any = None,
    reasoning_content: Any = None,
    tool_calls: Any = None,
) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                ),
                finish_reason=None,
            )
        ],
        usage=None,
    )


def _tool_call_chunk(call: _TextToolCall) -> Any:
    tool_call = SimpleNamespace(
        index=call.index,
        id=f"call_text_native_{uuid.uuid4().hex}",
        function=SimpleNamespace(
            name=call.name,
            arguments=json.dumps(
                call.arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ),
    )
    return _chunk(tool_calls=[tool_call])


def _unwrap_template_newlines(value: str) -> str:
    if value.startswith("\r\n"):
        value = value[2:]
    elif value.startswith("\n"):
        value = value[1:]
    if value.endswith("\r\n"):
        return value[:-2]
    if value.endswith("\n"):
        return value[:-1]
    return value


def _valid_tag_name(value: str) -> bool:
    return bool(value) and not any(
        character.isspace() or character in "<>" for character in value
    )


def _skip_whitespace(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor


def _partial_marker_suffix_length(text: str, marker: str) -> int:
    max_length = min(len(text), len(marker) - 1)
    for length in range(max_length, 0, -1):
        if marker.startswith(text[-length:]):
            return length
    return 0


def _join_optional_text(first: Any, second: str) -> Any:
    if not second:
        return first
    if isinstance(first, str):
        return "".join((first, second))
    return second


def _has_structured_tool_calls(value: Any) -> bool:
    return isinstance(value, list | tuple) and bool(value)
