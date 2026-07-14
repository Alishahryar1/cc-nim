"""Message and tool format converters."""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .content import get_block_attr, get_block_type
from .models import MessagesRequest
from .request_serialization import serialize_tool_result_content
from .utils import set_if_not_none


class OpenAIConversionError(Exception):
    """Raised when Anthropic content cannot be converted to OpenAI chat without data loss."""


class ReasoningReplayMode(StrEnum):
    """How assistant reasoning history is replayed to OpenAI-compatible providers."""

    DISABLED = "disabled"
    THINK_TAGS = "think_tags"
    REASONING_CONTENT = "reasoning_content"
    REASONING = "reasoning"


def _reasoning_replay_field(mode: ReasoningReplayMode) -> str | None:
    if mode in (
        ReasoningReplayMode.REASONING_CONTENT,
        ReasoningReplayMode.REASONING,
    ):
        return mode.value
    return None


def _set_replayed_reasoning(
    message: dict[str, Any], reasoning: str, mode: ReasoningReplayMode
) -> None:
    field_name = _reasoning_replay_field(mode)
    if field_name is not None:
        message[field_name] = reasoning


def _openai_reject_native_only_top_level_fields(
    request_data: MessagesRequest,
) -> None:
    """OpenAI chat providers may only convert known top-level request fields.

    First-class model fields (e.g. ``context_management``) are not forwarded to
    the OpenAI API but are allowed so clients do not hit spurious 400s.
    Unknown extra keys (``__pydantic_extra__``) are still rejected.
    """
    extra = request_data.model_extra
    if not extra:
        return
    raise OpenAIConversionError(
        "OpenAI chat conversion does not support these top-level request fields: "
        f"{sorted(str(k) for k in extra)}. Remove the unsupported fields."
    )


def _tool_input_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "input_schema", None)
    if isinstance(schema, dict):
        return schema
    return {"type": "object", "properties": {}}


def _clean_reasoning_content(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value


def _think_tag_content(reasoning: str) -> str:
    return f"<think>\n{reasoning}\n</think>"


def _tool_call_from_tool_use(block: Any) -> dict[str, Any]:
    tool_input = get_block_attr(block, "input", {})
    tool_call: dict[str, Any] = {
        "id": get_block_attr(block, "id"),
        "type": "function",
        "function": {
            "name": get_block_attr(block, "name"),
            "arguments": json.dumps(tool_input)
            if isinstance(tool_input, dict)
            else str(tool_input),
        },
    }
    extra_content = get_block_attr(block, "extra_content", None)
    if isinstance(extra_content, dict) and extra_content:
        tool_call["extra_content"] = deepcopy(extra_content)
    return tool_call


@dataclass
class _PlainSegment:
    messages: list[dict[str, Any]]


@dataclass
class _ToolTurnSegment:
    assistant_message: dict[str, Any]
    required_tool_ids: list[str]
    deferred_blocks: list[Any] = field(default_factory=list)
    top_level_reasoning: str | None = None
    reasoning_replay: ReasoningReplayMode = ReasoningReplayMode.THINK_TAGS
    assistant_emitted: bool = False


_TranscriptSegment = _PlainSegment | _ToolTurnSegment


def _tool_call_ids(tool_calls: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for tool_call in tool_calls:
        tool_id = tool_call.get("id")
        if tool_id is not None and str(tool_id).strip() != "":
            ids.append(str(tool_id))
    return ids


def _index_first_tool_use(blocks: list[Any]) -> int | None:
    for i, block in enumerate(blocks):
        if get_block_type(block) == "tool_use":
            return i
    return None


def _iter_tool_uses_in_order(blocks: list[Any]) -> list[dict[str, Any]]:
    return [
        _tool_call_from_tool_use(block)
        for block in blocks
        if get_block_type(block) == "tool_use"
    ]


def _deferred_post_tool_blocks(
    content: list[Any], *, first_tool_index: int
) -> list[Any]:
    return [
        b
        for i, b in enumerate(content)
        if i > first_tool_index and get_block_type(b) != "tool_use"
    ]


def _assert_no_forbidden_assistant_block(block: Any) -> None:
    block_type = get_block_type(block)
    if block_type == "image":
        raise OpenAIConversionError(
            "Assistant image blocks are not supported for OpenAI chat conversion."
        )
    if block_type in (
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
    ):
        raise OpenAIConversionError(
            "OpenAI chat conversion does not support Anthropic server tool blocks "
            f"({block_type!r} in an assistant message). Remove the unsupported block."
        )


def _openai_user_image_part(block: Any) -> dict[str, Any]:
    """Convert one Anthropic user image block without performing I/O."""
    source = get_block_attr(block, "source", {})
    source_type = get_block_attr(source, "type")

    if source_type == "base64":
        media_type = get_block_attr(source, "media_type")
        if not isinstance(media_type, str) or not media_type.strip():
            raise OpenAIConversionError(
                "Base64 image source requires a non-empty media_type."
            )
        data = get_block_attr(source, "data")
        if not isinstance(data, str) or not data.strip():
            raise OpenAIConversionError("Base64 image source requires non-empty data.")
        url = f"data:{media_type};base64,{data}"
    elif source_type == "url":
        url = get_block_attr(source, "url")
        if not isinstance(url, str) or not url.strip():
            raise OpenAIConversionError("URL image source requires a non-empty url.")
    else:
        raise OpenAIConversionError(
            f"Unsupported image source type {source_type!r}; expected 'base64' or 'url'."
        )

    return {"type": "image_url", "image_url": {"url": url}}


class _OpenAIChatHistoryLedger:
    """Assemble OpenAI chat history while respecting tool-result dependencies."""

    def __init__(self) -> None:
        self._output: list[dict[str, Any]] = []
        self._segments: list[_TranscriptSegment] = []
        self._tool_results: dict[str, dict[str, Any]] = {}

    def add_plain(self, messages: list[dict[str, Any]]) -> None:
        if messages:
            self._segments.append(_PlainSegment(messages))
            self._drain_ready_segments()

    def add_tool_turn(self, segment: _ToolTurnSegment) -> None:
        self._segments.append(segment)
        self._drain_ready_segments()

    def add_user_blocks(self, blocks: list[Any]) -> None:
        text_blocks: list[Any] = []
        for block in blocks:
            block_type = get_block_type(block)
            if block_type == "tool_result":
                self._add_text_blocks(text_blocks)
                self._record_tool_result(block)
            else:
                text_blocks.append(block)
        self._add_text_blocks(text_blocks)
        self._drain_ready_segments()

    def finish(self) -> list[dict[str, Any]]:
        self._drain_ready_segments()
        missing = self._missing_required_tool_ids()
        if missing:
            raise OpenAIConversionError(
                "OpenAI chat conversion cannot replay incomplete tool history; "
                f"missing tool_result blocks for tool_use ids: {missing}"
            )
        while self._segments:
            segment = self._segments.pop(0)
            if isinstance(segment, _PlainSegment):
                self._output.extend(segment.messages)
                continue
            self._emit_tool_turn(segment)
        return self._output

    def _add_text_blocks(self, blocks: list[Any]) -> None:
        if not blocks:
            return
        self.add_plain(AnthropicToOpenAIConverter._convert_user_message(blocks))
        blocks.clear()

    def _record_tool_result(self, block: Any) -> None:
        tuid = get_block_attr(block, "tool_use_id")
        tuid_s = str(tuid) if tuid is not None else ""
        if not tuid_s:
            self.add_plain(AnthropicToOpenAIConverter._convert_user_message([block]))
            return
        tool_content = get_block_attr(block, "content", "")
        serialized = serialize_tool_result_content(tool_content)
        tool_message = {
            "role": "tool",
            "tool_call_id": tuid,
            "content": serialized if serialized else "",
        }
        if self._has_pending_tool_id(tuid_s):
            self._tool_results[tuid_s] = tool_message
        else:
            self.add_plain([tool_message])

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

            missing = [
                tool_id
                for tool_id in segment.required_tool_ids
                if tool_id not in self._tool_results
            ]
            if missing:
                break

            self._segments.pop(0)
            for tool_id in segment.required_tool_ids:
                self._output.append(self._tool_results.pop(tool_id))
            deferred_messages = (
                AnthropicToOpenAIConverter._deferred_post_tool_to_messages(segment)
            )
            self._output.extend(deferred_messages)

    def _emit_tool_turn(self, segment: _ToolTurnSegment) -> None:
        if not segment.assistant_emitted:
            self._output.append(segment.assistant_message)
            segment.assistant_emitted = True
        for tool_id in segment.required_tool_ids:
            tool_result = self._tool_results.pop(tool_id, None)
            if tool_result is not None:
                self._output.append(tool_result)
        self._output.extend(
            AnthropicToOpenAIConverter._deferred_post_tool_to_messages(segment)
        )

    def _missing_required_tool_ids(self) -> list[str]:
        missing: list[str] = []
        for segment in self._segments:
            if not isinstance(segment, _ToolTurnSegment):
                continue
            missing.extend(
                tool_id
                for tool_id in segment.required_tool_ids
                if tool_id not in self._tool_results
            )
        return missing

    def _has_pending_tool_id(self, tool_id: str) -> bool:
        return any(
            isinstance(segment, _ToolTurnSegment)
            and tool_id in segment.required_tool_ids
            for segment in self._segments
        )


class AnthropicToOpenAIConverter:
    """Convert Anthropic message format to OpenAI-compatible format."""

    @staticmethod
    def convert_messages(
        messages: list[Any],
        *,
        reasoning_replay: ReasoningReplayMode = ReasoningReplayMode.THINK_TAGS,
    ) -> list[dict[str, Any]]:
        ledger = _OpenAIChatHistoryLedger()

        for msg in messages:
            role = msg.role
            content = msg.content
            reasoning_content = _clean_reasoning_content(
                getattr(msg, "reasoning_content", None)
            )

            if role == "user" and isinstance(content, list):
                ledger.add_user_blocks(content)
                continue

            segments = AnthropicToOpenAIConverter._convert_message_to_segments(
                role,
                content,
                reasoning_content=reasoning_content,
                reasoning_replay=reasoning_replay,
            )
            for segment in segments:
                if isinstance(segment, _PlainSegment):
                    ledger.add_plain(segment.messages)
                else:
                    ledger.add_tool_turn(segment)

        return ledger.finish()

    @staticmethod
    def _convert_message_to_segments(
        role: str,
        content: Any,
        *,
        reasoning_content: str | None,
        reasoning_replay: ReasoningReplayMode,
    ) -> list[_TranscriptSegment]:
        if role == "assistant" and isinstance(content, list):
            if (first_i := _index_first_tool_use(content)) is not None:
                for block in content:
                    if get_block_type(block) == "tool_use":
                        continue
                    _assert_no_forbidden_assistant_block(block)
                return [
                    AnthropicToOpenAIConverter._convert_assistant_message_with_split(
                        content,
                        first_tool_index=first_i,
                        reasoning_content=reasoning_content,
                        reasoning_replay=reasoning_replay,
                    )
                ]
            for block in content:
                _assert_no_forbidden_assistant_block(block)
            return [
                _PlainSegment(
                    AnthropicToOpenAIConverter._convert_assistant_message(
                        content,
                        reasoning_content=reasoning_content,
                        reasoning_replay=reasoning_replay,
                    )
                )
            ]
        if role == "user" and isinstance(content, list):
            return [
                _PlainSegment(AnthropicToOpenAIConverter._convert_user_message(content))
            ]
        if isinstance(content, str):
            converted = {"role": role, "content": content}
            if role == "assistant" and reasoning_content is not None:
                if _reasoning_replay_field(reasoning_replay) is not None:
                    _set_replayed_reasoning(
                        converted, reasoning_content, reasoning_replay
                    )
                elif (
                    reasoning_replay == ReasoningReplayMode.THINK_TAGS
                    and reasoning_content
                ):
                    content_parts = [_think_tag_content(reasoning_content)]
                    if content:
                        content_parts.append(content)
                    converted["content"] = "\n\n".join(content_parts)
            return [_PlainSegment([converted])]
        if isinstance(content, list):
            return []
        return [_PlainSegment([{"role": role, "content": str(content)}])]

    @staticmethod
    def _convert_assistant_message_with_split(
        content: list[Any],
        *,
        first_tool_index: int,
        reasoning_content: str | None,
        reasoning_replay: ReasoningReplayMode,
    ) -> _ToolTurnSegment:
        pre = content[:first_tool_index]
        tool_calls = _iter_tool_uses_in_order(content)
        if not tool_calls:
            return _ToolTurnSegment(
                assistant_message=AnthropicToOpenAIConverter._convert_assistant_message(
                    content,
                    reasoning_content=reasoning_content,
                    reasoning_replay=reasoning_replay,
                )[0],
                required_tool_ids=[],
            )
        deferred_blocks = _deferred_post_tool_blocks(
            content, first_tool_index=first_tool_index
        )

        pre_msg: dict[str, Any]
        if not pre:
            pre_msg = {
                "role": "assistant",
                "content": "",
            }
            if reasoning_content is not None:
                _set_replayed_reasoning(pre_msg, reasoning_content, reasoning_replay)
        else:
            pre_msg = AnthropicToOpenAIConverter._convert_assistant_message(
                pre,
                reasoning_content=reasoning_content,
                reasoning_replay=reasoning_replay,
            )[0]
        pre_msg["tool_calls"] = tool_calls
        if tool_calls and pre_msg.get("content") == " ":
            pre_msg["content"] = ""
        return _ToolTurnSegment(
            assistant_message=pre_msg,
            required_tool_ids=_tool_call_ids(tool_calls),
            deferred_blocks=deferred_blocks,
            top_level_reasoning=reasoning_content,
            reasoning_replay=reasoning_replay,
        )

    @staticmethod
    def _convert_assistant_message(
        content: list[Any],
        *,
        reasoning_content: str | None = None,
        reasoning_replay: ReasoningReplayMode = ReasoningReplayMode.THINK_TAGS,
    ) -> list[dict[str, Any]]:
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        thinking_seen = False
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            block_type = get_block_type(block)
            if block_type == "text":
                content_parts.append(get_block_attr(block, "text", ""))
            elif block_type == "thinking":
                if reasoning_replay == ReasoningReplayMode.DISABLED:
                    continue
                thinking = get_block_attr(block, "thinking", "")
                if reasoning_replay == ReasoningReplayMode.THINK_TAGS:
                    content_parts.append(_think_tag_content(thinking))
                elif reasoning_content is None:
                    thinking_seen = True
                    thinking_parts.append(thinking)
            elif block_type == "redacted_thinking":
                # Opaque provider continuation data; do not materialize as model-visible text
                # or native reasoning fields for OpenAI chat upstreams.
                continue
            elif block_type == "tool_use":
                tool_calls.append(_tool_call_from_tool_use(block))
            else:
                _assert_no_forbidden_assistant_block(block)

        content_str = "\n\n".join(content_parts)
        if not content_str and not tool_calls:
            content_str = " "

        msg: dict[str, Any] = {
            "role": "assistant",
            "content": content_str,
        }
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if _reasoning_replay_field(reasoning_replay) is not None:
            if reasoning_content is not None:
                _set_replayed_reasoning(msg, reasoning_content, reasoning_replay)
            elif thinking_seen:
                _set_replayed_reasoning(
                    msg, "\n".join(thinking_parts), reasoning_replay
                )

        return [msg]

    @staticmethod
    def _deferred_post_tool_to_messages(
        pending: _ToolTurnSegment,
    ) -> list[dict[str, Any]]:
        if not pending.deferred_blocks:
            return []
        return AnthropicToOpenAIConverter._convert_assistant_message(
            pending.deferred_blocks,
            reasoning_content=pending.top_level_reasoning,
            reasoning_replay=pending.reasoning_replay,
        )

    @staticmethod
    def _convert_user_message(content: list[Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        content_parts: list[dict[str, Any]] = []

        def flush_content() -> None:
            if not content_parts:
                return
            if all(part["type"] == "text" for part in content_parts):
                message_content: str | list[dict[str, Any]] = "\n".join(
                    part["text"] for part in content_parts
                )
            else:
                message_content = list(content_parts)
            result.append({"role": "user", "content": message_content})
            content_parts.clear()

        for block in content:
            block_type = get_block_type(block)

            if block_type == "text":
                content_parts.append(
                    {"type": "text", "text": get_block_attr(block, "text", "")}
                )
            elif block_type == "image":
                content_parts.append(_openai_user_image_part(block))
            elif block_type == "tool_result":
                flush_content()
                tool_content = get_block_attr(block, "content", "")
                serialized = serialize_tool_result_content(tool_content)
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": get_block_attr(block, "tool_use_id"),
                        "content": serialized if serialized else "",
                    }
                )

        flush_content()
        return result

    @staticmethod
    def convert_tools(tools: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": _tool_input_schema(tool),
                },
            }
            for tool in tools
        ]

    @staticmethod
    def convert_tool_choice(tool_choice: Any) -> Any:
        if not isinstance(tool_choice, dict):
            return tool_choice

        choice_type = tool_choice.get("type")
        if choice_type == "tool":
            name = tool_choice.get("name")
            if name:
                return {"type": "function", "function": {"name": name}}
        if choice_type == "any":
            return "required"
        if choice_type in {"auto", "none", "required"}:
            return choice_type
        if choice_type == "function" and isinstance(tool_choice.get("function"), dict):
            return tool_choice

        return tool_choice

    @staticmethod
    def convert_system_prompt(system: Any) -> dict[str, str] | None:
        if isinstance(system, str):
            return {"role": "system", "content": system}
        if isinstance(system, list):
            text_parts = [
                get_block_attr(block, "text", "")
                for block in system
                if get_block_type(block) == "text"
            ]
            if text_parts:
                return {"role": "system", "content": "\n\n".join(text_parts).strip()}
        return None


def build_base_request_body(
    request_data: MessagesRequest,
    *,
    default_max_tokens: int | None = None,
    reasoning_replay: ReasoningReplayMode = ReasoningReplayMode.THINK_TAGS,
) -> dict[str, Any]:
    """Build the common parts of an OpenAI-format request body."""
    _openai_reject_native_only_top_level_fields(request_data)
    messages = AnthropicToOpenAIConverter.convert_messages(
        request_data.messages,
        reasoning_replay=reasoning_replay,
    )

    system = request_data.system
    if system:
        system_msg = AnthropicToOpenAIConverter.convert_system_prompt(system)
        if system_msg:
            messages.insert(0, system_msg)

    body: dict[str, Any] = {"model": request_data.model, "messages": messages}

    max_tokens = request_data.max_tokens
    set_if_not_none(body, "max_tokens", max_tokens or default_max_tokens)
    set_if_not_none(body, "temperature", request_data.temperature)
    set_if_not_none(body, "top_p", request_data.top_p)

    stop_sequences = request_data.stop_sequences
    if stop_sequences:
        body["stop"] = stop_sequences

    tools = request_data.tools
    if tools:
        body["tools"] = AnthropicToOpenAIConverter.convert_tools(tools)
    tool_choice = request_data.tool_choice
    if tool_choice:
        body["tool_choice"] = AnthropicToOpenAIConverter.convert_tool_choice(
            tool_choice
        )

    return body
