"""SSE event builder for Anthropic-format streaming responses."""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

try:
    import tiktoken

    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODER = None


# Standard headers for Anthropic-style ``text/event-stream`` responses from this proxy.
ANTHROPIC_SSE_RESPONSE_HEADERS: dict[str, str] = {
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}

STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def map_stop_reason(openai_reason: str | None) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    return (
        STOP_REASON_MAP.get(openai_reason, "end_turn") if openai_reason else "end_turn"
    )


def _safe_usage_int(value: object) -> int:
    """Coerce streamed usage counters to int; non-integers become 0."""
    return value if isinstance(value, int) else 0


def format_sse_event(event_type: str, data: dict) -> str:
    """Format one Anthropic-style SSE event (no logging)."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@dataclass
class ToolCallState:
    """State for a single streaming tool call."""

    block_index: int
    tool_id: str
    name: str
    extra_content: dict[str, Any] | None = None
    contents: list[str] = field(default_factory=list)
    started: bool = False
    task_arg_buffer: str = ""
    task_args_emitted: bool = False
    pre_start_args: str = ""


@dataclass
class ContentBlockManager:
    """Manage content block indices and state."""

    next_index: int = 0
    thinking_index: int = -1
    text_index: int = -1
    thinking_started: bool = False
    text_started: bool = False
    tool_states: dict[int, ToolCallState] = field(default_factory=dict)

    def allocate_index(self) -> int:
        idx = self.next_index
        self.next_index += 1
        return idx

    def ensure_tool_state(self, index: int) -> ToolCallState:
        """Create tool stream state for ``index`` when the first tool delta arrives."""
        if index not in self.tool_states:
            self.tool_states[index] = ToolCallState(block_index=-1, tool_id="", name="")
        return self.tool_states[index]

    def set_stream_tool_id(self, index: int, tool_id: str | None) -> None:
        """Record OpenAI tool call id before ``content_block_start`` (split-stream providers)."""
        if not tool_id:
            return
        state = self.ensure_tool_state(index)
        state.tool_id = str(tool_id)

    def set_tool_extra_content(
        self, index: int, extra_content: dict[str, Any] | None
    ) -> None:
        """Record provider-specific OpenAI tool-call metadata before block start."""
        if not extra_content:
            return
        state = self.ensure_tool_state(index)
        state.extra_content = extra_content

    def register_tool_name(self, index: int, name: str) -> None:
        """Record tool name fragments as they arrive from chunked OpenAI streams.

        Names may be split across deltas; later chunks can extend (``ab`` + ``c``)
        or repeat prefixes, so we merge conservatively.
        """
        if index not in self.tool_states:
            self.tool_states[index] = ToolCallState(
                block_index=-1, tool_id="", name=name
            )
            return
        state = self.tool_states[index]
        prev = state.name
        if not prev or name.startswith(prev):
            state.name = name
        elif not prev.startswith(name):
            state.name = prev + name

    def buffer_task_args(self, index: int, args: str) -> dict | None:
        state = self.tool_states.get(index)
        if state is None or state.task_args_emitted:
            return None

        state.task_arg_buffer += args
        try:
            args_json = json.loads(state.task_arg_buffer)
        except Exception:
            return None

        _normalize_task_run_in_background(args_json)

        state.task_args_emitted = True
        state.task_arg_buffer = ""
        return args_json

    def has_emitted_tool_block(self) -> bool:
        """True when native OpenAI tool streaming has started a ``tool_use`` block."""
        return any(s.started for s in self.tool_states.values())

    def flush_task_arg_buffers(self) -> list[tuple[int, str]]:
        results: list[tuple[int, str]] = []
        for tool_index, state in list(self.tool_states.items()):
            if not state.task_arg_buffer or state.task_args_emitted:
                continue

            out = "{}"
            try:
                args_json = json.loads(state.task_arg_buffer)
                _normalize_task_run_in_background(args_json)
                out = json.dumps(args_json)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                digest = hashlib.sha256(
                    state.task_arg_buffer.encode("utf-8", errors="replace")
                ).hexdigest()[:16]
                logger.warning(
                    "Task args invalid JSON (id={} len={} buffer_sha256_prefix={}): {}",
                    state.tool_id or "unknown",
                    len(state.task_arg_buffer),
                    digest,
                    e,
                )

            state.task_args_emitted = True
            state.task_arg_buffer = ""
            results.append((tool_index, out))
        return results


def _normalize_task_run_in_background(args_json: dict) -> None:
    """Force Claude Code Task subagents to run in foreground (single shared rule)."""
    if args_json.get("run_in_background") is not False:
        args_json["run_in_background"] = False


def _strip_markdown_fences(text: str) -> tuple[str, int]:
    """Strip markdown code fences from tool argument text.

    Some models (especially NIM with reasoning) wrap tool JSON in
    `````json ... ````` fences. This strips them so JSON parsing/rescue
    can work on the raw JSON inside.

    Returns (cleaned_text, prefix_length) where prefix_length is the
    byte offset of the JSON start within the original text.
    """
    stripped = text.strip()
    prefix_len = len(text) - len(text.lstrip())

    if stripped.startswith("```json"):
        prefix_len += 7
        stripped = stripped[7:]
    elif stripped.startswith("```"):
        prefix_len += 3
        stripped = stripped[3:]

    stripped = stripped.lstrip("\n")
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    stripped = stripped.rstrip("\n ")

    return stripped, prefix_len


class SSEBuilder:
    """Builder for Anthropic SSE streaming events."""

    def __init__(
        self,
        message_id: str,
        model: str,
        input_tokens: int = 0,
        *,
        log_raw_events: bool = False,
    ):
        self.message_id = message_id
        self.model = model
        self.input_tokens = input_tokens
        self._log_raw_events = log_raw_events
        self.blocks = ContentBlockManager()
        self._accumulated_text_parts: list[str] = []
        self._accumulated_reasoning_parts: list[str] = []

    def _format_event(self, event_type: str, data: dict) -> str:
        event_str = format_sse_event(event_type, data)
        if self._log_raw_events:
            logger.debug("SSE_EVENT: {} - {}", event_type, event_str.strip())
        return event_str

    def message_start(self) -> str:
        safe_input = _safe_usage_int(self.input_tokens)
        usage = {"input_tokens": safe_input, "output_tokens": 1}
        return self._format_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self.message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": usage,
                },
            },
        )

    def message_delta(self, stop_reason: str, output_tokens: int | None) -> str:
        safe_in = _safe_usage_int(self.input_tokens)
        safe_out = output_tokens if isinstance(output_tokens, int) else 0
        return self._format_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {
                    "input_tokens": safe_in,
                    "output_tokens": safe_out,
                },
            },
        )

    def message_stop(self) -> str:
        return self._format_event("message_stop", {"type": "message_stop"})

    def content_block_start(self, index: int, block_type: str, **kwargs) -> str:
        content_block: dict = {"type": block_type}
        if block_type == "thinking":
            content_block["thinking"] = kwargs.get("thinking", "")
        elif block_type == "text":
            content_block["text"] = kwargs.get("text", "")
        elif block_type == "tool_use":
            content_block["id"] = kwargs.get("id", "")
            content_block["name"] = kwargs.get("name", "")
            content_block["input"] = kwargs.get("input", {})
            extra_content = kwargs.get("extra_content")
            if isinstance(extra_content, dict) and extra_content:
                content_block["extra_content"] = extra_content

        return self._format_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": content_block,
            },
        )

    def content_block_delta(self, index: int, delta_type: str, content: str) -> str:
        delta: dict = {"type": delta_type}
        if delta_type == "thinking_delta":
            delta["thinking"] = content
        elif delta_type == "text_delta":
            delta["text"] = content
        elif delta_type == "input_json_delta":
            delta["partial_json"] = content

        return self._format_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": delta,
            },
        )

    def content_block_stop(self, index: int) -> str:
        return self._format_event(
            "content_block_stop",
            {
                "type": "content_block_stop",
                "index": index,
            },
        )

    def start_thinking_block(self) -> str:
        self.blocks.thinking_index = self.blocks.allocate_index()
        self.blocks.thinking_started = True
        return self.content_block_start(self.blocks.thinking_index, "thinking")

    def emit_thinking_delta(self, content: str) -> str:
        self._accumulated_reasoning_parts.append(content)
        return self.content_block_delta(
            self.blocks.thinking_index, "thinking_delta", content
        )

    def stop_thinking_block(self) -> str:
        self.blocks.thinking_started = False
        return self.content_block_stop(self.blocks.thinking_index)

    def start_text_block(self) -> str:
        self.blocks.text_index = self.blocks.allocate_index()
        self.blocks.text_started = True
        return self.content_block_start(self.blocks.text_index, "text")

    def emit_text_delta(self, content: str) -> str:
        self._accumulated_text_parts.append(content)
        return self.content_block_delta(self.blocks.text_index, "text_delta", content)

    def stop_text_block(self) -> str:
        self.blocks.text_started = False
        return self.content_block_stop(self.blocks.text_index)

    def start_tool_block(
        self,
        tool_index: int,
        tool_id: str,
        name: str,
        *,
        extra_content: dict[str, Any] | None = None,
    ) -> str:
        if not tool_id or not name:
            logger.warning(
                "BLOCKED_START_TOOL_BLOCK: empty tool_id={} or name={} at index={} — skipping",
                tool_id,
                name,
                tool_index,
            )
            return ""
        block_idx = self.blocks.allocate_index()
        if tool_index in self.blocks.tool_states:
            state = self.blocks.tool_states[tool_index]
            state.block_index = block_idx
            state.tool_id = tool_id
            if extra_content:
                state.extra_content = extra_content
            state.started = True
        else:
            self.blocks.tool_states[tool_index] = ToolCallState(
                block_index=block_idx,
                tool_id=tool_id,
                name=name,
                extra_content=extra_content,
                started=True,
            )
        return self.content_block_start(
            block_idx,
            "tool_use",
            id=tool_id,
            name=name,
            extra_content=extra_content,
        )

    def emit_tool_delta(self, tool_index: int, partial_json: str) -> str:
        state = self.blocks.tool_states[tool_index]
        state.contents.append(partial_json)
        return self.content_block_delta(
            state.block_index, "input_json_delta", partial_json
        )

    def stop_tool_block(self, tool_index: int) -> str:
        block_idx = self.blocks.tool_states[tool_index].block_index
        return self.content_block_stop(block_idx)

    def ensure_thinking_block(self) -> Iterator[str]:
        if self.blocks.text_started:
            yield self.stop_text_block()
        if not self.blocks.thinking_started:
            yield self.start_thinking_block()

    def ensure_text_block(self) -> Iterator[str]:
        if self.blocks.thinking_started:
            yield self.stop_thinking_block()
        if not self.blocks.text_started:
            yield self.start_text_block()

    def close_content_blocks(self) -> Iterator[str]:
        if self.blocks.thinking_started:
            yield self.stop_thinking_block()
        if self.blocks.text_started:
            yield self.stop_text_block()

    def close_incomplete_tool_blocks(self) -> Iterator[str]:
        """Emit completing deltas for tool blocks with invalid accumulated JSON.

        When a transport error interrupts a stream mid-tool-call, the
        input_json_delta fragments already sent may concatenate to invalid
        JSON. Claude Code then fails with "tool call could not be parsed".

        Must be called **before** close_all_blocks so the rescue delta
        arrives before content_block_stop.
        """
        for _tool_index, state in list(self.blocks.tool_states.items()):
            if not state.started:
                continue
            if not state.contents:
                logger.warning(
                    "TOOL_BLOCK_NO_CONTENTS: tool_index={} name={} tool_id={} — "
                    "tool block started but has zero input_json_delta events",
                    _tool_index,
                    state.name or "unknown",
                    state.tool_id or "unknown",
                )
                continue
            concatenated = "".join(state.contents)
            if not concatenated.strip():
                continue
            cleaned, _prefix_len = _strip_markdown_fences(concatenated)
            try:
                json.loads(cleaned)
                continue
            except (json.JSONDecodeError, ValueError):
                pass
            rescue = self._rescue_partial_json(cleaned)
            repaired = cleaned + rescue
            try:
                json.loads(repaired)
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "TOOL_RESCUE_FAILED: tool_id={} len={} — rescue '{}' "
                    "did not produce valid JSON",
                    state.tool_id or "unknown",
                    len(cleaned),
                    rescue[:80],
                )
            else:
                logger.debug(
                    "TOOL_RESCUE_OK: tool_id={} name={} len={}→{}",
                    state.tool_id or "unknown",
                    state.name or "unknown",
                    len(concatenated),
                    len(repaired),
                )
            state.contents = [repaired]
            yield self.content_block_delta(
                state.block_index, "input_json_delta", rescue
            )

    def validate_tool_blocks(self, valid_tool_names: frozenset[str]) -> Iterator[str]:
        """Replace arguments of tool blocks with unrecognized names so the client can parse them.

        When a model hallucinates a tool name not in the request's tool list,
        Claude Code fails with "tool call could not be parsed". This method
        detects such blocks and ensures the accumulated input_json_delta
        fragments form parseable JSON so the client doesn't crash.
        """
        if not valid_tool_names:
            return
        for tool_index, state in list(self.blocks.tool_states.items()):
            if not state.started:
                continue
            tool_name = (state.name or "").strip()
            if not tool_name or tool_name in valid_tool_names:
                continue
            concatenated = "".join(state.contents)
            if concatenated.strip():
                rescue_suffix = self._rescue_partial_json(concatenated)
                logger.warning(
                    "INVALID_TOOL_RESCUE: tool_id={} name={} not in valid set — "
                    "rescued partial args (original_len={} suffix_len={})",
                    state.tool_id or "unknown",
                    tool_name,
                    len(concatenated),
                    len(rescue_suffix),
                )
                try:
                    total = concatenated + rescue_suffix
                    json.loads(total)
                    state.contents = [total]
                except (json.JSONDecodeError, ValueError):
                    state.contents = ['{"_invalid_tool": true}']
                yield self.content_block_delta(
                    state.block_index,
                    "input_json_delta",
                    rescue_suffix,
                )
            else:
                fallback = json.dumps({"_invalid_tool": True, "name": tool_name})
                state.contents = [fallback]
                yield self.emit_tool_delta(tool_index, fallback)

    @staticmethod
    def _rescue_partial_json(partial: str) -> str:
        """Return a JSON fragment that when appended to *partial* yields valid JSON."""
        s = partial.strip()
        if not s:
            return '{"_interrupted": true}'

        in_string = False
        escape_next = False
        open_stack: list[str] = []
        for ch in s:
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                open_stack.append("}")
            elif ch == "[":
                open_stack.append("]")
            elif ch in ("}", "]") and open_stack:
                open_stack.pop()

        suffix = ""
        was_in_string = in_string
        if in_string:
            suffix += '"'

        candidate = s + suffix
        if open_stack and open_stack[-1] == "}":
            stripped = candidate.rstrip()
            if stripped.endswith(":"):
                suffix += " null"
            elif stripped.endswith(","):
                suffix += '"_truncated": null'
            elif was_in_string and stripped.endswith('"'):
                before = s[: s.rfind('"')].rstrip()
                if before.endswith(",") or before.endswith("{"):
                    suffix += ": null"

        for bracket in reversed(open_stack):
            suffix += bracket

        candidate = s + suffix
        try:
            json.loads(candidate)
            return suffix
        except (json.JSONDecodeError, ValueError):
            pass

        close_brackets = "".join(reversed(open_stack))
        if was_in_string:
            before = s[: s.rfind('"')].rstrip()
            if before.endswith(",") or before.endswith("{"):
                candidate2 = s + '"' + ": null" + close_brackets
            else:
                candidate2 = s + '"' + " null" + close_brackets
        else:
            candidate2 = s + " null" + close_brackets
        try:
            json.loads(candidate2)
            return candidate2[len(s) :]
        except (json.JSONDecodeError, ValueError):
            pass

        truncation = ""
        if in_string:
            truncation += '"'
        truncation += ',"_interrupted": true'
        for bracket in reversed(open_stack):
            truncation += bracket

        candidate3 = s + truncation
        try:
            json.loads(candidate3)
            return truncation
        except (json.JSONDecodeError, ValueError):
            pass

        raw_opens = s.count("{") - s.count("}")
        raw_bracket_opens = s.count("[") - s.count("]")
        rescue = ""
        if in_string:
            rescue += '"'
        if raw_opens > 0:
            rescue += ',"_interrupted": true'
            rescue += "}" * raw_opens
        if raw_bracket_opens > 0:
            rescue += "]" * raw_bracket_opens
        if not rescue.strip():
            rescue = "}}"
        try:
            json.loads(s + rescue)
            return rescue
        except (json.JSONDecodeError, ValueError):
            return "}}"

    def close_all_blocks(self) -> Iterator[str]:
        yield from self.close_content_blocks()
        for tool_index, state in list(self.blocks.tool_states.items()):
            if state.started:
                yield self.stop_tool_block(tool_index)

    def emit_error(self, error_message: str) -> Iterator[str]:
        error_index = self.blocks.allocate_index()
        yield self.content_block_start(error_index, "text")
        yield self.content_block_delta(error_index, "text_delta", error_message)
        yield self.content_block_stop(error_index)

    def emit_top_level_error(self, error_message: str) -> str:
        """Emit a top-level ``event: error`` (not assistant text) for transport failures."""
        return self._format_event(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": error_message,
                },
            },
        )

    @property
    def accumulated_text(self) -> str:
        return "".join(self._accumulated_text_parts)

    @property
    def accumulated_reasoning(self) -> str:
        return "".join(self._accumulated_reasoning_parts)

    def estimate_output_tokens(self) -> int:
        accumulated_text = self.accumulated_text
        accumulated_reasoning = self.accumulated_reasoning
        if ENCODER:
            text_tokens = len(ENCODER.encode(accumulated_text))
            reasoning_tokens = len(ENCODER.encode(accumulated_reasoning))
            tool_tokens = 0
            started_tool_count = 0
            for state in self.blocks.tool_states.values():
                tool_tokens += len(ENCODER.encode(state.name))
                tool_tokens += len(ENCODER.encode("".join(state.contents)))
                tool_tokens += 15
                if state.started:
                    started_tool_count += 1

            block_count = (
                (1 if accumulated_reasoning else 0)
                + (1 if accumulated_text else 0)
                + started_tool_count
            )
            return text_tokens + reasoning_tokens + tool_tokens + (block_count * 4)

        text_tokens = len(accumulated_text) // 4
        reasoning_tokens = len(accumulated_reasoning) // 4
        tool_tokens = (
            sum(1 for state in self.blocks.tool_states.values() if state.started) * 50
        )
        return text_tokens + reasoning_tokens + tool_tokens
