"""SSE event builder for Anthropic-format streaming responses."""

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field

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
    "content_filter": "content_filtering",
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
            signature = kwargs.get("signature")
            if signature:
                content_block["signature"] = signature
        elif block_type == "text":
            content_block["text"] = kwargs.get("text", "")
        elif block_type == "tool_use":
            content_block["id"] = kwargs.get("id", "")
            content_block["name"] = kwargs.get("name", "")
            content_block["input"] = kwargs.get("input", {})

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

    def start_tool_block(self, tool_index: int, tool_id: str, name: str) -> str:
        block_idx = self.blocks.allocate_index()
        if tool_index in self.blocks.tool_states:
            state = self.blocks.tool_states[tool_index]
            state.block_index = block_idx
            state.tool_id = tool_id
            state.started = True
        else:
            self.blocks.tool_states[tool_index] = ToolCallState(
                block_index=block_idx,
                tool_id=tool_id,
                name=name,
                started=True,
            )
        return self.content_block_start(block_idx, "tool_use", id=tool_id, name=name)

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
