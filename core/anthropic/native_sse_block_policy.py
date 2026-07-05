"""Shared native Anthropic SSE thinking policy, block remapping, and overlap repair.

Used by :class:`OpenRouterProvider` and line-mode
:class:`providers.transports.anthropic_messages.AnthropicMessagesTransport` providers.
"""

import copy
import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "NativeSseBlockPolicyState",
    "format_native_sse_event",
    "is_terminal_openrouter_done_event",
    "parse_native_sse_event",
    "transform_native_sse_block_event",
]


@dataclass
class _UpstreamBlockState:
    """Per-upstream content block: segment index and liveness in the model stream."""

    block_type: str
    down_index: int
    open: bool
    last_start_block: dict[str, Any] | None = None


@dataclass
class NativeSseBlockPolicyState:
    """Track per-upstream content blocks and remapped Anthropic ``index`` field."""

    next_index: int = 0
    by_upstream: dict[int, _UpstreamBlockState] = field(default_factory=dict)
    dropped_indexes: set[int] = field(default_factory=set)
    pending_suppressed_stops: set[int] = field(default_factory=set)
    message_stopped: bool = False


def format_native_sse_event(event_name: str | None, data_text: str) -> str:
    """Format an SSE event from its event name and data payload."""
    lines: list[str] = []
    if event_name:
        lines.append(f"event: {event_name}")
    lines.extend(f"data: {line}" for line in data_text.splitlines())
    return "\n".join(lines) + "\n\n"


def parse_native_sse_event(event: str) -> tuple[str | None, str]:
    """Extract the event name and raw data payload from an SSE event."""
    event_name = None
    data_lines: list[str] = []
    for line in event.strip().splitlines():
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    return event_name, "\n".join(data_lines)


def is_terminal_openrouter_done_event(event_name: str | None, data_text: str) -> bool:
    """Return whether an event is OpenAI-style terminal noise (``[DONE]``)."""
    return (event_name is None or event_name in {"data", "done"}) and (
        data_text.strip().upper() == "[DONE]"
    )


def _delta_type_to_block_kind(delta_type: Any) -> str | None:
    """Map a content_block_delta type to a content block kind (text/thinking/tool_use)."""
    if not isinstance(delta_type, str):
        return None
    if delta_type in {"thinking_delta", "signature_delta"}:
        return "thinking"
    if delta_type == "text_delta":
        return "text"
    if delta_type == "input_json_delta":
        return "tool_use"
    return None


def _synthetic_start_content_block(
    block_kind: str,
    *,
    upstream_index: int,
    stored_tool_block: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a `content_block` for a `content_block_start` with empty streaming fields."""
    if block_kind == "tool_use":
        if (
            isinstance(stored_tool_block, dict)
            and stored_tool_block.get("type") == "tool_use"
        ):
            tool_id = stored_tool_block.get("id")
            name = stored_tool_block.get("name")
            inp = stored_tool_block.get("input")
            return {
                "type": "tool_use",
                "id": tool_id
                if isinstance(tool_id, str) and tool_id
                else f"toolu_or_{upstream_index}",
                "name": name if isinstance(name, str) else "",
                "input": inp if isinstance(inp, dict) else {},
            }
        return {
            "type": "tool_use",
            "id": f"toolu_or_{upstream_index}",
            "name": "",
            "input": {},
        }
    if block_kind == "thinking":
        return {"type": "thinking", "thinking": ""}
    if block_kind == "text":
        return {"type": "text", "text": ""}
    return {"type": "text", "text": ""}


def _should_drop_block_type(block_type: Any, *, thinking_enabled: bool) -> bool:
    if not isinstance(block_type, str):
        return False
    if block_type.startswith("redacted_thinking"):
        return not thinking_enabled
    return not thinking_enabled and "thinking" in block_type


def _synthetic_close_other_open_blocks(
    state: NativeSseBlockPolicyState, current_upstream: int
) -> str:
    """Close every open block except `current_upstream` and track duplicate upstream stops."""
    out: list[str] = []
    for upstream, seg in list(state.by_upstream.items()):
        if upstream == current_upstream or not seg.open:
            continue
        out.append(
            format_native_sse_event(
                "content_block_stop",
                json.dumps(
                    {
                        "type": "content_block_stop",
                        "index": seg.down_index,
                    }
                ),
            )
        )
        seg.open = False
        state.pending_suppressed_stops.add(upstream)
    return "".join(out)


def _allocate_new_segment(
    state: NativeSseBlockPolicyState,
    upstream_index: int,
    block_type: str,
    *,
    last_start_block: dict[str, Any] | None = None,
) -> int:
    """Assign a new downstream `index` for a segment and record upstream state."""
    new_idx = state.next_index
    state.next_index += 1
    state.by_upstream[upstream_index] = _UpstreamBlockState(
        block_type=block_type,
        down_index=new_idx,
        open=True,
        last_start_block=last_start_block,
    )
    return new_idx


def transform_native_sse_block_event(
    event: str,
    state: NativeSseBlockPolicyState,
    *,
    thinking_enabled: bool,
) -> str | None:
    """Normalize native Anthropic SSE events and enforce local thinking policy."""
    event_name, data_text = parse_native_sse_event(event)
    if not event_name or not data_text:
        return event

    try:
        payload = json.loads(data_text)
    except json.JSONDecodeError:
        return event

    if event_name == "content_block_start":
        block = payload.get("content_block")
        if not isinstance(block, dict):
            return event
        block_type = block.get("type")
        upstream_index = payload.get("index")
        if not isinstance(upstream_index, int):
            return event
        if _should_drop_block_type(block_type, thinking_enabled=thinking_enabled):
            state.dropped_indexes.add(upstream_index)
            return None

        if not isinstance(block_type, str):
            return event
        prefix = _synthetic_close_other_open_blocks(state, upstream_index)
        stored = copy.deepcopy(block)
        new_idx = _allocate_new_segment(
            state,
            upstream_index,
            block_type=block_type,
            last_start_block=stored,
        )
        payload["index"] = new_idx
        return prefix + format_native_sse_event(event_name, json.dumps(payload))

    if event_name == "content_block_delta":
        delta = payload.get("delta")
        if not isinstance(delta, dict):
            return event
        delta_type = delta.get("type")
        upstream_index = payload.get("index")
        if not isinstance(upstream_index, int):
            return event
        if upstream_index in state.dropped_indexes:
            return None
        if _should_drop_block_type(delta_type, thinking_enabled=thinking_enabled):
            return None

        block_kind = _delta_type_to_block_kind(delta_type)
        if block_kind is None:
            return event

        seg = state.by_upstream.get(upstream_index)
        if seg and seg.open:
            payload["index"] = seg.down_index
            return format_native_sse_event(event_name, json.dumps(payload))

        if seg is not None and not seg.open:
            # More deltas for an upstream block after a synthetic (or other) close:
            # reopen with a new downstream `index` and emit a synthetic `content_block_start` first.
            state.pending_suppressed_stops.discard(upstream_index)
            carry = seg.last_start_block
            new_idx = _allocate_new_segment(
                state,
                upstream_index,
                block_type=block_kind,
                last_start_block=carry,
            )
            stored_tool = (
                carry
                if isinstance(carry, dict) and carry.get("type") == "tool_use"
                else None
            )
            start_payload = {
                "type": "content_block_start",
                "index": new_idx,
                "content_block": _synthetic_start_content_block(
                    block_kind,
                    upstream_index=upstream_index,
                    stored_tool_block=stored_tool,
                ),
            }
            prefix = format_native_sse_event(
                "content_block_start", json.dumps(start_payload)
            )
            payload["index"] = new_idx
            return prefix + format_native_sse_event(event_name, json.dumps(payload))

        # Delta with no prior `content_block_start` in this stream
        if block_kind in ("text", "tool_use"):
            synthetic_block = _synthetic_start_content_block(
                block_kind,
                upstream_index=upstream_index,
            )
            new_idx = _allocate_new_segment(
                state,
                upstream_index,
                block_type=block_kind,
                last_start_block=copy.deepcopy(synthetic_block),
            )
            start_payload = {
                "type": "content_block_start",
                "index": new_idx,
                "content_block": synthetic_block,
            }
            prefix = format_native_sse_event(
                "content_block_start", json.dumps(start_payload)
            )
            payload["index"] = new_idx
            return prefix + format_native_sse_event(event_name, json.dumps(payload))
        # thinking: pass through raw (unusual upstream shape)
        return event

    if event_name == "content_block_stop":
        upstream_index = payload.get("index")
        if not isinstance(upstream_index, int):
            return event
        if upstream_index in state.dropped_indexes:
            return None
        if upstream_index in state.pending_suppressed_stops:
            state.pending_suppressed_stops.discard(upstream_index)
            return None

        seg = state.by_upstream.get(upstream_index)
        if seg is not None and seg.open:
            payload["index"] = seg.down_index
            seg.open = False
            return format_native_sse_event(event_name, json.dumps(payload))
        if seg is not None:
            # Spurious or duplicate `content_block_stop` for a closed block.
            return None
        if not thinking_enabled:
            return None
        return event

    return event
