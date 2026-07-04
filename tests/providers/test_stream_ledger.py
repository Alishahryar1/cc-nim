"""Tests for core.anthropic.streaming.ledger."""

from typing import Any, cast
from unittest.mock import patch

import pytest

from core.anthropic import AnthropicStreamLedger, StreamBlockLedger, map_stop_reason
from core.anthropic.stream_contracts import SSEEvent, parse_sse_text
from core.anthropic.streaming import ToolBlockState


def _parse_sse(sse_str: str) -> dict:
    """Parse an SSE event string into its data payload."""
    events = parse_sse_text(sse_str)
    if len(events) != 1:
        raise ValueError(f"expected 1 SSE event, got {len(events)} in {sse_str!r}")
    return events[0].data


class _CharEncoder:
    def encode(self, text: str) -> list[int]:
        return [0] * len(text)


class TestMapStopReason:
    """Tests for map_stop_reason function."""

    @pytest.mark.parametrize(
        "openai_reason,expected",
        [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
            ("content_filter", "end_turn"),
            (None, "end_turn"),
            ("unknown_value", "end_turn"),
            ("", "end_turn"),
        ],
        ids=[
            "stop",
            "length",
            "tool_calls",
            "content_filter",
            "none",
            "unknown",
            "empty_string",
        ],
    )
    def test_map_stop_reason(self, openai_reason, expected):
        assert map_stop_reason(openai_reason) == expected


class TestStreamBlockLedger:
    """Tests for StreamBlockLedger."""

    def test_allocate_index_increments(self):
        mgr = StreamBlockLedger()
        assert mgr.allocate_index() == 0
        assert mgr.allocate_index() == 1
        assert mgr.allocate_index() == 2

    def test_initial_state(self):
        mgr = StreamBlockLedger()
        assert mgr.thinking_index == -1
        assert mgr.text_index == -1
        assert mgr.thinking_started is False
        assert mgr.text_started is False
        assert mgr.tool_states == {}

    def test_flush_task_arg_buffers_logs_digest_not_secret(self, caplog):
        """Invalid Task JSON warnings must not echo argument prefixes (secrets)."""
        mgr = StreamBlockLedger()
        mgr.tool_states[0] = ToolBlockState(
            block_index=0, tool_id="call_x", name="Task", started=True
        )
        mgr.tool_states[
            0
        ].task_arg_buffer = (
            '{"api_key": "sk-live-super-secret-do-not-log"}not_valid_json'
        )
        with caplog.at_level("WARNING"):
            out = mgr.flush_task_arg_buffers()
        assert out == [(0, "{}")]
        text = " | ".join(r.message for r in caplog.records)
        assert "sk-live-super-secret" not in text
        assert "buffer_sha256_prefix=" in text


class TestAnthropicStreamLedgerMessageLifecycle:
    """Tests for message_start, message_delta, message_stop."""

    def test_message_start(self):
        builder = AnthropicStreamLedger("msg_123", "test-model", input_tokens=50)
        sse = builder.message_start()

        assert "event: message_start" in sse
        data = _parse_sse(sse)
        assert data["type"] == "message_start"
        msg = data["message"]
        assert msg["id"] == "msg_123"
        assert msg["model"] == "test-model"
        assert msg["role"] == "assistant"
        assert msg["content"] == []
        assert msg["usage"]["input_tokens"] == 50
        assert msg["usage"]["output_tokens"] == 1

    def test_message_delta(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        sse = builder.message_delta("end_turn", 42)

        assert "event: message_delta" in sse
        data = _parse_sse(sse)
        assert data["type"] == "message_delta"
        assert data["delta"]["stop_reason"] == "end_turn"
        assert data["usage"]["output_tokens"] == 42

    def test_message_start_coerces_non_int_input_tokens(self):
        builder = AnthropicStreamLedger("msg_1", "model", input_tokens=0)
        builder.input_tokens = cast(Any, "not_an_int")
        sse = builder.message_start()
        data = _parse_sse(sse)
        assert data["message"]["usage"]["input_tokens"] == 0
        assert data["message"]["usage"]["output_tokens"] == 1

    def test_message_delta_coerces_none_output_tokens(self):
        builder = AnthropicStreamLedger("msg_1", "model", input_tokens=3)
        sse = builder.message_delta("end_turn", None)
        data = _parse_sse(sse)
        assert data["usage"]["input_tokens"] == 3
        assert data["usage"]["output_tokens"] == 0

    def test_message_delta_preserves_zero_output_tokens(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        sse = builder.message_delta("end_turn", 0)
        data = _parse_sse(sse)
        assert data["usage"]["output_tokens"] == 0

    def test_message_stop(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        sse = builder.message_stop()

        assert "event: message_stop" in sse
        data = _parse_sse(sse)
        assert data["type"] == "message_stop"


class TestAnthropicStreamLedgerContentBlocks:
    """Tests for content block start/delta/stop events."""

    def test_content_block_start_text(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        sse = builder.content_block_start(0, "text", text="hello")

        data = _parse_sse(sse)
        assert data["type"] == "content_block_start"
        assert data["index"] == 0
        assert data["content_block"]["type"] == "text"
        assert data["content_block"]["text"] == "hello"

    def test_content_block_start_thinking(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        sse = builder.content_block_start(1, "thinking")

        data = _parse_sse(sse)
        assert data["content_block"]["type"] == "thinking"
        assert data["content_block"]["thinking"] == ""

    def test_content_block_start_tool_use(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        sse = builder.content_block_start(
            2, "tool_use", id="tool_123", name="Read", input={}
        )

        data = _parse_sse(sse)
        assert data["content_block"]["type"] == "tool_use"
        assert data["content_block"]["id"] == "tool_123"
        assert data["content_block"]["name"] == "Read"
        assert data["content_block"]["input"] == {}
        assert builder.has_emitted_tool_block()

    def test_content_block_start_tool_use_extra_content(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        sse = builder.content_block_start(
            2,
            "tool_use",
            id="tool_123",
            name="Read",
            input={},
            extra_content={"google": {"thought_signature": "sig"}},
        )

        data = _parse_sse(sse)
        assert data["content_block"]["extra_content"] == {
            "google": {"thought_signature": "sig"}
        }

    def test_content_block_delta_text(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        sse = builder.content_block_delta(0, "text_delta", "hello world")

        data = _parse_sse(sse)
        assert data["type"] == "content_block_delta"
        assert data["index"] == 0
        assert data["delta"]["type"] == "text_delta"
        assert data["delta"]["text"] == "hello world"

    def test_content_block_delta_thinking(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        sse = builder.content_block_delta(1, "thinking_delta", "reasoning...")

        data = _parse_sse(sse)
        assert data["delta"]["type"] == "thinking_delta"
        assert data["delta"]["thinking"] == "reasoning..."

    def test_content_block_delta_input_json(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        sse = builder.content_block_delta(2, "input_json_delta", '{"key": "val"}')

        data = _parse_sse(sse)
        assert data["delta"]["type"] == "input_json_delta"
        assert data["delta"]["partial_json"] == '{"key": "val"}'

    def test_content_block_stop(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        sse = builder.content_block_stop(0)

        data = _parse_sse(sse)
        assert data["type"] == "content_block_stop"
        assert data["index"] == 0


class TestAnthropicStreamLedgerHighLevelHelpers:
    """Tests for high-level thinking/text/tool block helpers."""

    def test_start_and_stop_thinking_block(self):
        builder = AnthropicStreamLedger("msg_1", "model")

        start_sse = builder.start_thinking_block()
        data = _parse_sse(start_sse)
        assert data["content_block"]["type"] == "thinking"
        assert builder.blocks.thinking_started is True
        assert builder.blocks.thinking_index == 0

        stop_sse = builder.stop_thinking_block()
        data = _parse_sse(stop_sse)
        assert data["type"] == "content_block_stop"
        assert builder.blocks.thinking_started is False

    def test_emit_thinking_delta_accumulates(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.start_thinking_block()

        builder.emit_thinking_delta("part1 ")
        builder.emit_thinking_delta("part2")

        assert builder.accumulated_reasoning == "part1 part2"

    def test_start_and_stop_text_block(self):
        builder = AnthropicStreamLedger("msg_1", "model")

        start_sse = builder.start_text_block()
        data = _parse_sse(start_sse)
        assert data["content_block"]["type"] == "text"
        assert builder.blocks.text_started is True
        assert builder.blocks.text_index == 0

        builder.stop_text_block()
        assert builder.blocks.text_started is False

    def test_emit_text_delta_accumulates(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.start_text_block()

        builder.emit_text_delta("hello ")
        builder.emit_text_delta("world")

        assert builder.accumulated_text == "hello world"

    def test_start_tool_block(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        sse = builder.start_tool_block(0, "tool_abc", "Grep")

        data = _parse_sse(sse)
        assert data["content_block"]["type"] == "tool_use"
        assert data["content_block"]["id"] == "tool_abc"
        assert data["content_block"]["name"] == "Grep"
        assert 0 in builder.blocks.tool_states

    def test_emit_tool_delta(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.start_tool_block(0, "tool_abc", "Grep")

        sse = builder.emit_tool_delta(0, '{"pattern":')
        data = _parse_sse(sse)
        assert data["delta"]["partial_json"] == '{"pattern":'
        block = builder.tool_block_for_tool_index(0)
        assert block is not None
        assert block.content == '{"pattern":'

    def test_stop_tool_block(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.start_tool_block(0, "tool_abc", "Grep")

        sse = builder.stop_tool_block(0)
        data = _parse_sse(sse)
        assert data["type"] == "content_block_stop"

    def test_text_suffix_after_closed_native_block_closes_once(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.ingest_native_event(
            SSEEvent(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                "",
            )
        )
        builder.ingest_native_event(
            SSEEvent(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hello"},
                },
                "",
            )
        )
        builder.ingest_native_event(
            SSEEvent(
                "content_block_stop",
                {"type": "content_block_stop", "index": 0},
                "",
            )
        )

        events = list(builder.append_text_suffix(" world"))
        events.extend(builder.success_tail("end_turn"))
        parsed = parse_sse_text("".join(events))

        assert [
            event.data["index"]
            for event in parsed
            if event.event == "content_block_start"
        ] == [1]
        assert [
            event.data["index"]
            for event in parsed
            if event.event == "content_block_delta"
        ] == [1]
        assert [
            event.data["index"]
            for event in parsed
            if event.event == "content_block_stop"
        ] == [1]

    def test_thinking_suffix_after_closed_native_block_closes_once(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.ingest_native_event(
            SSEEvent(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
                "",
            )
        )
        builder.ingest_native_event(
            SSEEvent(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "step one"},
                },
                "",
            )
        )
        builder.ingest_native_event(
            SSEEvent(
                "content_block_stop",
                {"type": "content_block_stop", "index": 0},
                "",
            )
        )

        events = list(builder.append_thinking_suffix(" step two"))
        events.extend(builder.success_tail("end_turn"))
        parsed = parse_sse_text("".join(events))

        assert [
            event.data["index"]
            for event in parsed
            if event.event == "content_block_start"
        ] == [1]
        assert [
            event.data["index"]
            for event in parsed
            if event.event == "content_block_delta"
        ] == [1]
        assert [
            event.data["index"]
            for event in parsed
            if event.event == "content_block_stop"
        ] == [1]

    def test_text_suffix_appends_to_open_native_block(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.ingest_native_event(
            SSEEvent(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                "",
            )
        )
        builder.ingest_native_event(
            SSEEvent(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hello wor"},
                },
                "",
            )
        )

        events = list(builder.append_text_suffix("ld"))
        events.extend(builder.success_tail("end_turn"))
        parsed = parse_sse_text("".join(events))

        assert [event.event for event in parsed] == [
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert parsed[0].data["index"] == 0
        assert parsed[0].data["delta"]["text"] == "ld"
        assert parsed[1].data["index"] == 0

    def test_thinking_suffix_appends_to_open_native_block(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.ingest_native_event(
            SSEEvent(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
                "",
            )
        )
        builder.ingest_native_event(
            SSEEvent(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "step o"},
                },
                "",
            )
        )

        events = list(builder.append_thinking_suffix("ne"))
        events.extend(builder.success_tail("end_turn"))
        parsed = parse_sse_text("".join(events))

        assert [event.event for event in parsed] == [
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert parsed[0].data["index"] == 0
        assert parsed[0].data["delta"]["thinking"] == "ne"
        assert parsed[1].data["index"] == 0

    def test_suffix_helpers_noop_after_message_delta(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.start_text_block()
        builder.emit_text_delta("hello")
        builder.stop_text_block()
        builder.start_tool_block(0, "toolu_1", "Read")
        builder.emit_tool_delta(0, '{"path":"test.py"}')
        builder.stop_tool_block(0)
        builder.message_delta("tool_use", 10)

        assert list(builder.append_text_suffix(" world")) == []
        assert list(builder.append_thinking_suffix(" step")) == []
        assert list(builder.append_tool_repair_suffix(0, " ")) == []

    def test_suffix_helpers_noop_after_message_stop(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.message_stop()

        assert list(builder.append_text_suffix("late")) == []
        assert list(builder.append_thinking_suffix("late")) == []

    def test_final_stop_reason_uses_emitted_tool_content(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        assert builder.final_stop_reason("end_turn") == "end_turn"

        builder.content_block_start(0, "tool_use", id="toolu_1", name="Read")

        assert builder.final_stop_reason("end_turn") == "tool_use"


class TestAnthropicStreamLedgerStateManagement:
    """Tests for ensure_thinking_block, ensure_text_block, close_all_blocks."""

    def test_ensure_thinking_block_closes_text_first(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.start_text_block()
        assert builder.blocks.text_started is True

        events = list(builder.ensure_thinking_block())
        # Should close text then start thinking
        assert len(events) == 2
        assert builder.blocks.text_started is False
        assert builder.blocks.thinking_started is True

    def test_ensure_thinking_block_noop_if_already_started(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.start_thinking_block()

        events = list(builder.ensure_thinking_block())
        assert events == []

    def test_ensure_text_block_closes_thinking_first(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.start_thinking_block()
        assert builder.blocks.thinking_started is True

        events = list(builder.ensure_text_block())
        # Should close thinking then start text
        assert len(events) == 2
        assert builder.blocks.thinking_started is False
        assert builder.blocks.text_started is True

    def test_ensure_text_block_noop_if_already_started(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.start_text_block()

        events = list(builder.ensure_text_block())
        assert events == []

    def test_append_text_suffix_closes_open_thinking_before_switching(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.start_thinking_block()

        events = list(builder.append_thinking_suffix(" step"))
        events.extend(builder.append_text_suffix(" answer"))
        parsed = parse_sse_text("".join(events))

        assert [event.event for event in parsed] == [
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
        ]
        assert parsed[0].data["index"] == 0
        assert parsed[1].data["index"] == 0
        assert parsed[2].data["index"] == 1
        assert parsed[2].data["content_block"]["type"] == "text"
        assert parsed[3].data["index"] == 1
        assert builder.blocks.thinking_started is False
        assert builder.blocks.text_started is True

    def test_append_thinking_suffix_closes_open_text_before_switching(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.start_text_block()

        events = list(builder.append_text_suffix(" answer"))
        events.extend(builder.append_thinking_suffix(" step"))
        parsed = parse_sse_text("".join(events))

        assert [event.event for event in parsed] == [
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
        ]
        assert parsed[0].data["index"] == 0
        assert parsed[1].data["index"] == 0
        assert parsed[2].data["index"] == 1
        assert parsed[2].data["content_block"]["type"] == "thinking"
        assert parsed[3].data["index"] == 1
        assert builder.blocks.text_started is False
        assert builder.blocks.thinking_started is True

    def test_close_content_blocks(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.start_thinking_block()
        builder.stop_thinking_block()
        builder.start_text_block()

        events = list(builder.close_content_blocks())
        # Should close text (thinking already stopped)
        assert len(events) == 1
        assert builder.blocks.text_started is False

    def test_close_all_blocks(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.start_thinking_block()
        builder.stop_thinking_block()
        builder.start_text_block()
        builder.start_tool_block(0, "t1", "Read")
        builder.start_tool_block(1, "t2", "Write")

        events = list(builder.close_all_blocks())
        # Close text + 2 tool blocks (thinking already stopped)
        assert len(events) == 3
        assert builder.blocks.text_started is False

    def test_close_all_blocks_empty(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        events = list(builder.close_all_blocks())
        assert events == []


class TestAnthropicStreamLedgerError:
    """Tests for emit_error."""

    def test_emit_error(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        events = list(builder.emit_error("Something went wrong"))

        assert len(events) == 3  # start, delta, stop
        start_data = _parse_sse(events[0])
        assert start_data["content_block"]["type"] == "text"

        delta_data = _parse_sse(events[1])
        assert delta_data["delta"]["text"] == "Something went wrong"

        stop_data = _parse_sse(events[2])
        assert stop_data["type"] == "content_block_stop"

    def test_midstream_error_tail_without_terminal_state_emits_message_tail(self):
        builder = AnthropicStreamLedger("msg_1", "model")

        events = list(builder.midstream_error_tail("Something went wrong"))
        parsed = parse_sse_text("".join(events))

        assert [event.event for event in parsed] == [
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert parsed[1].data["delta"]["text"] == "Something went wrong"
        assert parsed[3].data["delta"]["stop_reason"] == "end_turn"

    def test_midstream_error_tail_after_message_delta_uses_top_level_error(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.ingest_native_event(
            SSEEvent(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                },
                "",
            )
        )

        events = list(builder.midstream_error_tail("Something went wrong"))
        parsed = parse_sse_text("".join(events))

        assert [event.event for event in parsed] == ["error", "message_stop"]
        assert parsed[0].data["error"]["message"] == "Something went wrong"

    def test_midstream_error_tail_after_message_stop_does_not_duplicate_terminal(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.ingest_native_event(
            SSEEvent(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"input_tokens": 1, "output_tokens": 2},
                },
                "",
            )
        )
        builder.ingest_native_event(
            SSEEvent("message_stop", {"type": "message_stop"}, "")
        )

        events = list(builder.midstream_error_tail("Something went wrong"))
        parsed = parse_sse_text("".join(events))

        assert [event.event for event in parsed] == ["error"]
        assert parsed[0].data["error"]["message"] == "Something went wrong"


class TestAnthropicStreamLedgerTokenEstimation:
    """Tests for estimate_output_tokens."""

    def test_estimate_with_text_only(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.start_text_block()
        builder.emit_text_delta("hello world")

        tokens = builder.estimate_output_tokens()
        assert tokens > 0

    def test_estimate_with_reasoning(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.start_thinking_block()
        builder.emit_thinking_delta("deep thought")
        builder.stop_thinking_block()
        builder.start_text_block()
        builder.emit_text_delta("answer")

        tokens = builder.estimate_output_tokens()
        assert tokens > 0

    def test_estimate_empty(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        tokens = builder.estimate_output_tokens()
        assert tokens == 0

    def test_estimate_without_tiktoken(self):
        """Fallback estimation when tiktoken is not available."""
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.start_text_block()
        builder.emit_text_delta("a" * 100)  # 100 chars -> ~25 tokens

        with patch("core.anthropic.streaming.ledger.ENCODER", None):
            tokens = builder.estimate_output_tokens()
            assert tokens == 25  # 100 // 4

    def test_estimate_with_tools_no_tiktoken(self):
        """Fallback tool token estimation."""
        builder = AnthropicStreamLedger("msg_1", "model")
        builder.start_tool_block(0, "t1", "Read")
        builder.emit_tool_delta(0, '{"path":"test.py"}')

        with patch("core.anthropic.streaming.ledger.ENCODER", None):
            tokens = builder.estimate_output_tokens()
            # 1 tool * 50 = 50
            assert tokens == 50

    def test_estimate_counts_native_tool_payload(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        tool_name = "Read"
        tool_args = '{"path":"test.py"}'
        builder.ingest_native_event(
            SSEEvent(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_native",
                        "name": tool_name,
                        "input": {},
                    },
                },
                "",
            )
        )
        builder.ingest_native_event(
            SSEEvent(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": tool_args,
                    },
                },
                "",
            )
        )

        with patch("core.anthropic.streaming.ledger.ENCODER", _CharEncoder()):
            tokens = builder.estimate_output_tokens()

        assert tokens == len(tool_name) + len(tool_args) + 15 + 4

    def test_estimate_does_not_double_count_openai_tool_content_block(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        tool_name = "Read"
        tool_args = '{"path":"test.py"}'
        builder.start_tool_block(0, "toolu_openai", tool_name)
        builder.emit_tool_delta(0, tool_args)

        with patch("core.anthropic.streaming.ledger.ENCODER", _CharEncoder()):
            tokens = builder.estimate_output_tokens()

        assert tokens == len(tool_name) + len(tool_args) + 15 + 4

    def test_estimate_ignores_unstarted_openai_tool_state(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        state = builder.blocks.ensure_tool_state(0)
        state.name = "Read"
        state.pre_start_args = '{"path":"test.py"}'

        with patch("core.anthropic.streaming.ledger.ENCODER", _CharEncoder()):
            tokens = builder.estimate_output_tokens()

        assert tokens == 0

    def test_estimate_heuristic_style_tool_content_block(self):
        builder = AnthropicStreamLedger("msg_1", "model")
        tool_name = "Read"
        tool_args = '{"path":"test.py"}'
        builder.content_block_start(0, "tool_use", id="toolu_heuristic", name=tool_name)
        builder.content_block_delta(0, "input_json_delta", tool_args)
        builder.content_block_stop(0)

        with patch("core.anthropic.streaming.ledger.ENCODER", _CharEncoder()):
            tokens = builder.estimate_output_tokens()

        assert tokens == len(tool_name) + len(tool_args) + 15 + 4
