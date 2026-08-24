"""Contracts for the canonical ledger and Anthropic event presenter."""

from typing import cast
from unittest.mock import patch

import pytest

from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
)
from free_claude_code.core.anthropic.streaming import map_stop_reason
from free_claude_code.core.inference import (
    FinishReason,
    InferenceBlockLedger,
    InferenceStreamLedger,
    ReasoningBlockCompleted,
    TextBlockCompleted,
    TextBlockStarted,
    TokenMeasurement,
    ToolCallCompleted,
    UsageSource,
)
from tests.inference_support import present_anthropic, reported_usage


@pytest.mark.parametrize("value", [True, -1, cast(int, 1.5)])
def test_token_measurement_rejects_non_integer_or_negative_values(value: int) -> None:
    with pytest.raises(ValueError, match="non-negative integers"):
        TokenMeasurement(value, UsageSource.REPORTED)


@pytest.mark.parametrize(
    ("upstream", "anthropic"),
    [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("tool_calls", "tool_use"),
        ("content_filter", "end_turn"),
        (None, "end_turn"),
    ],
)
def test_map_stop_reason(upstream: str | None, anthropic: str) -> None:
    assert map_stop_reason(upstream) == anthropic


def test_block_ledger_reuses_tool_state_by_upstream_index() -> None:
    blocks = InferenceBlockLedger()

    assert blocks.ensure_tool_state(2) is blocks.ensure_tool_state(2)
    assert set(blocks.tool_states) == {2}


def test_response_lifecycle_is_canonical_until_anthropic_presentation() -> None:
    ledger = InferenceStreamLedger("response_1", "model", input_tokens=7)
    events = [ledger.start_response()]
    events.extend(ledger.ensure_text_block())
    events.append(ledger.emit_text_delta("done"))
    events.extend(ledger.close_all_blocks())
    events.extend(
        ledger.finish_events(
            FinishReason.END_TURN,
            reported_usage(input_tokens=7, output_tokens=3),
        )
    )

    parsed = parse_sse_text("".join(present_anthropic(events)))

    assert_anthropic_stream_contract(parsed)
    start = parsed[0].data["message"]
    assert start["id"].startswith("msg_")
    assert start["model"] == "model"
    assert start["usage"]["input_tokens"] == 7
    delta = next(event.data for event in parsed if event.event == "message_delta")
    assert delta["delta"]["stop_reason"] == "end_turn"
    assert delta["usage"]["output_tokens"] == 3
    assert parsed[-1].data == {"type": "message_stop"}


def test_text_and_reasoning_blocks_accumulate_semantic_content() -> None:
    ledger = InferenceStreamLedger("response_1", "model")
    ledger.start_response()
    ledger.start_reasoning_block()
    ledger.emit_reasoning_delta("step")
    reasoning = ledger.stop_reasoning_block()
    ledger.start_text_block()
    ledger.emit_text_delta("answer")
    text = ledger.stop_text_block()

    assert isinstance(reasoning, ReasoningBlockCompleted)
    assert reasoning.reasoning == "step"
    assert isinstance(text, TextBlockCompleted)
    assert text.text == "answer"
    assert ledger.accumulated_reasoning == "step"
    assert ledger.accumulated_text == "answer"


def test_ensure_block_switch_closes_previous_kind() -> None:
    ledger = InferenceStreamLedger("response_1", "model")
    ledger.start_response()
    ledger.start_reasoning_block()

    events = list(ledger.ensure_text_block())

    assert isinstance(events[0], ReasoningBlockCompleted)
    assert isinstance(events[1], TextBlockStarted)
    assert not ledger.blocks.reasoning_started
    assert ledger.blocks.text_started


def test_tool_blocks_drive_finish_reason() -> None:
    ledger = InferenceStreamLedger("response_1", "model")
    ledger.start_response()
    ledger.start_tool_block(0, "toolu_1", "Read")
    ledger.emit_tool_delta(0, '{"path":"test.py"}')

    assert ledger.final_finish_reason(FinishReason.END_TURN) is FinishReason.TOOL_CALLS


def test_close_unclosed_blocks_closes_each_semantic_block_once() -> None:
    ledger = InferenceStreamLedger("response_1", "model")
    ledger.start_response()
    ledger.start_text_block()
    ledger.start_tool_block(0, "toolu_1", "Read")

    events = list(ledger.close_unclosed_blocks())

    assert len(events) == 2
    assert isinstance(events[0], ToolCallCompleted)
    assert isinstance(events[1], TextBlockCompleted)
    assert list(ledger.close_unclosed_blocks()) == []


def test_parallel_tool_blocks_complete_independently() -> None:
    ledger = InferenceStreamLedger("response_1", "model")
    events = [ledger.start_response()]
    events.extend(
        [
            ledger.start_tool_block(0, "toolu_1", "Read"),
            ledger.start_tool_block(1, "toolu_2", "Write"),
            ledger.emit_tool_delta(0, '{"path":"README.md"}'),
            ledger.emit_tool_delta(1, '{"path":"notes.md"}'),
            ledger.stop_tool_block(0),
            ledger.stop_tool_block(1),
        ]
    )
    events.extend(
        ledger.finish_events(
            FinishReason.TOOL_CALLS,
            reported_usage(input_tokens=7, output_tokens=8),
        )
    )

    parsed = parse_sse_text("".join(present_anthropic(events)))

    assert_anthropic_stream_contract(parsed)
    starts = [
        event.data["content_block"]
        for event in parsed
        if event.event == "content_block_start"
    ]
    assert [block["id"] for block in starts] == ["toolu_1", "toolu_2"]
    assert [block["name"] for block in starts] == ["Read", "Write"]


def test_output_token_estimate_combines_shared_estimates_and_block_overhead() -> None:
    ledger = InferenceStreamLedger("response_1", "model")
    ledger.start_response()
    ledger.start_reasoning_block()
    ledger.emit_reasoning_delta("why")
    ledger.stop_reasoning_block()
    ledger.start_text_block()
    ledger.emit_text_delta("abcd")
    ledger.stop_text_block()
    ledger.start_tool_block(0, "toolu_1", "Read")
    ledger.emit_tool_delta(0, "{}")

    with patch(
        "free_claude_code.core.inference.ledger.estimate_text_tokens",
        side_effect=len,
    ):
        assert ledger.estimate_output_tokens() == 40
