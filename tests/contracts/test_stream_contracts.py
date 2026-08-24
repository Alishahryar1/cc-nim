"""Cross-protocol stream ordering contracts."""

from collections.abc import Iterable

from free_claude_code.core.anthropic import (
    ContentType,
    HeuristicToolParser,
    ThinkTagParser,
)
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    event_names,
    parse_sse_text,
    text_content,
    thinking_content,
)
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.inference import (
    FinishReason,
    InferenceEvent,
    InferenceStreamLedger,
)
from tests.inference_support import present_anthropic, reported_usage


def test_interleaved_thinking_text_blocks_are_valid() -> None:
    events = _parse_builder_events(
        _interleaved_thinking_text_events(
            ("first thought", "first answer", "second thought", "final answer")
        )
    )
    assert_anthropic_stream_contract(events)
    assert event_names(events).count("content_block_start") == 4
    assert thinking_content(events) == "first thoughtsecond thought"
    assert text_content(events) == "first answerfinal answer"


def test_split_think_tags_preserve_text_and_thinking() -> None:
    events = _parse_builder_events(
        _events_from_text_chunks(["before <thi", "nk>hidden", "</think> after"])
    )
    assert_anthropic_stream_contract(events)
    assert thinking_content(events) == "hidden"
    assert text_content(events) == "before  after"


def test_mixed_reasoning_content_and_think_tags_keep_order() -> None:
    ledger = InferenceStreamLedger("response_contract", "contract-model")
    events: list[InferenceEvent] = [ledger.start_response()]
    events.extend(ledger.ensure_reasoning_block())
    events.append(ledger.emit_reasoning_delta("reasoning field"))
    events.extend(
        _events_from_text_chunks(
            [" visible <think>tagged</think> done"],
            ledger,
        )
    )
    events.extend(ledger.close_all_blocks())
    events.extend(
        ledger.finish_events(
            FinishReason.END_TURN,
            reported_usage(input_tokens=0, output_tokens=10),
        )
    )

    parsed = parse_sse_text("".join(present_anthropic(events)))
    assert_anthropic_stream_contract(parsed)
    assert thinking_content(parsed) == "reasoning fieldtagged"
    assert text_content(parsed) == " visible  done"


def test_redacted_thinking_block_start_stop_is_valid() -> None:
    """Native redacted_thinking uses start/stop only (no deltas)."""
    chunks = [
        format_sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_r",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "m",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            },
        ),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "redacted_thinking", "data": "opaque"},
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]
    events = parse_sse_text("".join(chunks))
    assert_anthropic_stream_contract(events)


def test_enable_thinking_false_suppresses_reasoning_only() -> None:
    events = _parse_builder_events(
        _events_from_text_chunks(
            ["hello <think>secret</think> world"], enable_thinking=False
        )
    )
    assert_anthropic_stream_contract(events)
    assert "secret" not in thinking_content(events)
    assert text_content(events) == "hello  world"


def test_task_tool_arguments_force_foreground_execution() -> None:
    parser = HeuristicToolParser()
    filtered, detected = parser.feed(
        "● <function=Task><parameter=description>Inspect</parameter>"
        "<parameter=run_in_background>true</parameter> trailing"
    )
    detected.extend(parser.flush())
    assert "trailing" in filtered
    task = detected[0]
    assert task["name"] == "Task"
    if isinstance(task.get("input"), dict):
        task["input"]["run_in_background"] = False
    assert task["input"]["run_in_background"] is False


def _interleaved_thinking_text_events(
    parts: tuple[str, str, str, str],
) -> Iterable[InferenceEvent]:
    ledger = InferenceStreamLedger("response_contract", "contract-model")
    yield ledger.start_response()
    yield from ledger.ensure_reasoning_block()
    yield ledger.emit_reasoning_delta(parts[0])
    yield from ledger.ensure_text_block()
    yield ledger.emit_text_delta(parts[1])
    yield from ledger.ensure_reasoning_block()
    yield ledger.emit_reasoning_delta(parts[2])
    yield from ledger.ensure_text_block()
    yield ledger.emit_text_delta(parts[3])
    yield from ledger.close_all_blocks()
    yield from ledger.finish_events(
        FinishReason.END_TURN,
        reported_usage(input_tokens=0, output_tokens=20),
    )


def _events_from_text_chunks(
    chunks: list[str],
    ledger: InferenceStreamLedger | None = None,
    *,
    enable_thinking: bool = True,
) -> list[InferenceEvent]:
    stream = ledger or InferenceStreamLedger(
        "response_contract",
        "contract-model",
    )
    out: list[InferenceEvent] = [] if ledger else [stream.start_response()]
    parser = ThinkTagParser()

    for chunk in chunks:
        out.extend(_emit_parser_parts(stream, parser.feed(chunk), enable_thinking))

    remaining = parser.flush()
    if remaining is not None:
        out.extend(_emit_parser_parts(stream, [remaining], enable_thinking))

    if ledger is None:
        out.extend(stream.close_all_blocks())
        out.extend(
            stream.finish_events(
                FinishReason.END_TURN,
                reported_usage(input_tokens=0, output_tokens=20),
            )
        )
    return out


def _emit_parser_parts(
    ledger: InferenceStreamLedger,
    parts: Iterable,
    enable_thinking: bool,
) -> list[InferenceEvent]:
    out: list[InferenceEvent] = []
    for part in parts:
        if part.type == ContentType.THINKING:
            if enable_thinking:
                out.extend(ledger.ensure_reasoning_block())
                out.append(ledger.emit_reasoning_delta(part.content))
            continue
        out.extend(ledger.ensure_text_block())
        out.append(ledger.emit_text_delta(part.content))
    return out


def _parse_builder_events(events: Iterable[InferenceEvent]):
    return parse_sse_text("".join(present_anthropic(events)))
