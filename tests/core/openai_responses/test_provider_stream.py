import pytest

from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    thinking_content,
)
from free_claude_code.core.inference import (
    ReplayAttachment,
    ReplayCompatibilityScope,
)
from free_claude_code.core.replay_envelope import decode_replay_envelope
from free_claude_code.providers.openai_compat import OpenAIToolNameCodec
from free_claude_code.providers.openai_responses.events import (
    ResponsesEventDecoder,
    ResponsesStreamFailure,
)
from tests.inference_support import present_anthropic

_REPLAY_SCOPE = ReplayCompatibilityScope("openai_responses:test-model")


def test_responses_decoder_preserves_reasoning_tools_usage_and_ids() -> None:
    stream = ResponsesEventDecoder(
        response_id="response_test",
        model="openai/gpt-test",
        input_tokens=12,
        replay_scope=_REPLAY_SCOPE,
    )
    output = stream.start()
    output.extend(
        stream.feed(
            "response.output_item.added",
            {
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                }
            },
        )
    )
    output.extend(
        stream.feed(
            "response.reasoning_summary_text.delta",
            {"item_id": "rs_1", "delta": "reasoning"},
        )
    )
    output.extend(
        stream.feed(
            "response.output_item.done",
            {
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "opaque",
                }
            },
        )
    )
    output.extend(
        stream.feed(
            "response.output_item.added",
            {
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "lookup",
                }
            },
        )
    )
    output.extend(
        stream.feed(
            "response.function_call_arguments.delta",
            {"item_id": "fc_1", "delta": '{"q":'},
        )
    )
    output.extend(
        stream.feed(
            "response.function_call_arguments.delta",
            {"item_id": "fc_1", "delta": '"x"}'},
        )
    )
    output.extend(
        stream.feed(
            "response.output_item.done",
            {
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": '{"q":"x"}',
                }
            },
        )
    )
    output.extend(
        stream.feed(
            "response.completed",
            {
                "response": {
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 8,
                        "input_tokens_details": {"cached_tokens": 15},
                    }
                }
            },
        )
    )

    events = parse_sse_text("".join(present_anthropic(output)))
    assert_anthropic_stream_contract(events)
    assert thinking_content(events) == "reasoning"
    starts = [
        event.data["content_block"]
        for event in events
        if event.event == "content_block_start"
    ]
    redacted = next(block for block in starts if block["type"] == "redacted_thinking")
    assert isinstance(redacted["data"], str)
    artifacts = decode_replay_envelope(
        redacted["data"],
        attachment=ReplayAttachment.REASONING,
    )
    assert artifacts is not None
    assert [artifact.payload for artifact in artifacts] == ["opaque"]
    assert [artifact.scope for artifact in artifacts] == [_REPLAY_SCOPE]
    assert {
        "type": "tool_use",
        "id": "call_1",
        "name": "lookup",
        "input": {},
    } in starts
    argument_deltas = [
        event.data["delta"]["partial_json"]
        for event in events
        if event.data.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert argument_deltas == ['{"q":', '"x"}']
    message_delta = next(event for event in events if event.event == "message_delta")
    assert message_delta.data["usage"] == {
        "input_tokens": 5,
        "output_tokens": 8,
        "cache_read_input_tokens": 15,
    }


@pytest.mark.parametrize(
    ("input_tokens", "cached_tokens", "expected_input_tokens"),
    [
        (20, -1, 20),
        (20, 21, 20),
        (20, True, 20),
        (None, 5, 12),
    ],
)
def test_responses_decoder_ignores_invalid_cache_partitions(
    input_tokens: int | None,
    cached_tokens: int | bool,
    expected_input_tokens: int,
) -> None:
    stream = ResponsesEventDecoder(
        response_id="response_test",
        model="openai/gpt-test",
        input_tokens=12,
        replay_scope=_REPLAY_SCOPE,
    )
    output = stream.start()
    output.extend(
        stream.feed(
            "response.completed",
            {
                "response": {
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 8,
                        "input_tokens_details": {"cached_tokens": cached_tokens},
                    }
                }
            },
        )
    )

    message_delta = next(
        event
        for event in parse_sse_text("".join(present_anthropic(output)))
        if event.event == "message_delta"
    )
    assert message_delta.data["usage"] == {
        "input_tokens": expected_input_tokens,
        "output_tokens": 8,
    }


def test_responses_decoder_surfaces_failed_event() -> None:
    stream = ResponsesEventDecoder(
        response_id="response_test",
        model="gpt-test",
        input_tokens=0,
        replay_scope=_REPLAY_SCOPE,
    )

    with pytest.raises(ResponsesStreamFailure, match="capacity") as exc_info:
        stream.feed(
            "response.failed",
            {
                "response": {
                    "error": {
                        "code": "server_error",
                        "message": "No capacity",
                    }
                }
            },
        )

    assert exc_info.value.code == "server_error"


def test_responses_decoder_restores_added_and_done_only_tool_names() -> None:
    originals = (
        "mcp__responses_added__" + "x" * 70,
        "mcp__responses_done__" + "y" * 70,
    )
    codec = OpenAIToolNameCodec.from_names(originals)
    stream = ResponsesEventDecoder(
        response_id="response_test",
        model="gpt-test",
        input_tokens=0,
        tool_names=codec,
        replay_scope=_REPLAY_SCOPE,
    )
    output = stream.start()
    output.extend(
        stream.feed(
            "response.output_item.added",
            {
                "item": {
                    "type": "function_call",
                    "id": "fc_added",
                    "call_id": "call_added",
                    "name": codec.encode(originals[0]),
                }
            },
        )
    )
    output.extend(
        stream.feed(
            "response.output_item.done",
            {
                "item": {
                    "type": "function_call",
                    "id": "fc_added",
                    "call_id": "call_added",
                    "name": codec.encode(originals[0]),
                    "arguments": "{}",
                }
            },
        )
    )
    output.extend(
        stream.feed(
            "response.output_item.done",
            {
                "item": {
                    "type": "function_call",
                    "id": "fc_done",
                    "call_id": "call_done",
                    "name": codec.encode(originals[1]),
                    "arguments": "{}",
                }
            },
        )
    )

    event_text = "".join(present_anthropic(output))
    starts = [
        event.data["content_block"]
        for event in parse_sse_text(event_text)
        if event.event == "content_block_start"
    ]
    assert [start["name"] for start in starts] == list(originals)
    assert all(codec.encode(name) not in event_text for name in originals)
