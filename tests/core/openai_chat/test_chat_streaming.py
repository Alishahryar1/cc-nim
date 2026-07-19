import asyncio
import json
from collections.abc import AsyncIterator

from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.openai_chat import OpenAIChatCompletionsRequest
from free_claude_code.core.openai_chat.streaming import (
    iter_chat_completions_sse_from_anthropic,
)


def _request(**kwargs) -> OpenAIChatCompletionsRequest:
    kwargs.setdefault("model", "nvidia_nim/x")
    kwargs.setdefault("messages", [{"role": "user", "content": "hi"}])
    return OpenAIChatCompletionsRequest(**kwargs)


async def _source(chunks: list[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


def _collect(chunks: list[str], request: OpenAIChatCompletionsRequest) -> list[str]:
    async def run() -> list[str]:
        return [
            frame
            async for frame in iter_chat_completions_sse_from_anthropic(
                _source(chunks), request
            )
        ]

    return asyncio.run(run())


def _data_objects(frames: list[str]) -> list[dict]:
    objects: list[dict] = []
    for frame in frames:
        payload = frame.removeprefix("data: ").strip()
        if payload and payload != "[DONE]":
            objects.append(json.loads(payload))
    return objects


def _text_stream(text: str, *, stop_reason: str = "end_turn") -> list[str]:
    return [
        format_sse_event(
            "message_start",
            {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
        ),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        format_sse_event(
            "content_block_stop", {"type": "content_block_stop", "index": 0}
        ),
        format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason},
                "usage": {"output_tokens": 4},
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]


def test_text_stream_emits_role_content_and_finish() -> None:
    frames = _collect(_text_stream("Hello world"), _request())
    assert frames[-1] == "data: [DONE]\n\n"
    objects = _data_objects(frames)
    assert objects[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert all(obj["object"] == "chat.completion.chunk" for obj in objects)
    content = "".join(
        obj["choices"][0]["delta"].get("content", "")
        for obj in objects
        if obj["choices"]
    )
    assert content == "Hello world"
    assert objects[-1]["choices"][0]["finish_reason"] == "stop"


def test_max_tokens_stream_finish_reason_is_length() -> None:
    frames = _collect(_text_stream("partial", stop_reason="max_tokens"), _request())
    objects = _data_objects(frames)
    assert objects[-1]["choices"][0]["finish_reason"] == "length"


def test_tool_use_stream_emits_tool_call_deltas() -> None:
    chunks = [
        format_sse_event("message_start", {"type": "message_start", "message": {}}),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_9",
                    "name": "get_weather",
                    "input": {},
                },
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": ' "Paris"}'},
            },
        ),
        format_sse_event(
            "content_block_stop", {"type": "content_block_stop", "index": 0}
        ),
        format_sse_event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]
    objects = _data_objects(_collect(chunks, _request()))
    tool_deltas = [
        obj["choices"][0]["delta"]["tool_calls"][0]
        for obj in objects
        if obj["choices"] and obj["choices"][0]["delta"].get("tool_calls")
    ]
    assert tool_deltas[0]["index"] == 0
    assert tool_deltas[0]["id"] == "toolu_9"
    assert tool_deltas[0]["function"]["name"] == "get_weather"
    arguments = "".join(delta["function"]["arguments"] for delta in tool_deltas)
    assert json.loads(arguments) == {"city": "Paris"}
    assert objects[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_include_usage_emits_trailing_usage_chunk() -> None:
    frames = _collect(
        _text_stream("hi"),
        _request(stream_options={"include_usage": True}),
    )
    objects = _data_objects(frames)
    # Every content chunk carries usage=None; the trailing chunk has empty choices.
    assert objects[0]["usage"] is None
    usage_chunk = objects[-1]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
    }


def test_crlf_framed_stream_parses() -> None:
    crlf = [chunk.replace("\n", "\r\n") for chunk in _text_stream("Hello world")]
    objects = _data_objects(_collect(crlf, _request()))
    content = "".join(
        o["choices"][0]["delta"].get("content", "") for o in objects if o["choices"]
    )
    assert content == "Hello world"
    assert objects[-1]["choices"][0]["finish_reason"] == "stop"


def test_truncated_stream_without_completion_emits_error() -> None:
    # Content, then the provider connection ends with no message_delta/message_stop.
    chunks = [
        format_sse_event("message_start", {"type": "message_start", "message": {}}),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "partial"},
            },
        ),
    ]
    frames = _collect(chunks, _request())
    assert frames[-1] == "data: [DONE]\n\n"
    objects = _data_objects(frames)
    assert any("error" in o for o in objects)
    # never fabricate a successful finish_reason for a cut-off stream
    assert not any(
        o.get("choices") and o["choices"][0].get("finish_reason") for o in objects
    )


def test_stop_reason_without_message_stop_finishes_normally() -> None:
    # The model signaled completion via message_delta; only message_stop is missing.
    chunks = [
        format_sse_event("message_start", {"type": "message_start", "message": {}}),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "done"},
            },
        ),
        format_sse_event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        ),
    ]
    objects = _data_objects(_collect(chunks, _request()))
    assert not any("error" in o for o in objects)
    assert objects[-1]["choices"][0]["finish_reason"] == "stop"
