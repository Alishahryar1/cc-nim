from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from core.anthropic.sse import format_sse_event
from core.anthropic.stream_contracts import parse_sse_text
from core.openai_responses import (
    collect_openai_response_from_anthropic_sse,
    iter_anthropic_sse_as_openai_responses,
)


@pytest.mark.asyncio
async def test_anthropic_text_stream_converts_to_responses_sse() -> None:
    text = await _collect_sse(
        iter_anthropic_sse_as_openai_responses(
            _aiter(_anthropic_text_stream("Hello Codex")),
            {"model": "nvidia_nim/test-model", "stream": True},
        )
    )

    events = parse_sse_text(text)
    event_names = [event.event for event in events]
    assert event_names[:3] == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
    ]
    assert "response.output_text.delta" in event_names
    assert events[-1].event == "response.completed"
    assert events[-1].data["response"]["output"][0]["content"][0]["text"] == (
        "Hello Codex"
    )


@pytest.mark.asyncio
async def test_anthropic_tool_stream_converts_to_function_call_item() -> None:
    text = await _collect_sse(
        iter_anthropic_sse_as_openai_responses(
            _aiter(_anthropic_tool_stream()),
            {"model": "nvidia_nim/test-model", "stream": True},
        )
    )

    events = parse_sse_text(text)
    names = [event.event for event in events]
    assert "response.function_call_arguments.delta" in names
    assert "response.function_call_arguments.done" in names
    completed = events[-1].data["response"]
    function_call = completed["output"][0]
    assert function_call["type"] == "function_call"
    assert function_call["call_id"] == "toolu_1"
    assert function_call["name"] == "echo"
    assert function_call["arguments"] == '{"value":"FCC"}'


@pytest.mark.asyncio
async def test_namespaced_anthropic_tool_stream_restores_responses_namespace() -> None:
    text = await _collect_sse(
        iter_anthropic_sse_as_openai_responses(
            _aiter(_anthropic_tool_stream(tool_name="mcp__node_repl__js")),
            {
                "model": "nvidia_nim/test-model",
                "stream": True,
                "tools": [
                    {
                        "type": "namespace",
                        "name": "mcp__node_repl",
                        "tools": [
                            {
                                "type": "function",
                                "name": "js",
                                "parameters": {"type": "object", "properties": {}},
                            }
                        ],
                    }
                ],
            },
        )
    )

    events = parse_sse_text(text)
    completed = events[-1].data["response"]
    function_call = completed["output"][0]
    assert function_call["type"] == "function_call"
    assert function_call["namespace"] == "mcp__node_repl"
    assert function_call["name"] == "js"


@pytest.mark.asyncio
async def test_anthropic_error_stream_converts_to_responses_error_event() -> None:
    text = await _collect_sse(
        iter_anthropic_sse_as_openai_responses(
            _aiter(
                [
                    format_sse_event(
                        "error",
                        {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": "upstream failed",
                            },
                        },
                    )
                ]
            ),
            {"model": "nvidia_nim/test-model", "stream": True},
        )
    )

    events = parse_sse_text(text)
    assert events[0].event == "response.created"
    assert events[1].event == "error"
    assert events[1].data["error"]["message"] == "upstream failed"


@pytest.mark.asyncio
async def test_collect_non_stream_response_from_anthropic_sse() -> None:
    response = await collect_openai_response_from_anthropic_sse(
        _aiter(_anthropic_text_stream("Non-stream")),
        {"model": "deepseek/deepseek-chat", "stream": False},
    )

    assert response["status"] == "completed"
    assert response["model"] == "deepseek/deepseek-chat"
    assert response["output"][0]["content"][0]["text"] == "Non-stream"


async def _collect_sse(chunks: AsyncIterator[str]) -> str:
    parts = [chunk async for chunk in chunks]
    return "".join(parts)


async def _aiter(chunks: list[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


def _anthropic_text_stream(text: str) -> list[str]:
    return [
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
                "delta": {"type": "text_delta", "text": text},
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
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]


def _anthropic_tool_stream(tool_name: str = "echo") -> list[str]:
    return [
        format_sse_event("message_start", {"type": "message_start", "message": {}}),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": tool_name,
                    "input": {},
                },
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"value":"FCC"}',
                },
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
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]
