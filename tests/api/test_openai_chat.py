import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.streaming import format_sse_event
from tests.api.support import create_test_app


class FakeProvider:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.preflight_stream = MagicMock()
        self.requests: list[Any] = []
        self.stream_kwargs: list[dict[str, Any]] = []

    async def stream_response(self, request_data, **_kwargs):
        self.requests.append(request_data)
        self.stream_kwargs.append(_kwargs)
        for chunk in self.chunks:
            yield chunk


def _anthropic_text_stream(text: str, *, stop_reason: str = "end_turn") -> list[str]:
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
                "usage": {"output_tokens": 5},
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]


def _anthropic_tool_use_stream() -> list[str]:
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
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"city": "Paris"}',
                },
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


@pytest.fixture
def chat_client():
    provider = FakeProvider(_anthropic_text_stream("Hello from provider"))
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        yield client, provider


def test_chat_probe_endpoints_return_204(chat_client) -> None:
    client, _provider = chat_client
    assert client.head("/v1/chat/completions").status_code == 204
    assert client.options("/v1/chat/completions").status_code == 204


def test_non_stream_chat_completion_routes_through_provider(chat_client) -> None:
    client, provider = chat_client
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "nvidia_nim/test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 32,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert choice["message"]["content"] == "Hello from provider"
    assert choice["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 3
    assert body["usage"]["completion_tokens"] == 5
    assert provider.preflight_stream.called
    routed = provider.requests[0]
    assert routed.model == "test-model"
    assert routed.messages[0].role == "user"
    assert routed.max_tokens == 32


def test_stream_chat_completion_emits_chunks_and_done(chat_client) -> None:
    client, _provider = chat_client
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "nvidia_nim/test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    payloads = [
        line[len("data: ") :]
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert payloads[-1] == "[DONE]"
    objects = [json.loads(p) for p in payloads if p != "[DONE]"]
    assert objects[0]["choices"][0]["delta"] == {"role": "assistant"}
    content = "".join(
        obj["choices"][0]["delta"].get("content", "")
        for obj in objects
        if obj["choices"]
    )
    assert content == "Hello from provider"
    assert objects[-1]["choices"][0]["finish_reason"] == "stop"


def test_non_stream_tool_call_completion() -> None:
    provider = FakeProvider(_anthropic_tool_use_stream())
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "nvidia_nim/test-model",
                "messages": [{"role": "user", "content": "weather in Paris?"}],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            },
        )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Paris"}


def test_preflight_rejection_is_ordinary_openai_error() -> None:
    provider = FakeProvider(_anthropic_text_stream("unused"))
    provider.preflight_stream.side_effect = InvalidRequestError("bad tool shape")
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "nvidia_nim/test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "bad tool shape",
        "type": "invalid_request_error",
        "param": None,
        "code": None,
    }
    assert provider.requests == []
