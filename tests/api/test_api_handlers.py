import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.responses import JSONResponse, StreamingResponse

from api.handlers import MessagesHandler, ResponsesHandler, TokenCountHandler
from api.models.anthropic import Message, MessagesRequest, TokenCountRequest
from api.models.openai_responses import OpenAIResponsesRequest
from config.settings import Settings
from core.anthropic.streaming import format_sse_event
from providers.base import BaseProvider, ProviderConfig

_CLASSIFIER_SYSTEM = (
    "You are a security monitor. Respond with <block>yes</block> or <block>no</block>."
)
_CLASSIFIER_USER = (
    "<transcript>\nUser: review the repo\nWebFetch https://example.com: fetch\n"
    "</transcript>\n<block> immediately."
)


class FakeProvider(BaseProvider):
    def __init__(self, events: list[str] | None = None) -> None:
        super().__init__(ProviderConfig(api_key="test"))
        self.preflight_calls: list[tuple[Any, bool | None]] = []
        self.requests: list[Any] = []
        self.stream_kwargs: list[dict[str, Any]] = []
        self.events = events or [
            'event: message_start\ndata: {"type":"message_start"}\n\n',
            'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ]

    def preflight_stream(
        self, request: Any, *, thinking_enabled: bool | None = None
    ) -> None:
        self.preflight_calls.append((request, thinking_enabled))

    async def cleanup(self) -> None:
        return None

    async def list_model_ids(self) -> frozenset[str]:
        return frozenset({"test-model"})

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        self.requests.append(request)
        self.stream_kwargs.append(
            {
                "input_tokens": input_tokens,
                "request_id": request_id,
                "thinking_enabled": thinking_enabled,
            }
        )
        for event in self.events:
            yield event


async def _streaming_body_text(response: StreamingResponse) -> str:
    parts: list[str] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            parts.append(chunk.decode("utf-8"))
        else:
            parts.append(str(chunk))
    return "".join(parts)


def _json_response_content(response: JSONResponse) -> dict[str, Any]:
    content = json.loads(bytes(response.body).decode("utf-8"))
    assert isinstance(content, dict)
    return content


def _trace_events(trace_mock: MagicMock, event: str) -> list[dict[str, Any]]:
    return [
        dict(call.kwargs)
        for call in trace_mock.call_args_list
        if call.kwargs.get("event") == event
    ]


@pytest.mark.asyncio
async def test_messages_handler_passes_routed_request_and_stream_metadata() -> None:
    provider = FakeProvider()
    handler = MessagesHandler(Settings(), provider_getter=lambda _: provider)
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        messages=[Message(role="user", content="hi")],
    )

    response = await handler.create(request)
    assert isinstance(response, StreamingResponse)

    body = await _streaming_body_text(response)
    assert "message_start" in body
    assert provider.requests[0].model == "test-model"
    assert provider.stream_kwargs[0]["input_tokens"] > 0
    assert provider.stream_kwargs[0]["request_id"].startswith("req_")
    assert provider.stream_kwargs[0]["thinking_enabled"] is True
    assert len(provider.preflight_calls) == 1


@pytest.mark.asyncio
async def test_messages_handler_aggregates_provider_stream_when_stream_false() -> None:
    provider = FakeProvider(
        [
            format_sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "test-model",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 7, "output_tokens": 1},
                    },
                },
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
                    "delta": {"type": "text_delta", "text": "OK"},
                },
            ),
            format_sse_event(
                "content_block_stop", {"type": "content_block_stop", "index": 0}
            ),
            format_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"input_tokens": 7, "output_tokens": 2},
                },
            ),
            format_sse_event("message_stop", {"type": "message_stop"}),
        ]
    )
    handler = MessagesHandler(Settings(), provider_getter=lambda _: provider)
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        stream=False,
        messages=[Message(role="user", content="hi")],
    )

    response = await handler.create(request)

    assert isinstance(response, JSONResponse)
    assert response.headers["content-type"].startswith("application/json")
    body = _json_response_content(response)
    assert body["id"] == "msg_test"
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["model"] == "test-model"
    assert body["content"] == [{"type": "text", "text": "OK"}]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"] == {"input_tokens": 7, "output_tokens": 2}


@pytest.mark.asyncio
async def test_messages_handler_returns_error_json_for_stream_false_sse_error() -> None:
    provider = FakeProvider(
        [
            format_sse_event(
                "error",
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": "upstream failed"},
                },
            )
        ]
    )
    handler = MessagesHandler(Settings(), provider_getter=lambda _: provider)
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        stream=False,
        messages=[Message(role="user", content="hi")],
    )

    response = await handler.create(request)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 502
    body = _json_response_content(response)
    assert body == {
        "type": "error",
        "error": {"type": "api_error", "message": "upstream failed"},
    }


@pytest.mark.asyncio
async def test_messages_handler_forces_no_thinking_for_safety_classifier() -> None:
    provider = FakeProvider()
    handler = MessagesHandler(Settings(), provider_getter=lambda _: provider)
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        system=_CLASSIFIER_SYSTEM,
        messages=[Message(role="user", content=_CLASSIFIER_USER)],
    )

    with patch("api.handlers.messages.trace_event") as trace_mock:
        response = await handler.create(request)
        assert isinstance(response, StreamingResponse)
        await _streaming_body_text(response)

    assert provider.preflight_calls[0][1] is False
    assert provider.stream_kwargs[0]["thinking_enabled"] is False
    assert provider.requests[0].model == "test-model"
    assert provider.requests[0].system == _CLASSIFIER_SYSTEM
    assert _trace_events(
        trace_mock, "api.optimization.safety_classifier_no_thinking"
    ) == [
        {
            "stage": "routing",
            "event": "api.optimization.safety_classifier_no_thinking",
            "source": "api",
            "model": "test-model",
            "changed": True,
        }
    ]


@pytest.mark.asyncio
async def test_messages_handler_preserves_thinking_for_non_classifier() -> None:
    provider = FakeProvider()
    handler = MessagesHandler(Settings(), provider_getter=lambda _: provider)
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        system="Explain XML formats.",
        messages=[
            Message(
                role="user",
                content=(
                    "Explain <transcript>...</transcript> and a <block> tag "
                    "without making a verdict."
                ),
            )
        ],
    )

    with patch("api.handlers.messages.trace_event") as trace_mock:
        response = await handler.create(request)
        assert isinstance(response, StreamingResponse)
        await _streaming_body_text(response)

    assert provider.preflight_calls[0][1] is True
    assert provider.stream_kwargs[0]["thinking_enabled"] is True
    assert (
        _trace_events(trace_mock, "api.optimization.safety_classifier_no_thinking")
        == []
    )


@pytest.mark.asyncio
async def test_messages_handler_keeps_existing_no_thinking_for_classifier() -> None:
    provider = FakeProvider()
    handler = MessagesHandler(Settings(), provider_getter=lambda _: provider)
    request = MessagesRequest(
        model="claude-3-freecc-no-thinking/nvidia_nim/test-model",
        max_tokens=100,
        system=_CLASSIFIER_SYSTEM,
        messages=[Message(role="user", content=_CLASSIFIER_USER)],
    )

    with patch("api.handlers.messages.trace_event") as trace_mock:
        response = await handler.create(request)
        assert isinstance(response, StreamingResponse)
        await _streaming_body_text(response)

    assert provider.preflight_calls[0][1] is False
    assert provider.stream_kwargs[0]["thinking_enabled"] is False
    assert _trace_events(
        trace_mock, "api.optimization.safety_classifier_no_thinking"
    ) == [
        {
            "stage": "routing",
            "event": "api.optimization.safety_classifier_no_thinking",
            "source": "api",
            "model": "test-model",
            "changed": False,
        }
    ]


@pytest.mark.asyncio
async def test_messages_handler_optimization_intercepts_before_provider_execution() -> (
    None
):
    provider_getter = MagicMock()
    handler = MessagesHandler(Settings(), provider_getter=provider_getter)
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        messages=[Message(role="user", content="quota check")],
    )
    optimized = object()

    with patch("api.handlers.messages.try_optimizations", return_value=optimized):
        assert await handler.create(request) is optimized

    provider_getter.assert_not_called()


@pytest.mark.asyncio
async def test_responses_handler_bypasses_message_only_optimizations() -> None:
    provider = FakeProvider()
    handler = ResponsesHandler(Settings(), provider_getter=lambda _: provider)

    with patch(
        "api.handlers.messages.try_optimizations",
        side_effect=AssertionError("Responses must not use message optimizations"),
    ):
        response = await handler.create(
            OpenAIResponsesRequest(
                model="nvidia_nim/test-model",
                input="quota check",
            )
        )

    assert isinstance(response, StreamingResponse)
    body = await _streaming_body_text(response)
    assert "response.completed" in body
    assert provider.requests[0].messages[0].content == "quota check"


@pytest.mark.asyncio
async def test_responses_handler_does_not_apply_safety_classifier_policy() -> None:
    provider = FakeProvider()
    handler = ResponsesHandler(Settings(), provider_getter=lambda _: provider)

    with patch("api.handlers.messages.trace_event") as trace_mock:
        response = await handler.create(
            OpenAIResponsesRequest(
                model="nvidia_nim/test-model",
                input=_CLASSIFIER_USER,
                instructions=_CLASSIFIER_SYSTEM,
            )
        )

        assert isinstance(response, StreamingResponse)
        await _streaming_body_text(response)

    assert provider.preflight_calls[0][1] is True
    assert provider.stream_kwargs[0]["thinking_enabled"] is True
    assert (
        _trace_events(trace_mock, "api.optimization.safety_classifier_no_thinking")
        == []
    )


def test_token_count_handler_routes_and_counts_tokens() -> None:
    handler = TokenCountHandler(
        Settings(),
        token_counter=lambda messages, system, tools: len(messages) + 41,
    )

    response = handler.count(
        TokenCountRequest(
            model="nvidia_nim/test-model",
            messages=[Message(role="user", content="hi")],
        )
    )

    assert response.input_tokens == 42
