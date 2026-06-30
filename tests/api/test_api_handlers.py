from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.responses import StreamingResponse

from api.handlers import MessagesHandler, ResponsesHandler, TokenCountHandler
from api.models.anthropic import Message, MessagesRequest, TokenCountRequest
from api.models.openai_responses import OpenAIResponsesRequest
from config.settings import Settings
from providers.base import BaseProvider, ProviderConfig

_CLASSIFIER_SYSTEM = (
    "You are a security monitor. Respond with <block>yes</block> or <block>no</block>."
)
_CLASSIFIER_USER = (
    "<transcript>\nUser: review the repo\nWebFetch https://example.com: fetch\n"
    "</transcript>\n<block> immediately."
)


class FakeProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(ProviderConfig(api_key="test"))
        self.preflight_calls: list[tuple[Any, bool | None]] = []
        self.requests: list[Any] = []
        self.stream_kwargs: list[dict[str, Any]] = []

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
        yield 'event: message_start\ndata: {"type":"message_start"}\n\n'
        yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'


async def _streaming_body_text(response: StreamingResponse) -> str:
    parts: list[str] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            parts.append(chunk.decode("utf-8"))
        else:
            parts.append(str(chunk))
    return "".join(parts)


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

    response = handler.create(request)
    assert isinstance(response, StreamingResponse)

    body = await _streaming_body_text(response)
    assert "message_start" in body
    assert provider.requests[0].model == "test-model"
    assert provider.stream_kwargs[0]["input_tokens"] > 0
    assert provider.stream_kwargs[0]["request_id"].startswith("req_")
    assert provider.stream_kwargs[0]["thinking_enabled"] is True
    assert len(provider.preflight_calls) == 1


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
        response = handler.create(request)
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
        response = handler.create(request)
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
        response = handler.create(request)
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


def test_messages_handler_optimization_intercepts_before_provider_execution() -> None:
    provider_getter = MagicMock()
    handler = MessagesHandler(Settings(), provider_getter=provider_getter)
    request = MessagesRequest(
        model="nvidia_nim/test-model",
        max_tokens=100,
        messages=[Message(role="user", content="quota check")],
    )
    optimized = object()

    with patch("api.handlers.messages.try_optimizations", return_value=optimized):
        assert handler.create(request) is optimized

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
