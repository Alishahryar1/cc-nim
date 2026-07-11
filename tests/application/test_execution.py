"""Application-owned provider execution contracts."""

from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.routing import ResolvedModel, RoutedMessagesRequest
from free_claude_code.config.provider_catalog import ProviderCapabilities
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.async_iterators import AsyncCloseable


class FakeProvider:
    def __init__(self) -> None:
        self.preflight_calls: list[tuple[MessagesRequest, bool]] = []
        self.stream_calls: list[dict[str, object]] = []
        self.stream_close_calls = 0

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        thinking_enabled: bool,
    ) -> None:
        self.preflight_calls.append((request, thinking_enabled))

    async def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        self.stream_calls.append(
            {
                "request": request,
                "input_tokens": input_tokens,
                "request_id": request_id,
                "thinking_enabled": thinking_enabled,
            }
        )
        try:
            yield "event: message_stop\ndata: {}\n\n"
        finally:
            self.stream_close_calls += 1


class FailingPreflightProvider(FakeProvider):
    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        thinking_enabled: bool,
    ) -> None:
        raise ValueError("invalid provider request")


class FailingStreamConstructionProvider(FakeProvider):
    def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        raise RuntimeError("stream construction failed")


def _routed_request() -> RoutedMessagesRequest:
    request = MessagesRequest(
        model="provider-model",
        messages=[Message(role="user", content="hello")],
    )
    return RoutedMessagesRequest(
        request=request,
        resolved=ResolvedModel(
            original_model="gateway-model",
            provider_id="provider",
            provider_model="provider-model",
            provider_model_ref="provider/provider-model",
            thinking_enabled=True,
            capabilities=ProviderCapabilities(),
        ),
    )


@pytest.mark.asyncio
async def test_executor_uses_structural_provider_port_and_preflights_eagerly() -> None:
    provider = FakeProvider()
    routed = _routed_request()
    request = routed.request
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        token_counter=lambda _messages, _system, _tools: 17,
    )

    stream = executor.stream(
        routed,
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload=request.model_dump(),
        request_id="req_application",
    )

    assert provider.preflight_calls == [(request, True)]
    assert [chunk async for chunk in stream] == ["event: message_stop\ndata: {}\n\n"]
    assert provider.stream_calls == [
        {
            "request": request,
            "input_tokens": 17,
            "request_id": "req_application",
            "thinking_enabled": True,
        }
    ]
    assert provider.stream_close_calls == 1


@pytest.mark.asyncio
async def test_closing_executor_stream_closes_provider_stream_once() -> None:
    provider = FakeProvider()
    routed = _routed_request()
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        token_counter=lambda _messages, _system, _tools: 17,
    )
    stream = executor.stream(
        routed,
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_early_close",
    )

    assert await anext(stream) == "event: message_stop\ndata: {}\n\n"
    assert isinstance(stream, AsyncCloseable)
    await stream.aclose()

    assert provider.stream_close_calls == 1


@pytest.mark.asyncio
async def test_stream_construction_failure_remains_deferred_to_iteration() -> None:
    provider = FailingStreamConstructionProvider()
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        token_counter=lambda _messages, _system, _tools: 17,
    )

    stream = executor.stream(
        _routed_request(),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_deferred_construction",
    )

    with pytest.raises(RuntimeError, match="stream construction failed"):
        await anext(stream)


def test_executor_preflight_failure_stays_before_token_count_and_stream() -> None:
    provider = FailingPreflightProvider()
    token_counter = MagicMock(return_value=17)
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        token_counter=token_counter,
    )

    with pytest.raises(ValueError, match="invalid provider request"):
        executor.stream(
            _routed_request(),
            wire_api="messages",
            raw_log_label="FULL_PAYLOAD",
            raw_log_payload={},
            request_id="req_application",
        )

    token_counter.assert_not_called()
    assert provider.stream_calls == []
