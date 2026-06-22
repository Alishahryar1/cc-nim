"""ClaudeProxyService._stream_with_failover: transparent retry on pre-stream failure."""

from collections.abc import AsyncIterator

import pytest

from api.model_router import ModelRouter, ResolvedModel, RoutedMessagesRequest
from api.models.anthropic import Message, MessagesRequest
from api.services import ClaudeProxyService
from config.settings import Settings
from providers.exceptions import PreStreamProviderError


class FakeProvider:
    """Provider double: optionally fails pre-stream when failover is requested."""

    def __init__(self, fail_prestream: bool):
        self._fail_prestream = fail_prestream
        self.calls: list[tuple[str, bool]] = []

    async def stream_response(
        self,
        request,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
        raise_on_prestream_error: bool = False,
    ) -> AsyncIterator[str]:
        self.calls.append((request.model, raise_on_prestream_error))
        if raise_on_prestream_error and self._fail_prestream:
            raise PreStreamProviderError("simulated pre-stream failure")
        for event in (
            "event: message_start\n",
            f'data: {{"model":"{request.model}"}}\n',
            "\n",
        ):
            yield event


def _routed() -> RoutedMessagesRequest:
    primary = ResolvedModel(
        original_model="claude-3-5-haiku-20241022",
        provider_id="open_router",
        provider_model="openai/gpt-oss-120b",
        provider_model_ref="open_router/openai/gpt-oss-120b",
        thinking_enabled=False,
    )
    fallback = ResolvedModel(
        original_model="claude-3-5-haiku-20241022",
        provider_id="open_router",
        provider_model="z-ai/glm-4.7-flash",
        provider_model_ref="open_router/z-ai/glm-4.7-flash",
        thinking_enabled=False,
    )
    return RoutedMessagesRequest(
        request=MessagesRequest(
            model="openai/gpt-oss-120b",
            max_tokens=100,
            messages=[Message(role="user", content="hi")],
        ),
        resolved=primary,
        fallback_request=MessagesRequest(
            model="z-ai/glm-4.7-flash",
            max_tokens=100,
            messages=[Message(role="user", content="hi")],
        ),
        fallback_resolved=fallback,
    )


def _service(provider: FakeProvider) -> ClaudeProxyService:
    settings = Settings()
    return ClaudeProxyService(
        settings=settings,
        provider_getter=lambda _pid: provider,
        model_router=ModelRouter(settings),
    )


@pytest.mark.asyncio
async def test_failover_switches_to_fallback_on_prestream_error():
    provider = FakeProvider(fail_prestream=True)
    service = _service(provider)

    events = [
        e
        async for e in service._stream_with_failover(
            _routed(), request_id="req_test", input_tokens=0
        )
    ]

    # Primary attempted with the failover flag, then fallback streamed.
    assert provider.calls == [
        ("openai/gpt-oss-120b", True),
        ("z-ai/glm-4.7-flash", False),
    ]
    assert any('"model":"z-ai/glm-4.7-flash"' in e for e in events)
    assert not any('"model":"openai/gpt-oss-120b"' in e for e in events)


@pytest.mark.asyncio
async def test_failover_not_triggered_when_primary_succeeds():
    provider = FakeProvider(fail_prestream=False)
    service = _service(provider)

    events = [
        e
        async for e in service._stream_with_failover(
            _routed(), request_id="req_test", input_tokens=0
        )
    ]

    # Only the primary was called; no fallback attempt.
    assert provider.calls == [("openai/gpt-oss-120b", True)]
    assert any('"model":"openai/gpt-oss-120b"' in e for e in events)
