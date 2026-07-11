"""Provider transport preflight is explicit at every implementation boundary."""

from collections.abc import AsyncIterator

import pytest

from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.transports.anthropic_messages import (
    AnthropicMessagesTransport,
)
from free_claude_code.providers.transports.openai_chat import OpenAIChatTransport


class RecordingOpenAITransport(OpenAIChatTransport):
    def __init__(self) -> None:
        self.build_calls: list[tuple[MessagesRequest, bool | None]] = []

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        self.build_calls.append((request, thinking_enabled))
        return {}


class RecordingAnthropicTransport(AnthropicMessagesTransport):
    def __init__(self) -> None:
        self.build_calls: list[tuple[MessagesRequest, bool | None]] = []

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        self.build_calls.append((request, thinking_enabled))
        return {}


class ProviderWithoutPreflight(BaseProvider):
    async def cleanup(self) -> None:
        return None

    async def list_model_ids(self) -> frozenset[str]:
        return frozenset()

    async def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        if False:
            yield ""


def test_provider_base_requires_an_explicit_preflight_implementation() -> None:
    with pytest.raises(TypeError, match="preflight_stream"):
        ProviderWithoutPreflight(ProviderConfig(api_key="test"))


def test_each_transport_family_owns_preflight() -> None:
    assert OpenAIChatTransport.preflight_stream is not BaseProvider.preflight_stream
    assert (
        AnthropicMessagesTransport.preflight_stream is not BaseProvider.preflight_stream
    )


@pytest.mark.parametrize(
    "transport",
    [RecordingOpenAITransport(), RecordingAnthropicTransport()],
)
def test_transport_preflight_calls_its_builder_and_preserves_false(
    transport: RecordingOpenAITransport | RecordingAnthropicTransport,
) -> None:
    request = MessagesRequest(
        model="test-model",
        messages=[Message(role="user", content="hello")],
    )

    transport.preflight_stream(request, thinking_enabled=False)

    assert transport.build_calls == [(request, False)]
