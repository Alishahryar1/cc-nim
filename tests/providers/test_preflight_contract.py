"""The shared OpenAI-chat provider owns explicit request preflight."""

from collections.abc import AsyncIterator

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.openai_chat import OpenAIChatProvider


class RecordingOpenAIProvider(OpenAIChatProvider):
    def __init__(self) -> None:
        self.build_calls: list[tuple[MessagesRequest, ReasoningPolicy]] = []

    def _build_request_body(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> dict:
        self.build_calls.append((request, reasoning))
        return {}


class ProviderWithoutPreflight(BaseProvider):
    async def cleanup(self) -> None:
        return None

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        return frozenset()

    async def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        if False:
            yield ""


def test_provider_base_requires_an_explicit_preflight_implementation() -> None:
    with pytest.raises(TypeError, match="preflight_stream"):
        ProviderWithoutPreflight(
            ProviderConfig(api_key="test", base_url="https://test.invalid")
        )


def test_openai_provider_owns_preflight() -> None:
    assert OpenAIChatProvider.preflight_stream is not BaseProvider.preflight_stream


def test_provider_preflight_calls_builder_and_preserves_policy() -> None:
    provider = RecordingOpenAIProvider()
    request = MessagesRequest(
        model="test-model",
        messages=[Message(role="user", content="hello")],
    )

    provider.preflight_stream(request, reasoning=ReasoningPolicy.off())

    assert provider.build_calls == [(request, ReasoningPolicy.off())]
