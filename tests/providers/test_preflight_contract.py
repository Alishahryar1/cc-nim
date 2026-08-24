"""The shared OpenAI-chat provider owns explicit request preflight."""

from collections.abc import AsyncIterator

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.inference import (
    InferenceEvent,
    InferenceRequest,
    InferenceStreamLedger,
)
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.request_factory import canonical_request
from tests.providers.support import make_provider_config


class RecordingOpenAIProvider(OpenAIChatProvider):
    def __init__(self) -> None:
        self.build_calls: list[tuple[InferenceRequest, str, ReasoningPolicy]] = []

    def _build_request_body(
        self,
        request: InferenceRequest,
        *,
        provider_model: str,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> dict:
        self.build_calls.append((request, provider_model, reasoning))
        return {}


class ProviderWithoutPreflight(BaseProvider):
    async def cleanup(self) -> None:
        return None

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        return frozenset()

    async def stream_response(
        self,
        request: InferenceRequest,
        input_tokens: int = 0,
        *,
        provider_model: str,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[InferenceEvent]:
        if False:
            yield InferenceStreamLedger("unused", "unused").start_response()


def test_provider_base_requires_an_explicit_preflight_implementation() -> None:
    with pytest.raises(TypeError, match="preflight_stream"):
        ProviderWithoutPreflight(
            make_provider_config(api_key="test", base_url="https://test.invalid")
        )


def test_openai_provider_owns_preflight() -> None:
    assert OpenAIChatProvider.preflight_stream is not BaseProvider.preflight_stream


def test_provider_preflight_calls_builder_and_preserves_policy() -> None:
    provider = RecordingOpenAIProvider()
    request = MessagesRequest(
        model="test-model",
        messages=[Message(role="user", content="hello")],
    )

    provider.preflight_stream(
        canonical_request(request),
        reasoning=ReasoningPolicy.off(),
        provider_model=(request).model,
    )

    assert provider.build_calls == [
        (canonical_request(request), request.model, ReasoningPolicy.off())
    ]
