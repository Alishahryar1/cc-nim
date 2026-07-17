"""Mistral La Plateforme provider implementation (OpenAI-compatible chat completions)."""

from typing import Any

from loguru import logger

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
    build_openai_chat_request_body,
)
from free_claude_code.providers.rate_limit import ProviderRateLimiter

from .reasoning import (
    apply_mistral_reasoning_request_shape,
    clone_body_without_mistral_reasoning,
    is_mistral_reasoning_rejection,
    normalize_mistral_stream,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="MISTRAL",
    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
)
_PROFILE = OpenAIChatProfile(_REQUEST_POLICY, NO_REASONING)


class MistralProvider(OpenAIChatProvider):
    """Mistral API using ``https://api.mistral.ai/v1/chat/completions``."""

    def __init__(self, config: ProviderConfig, *, rate_limiter: ProviderRateLimiter):
        super().__init__(
            config,
            profile=_PROFILE,
            rate_limiter=rate_limiter,
        )

    def _build_request_body(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> dict:
        body = build_openai_chat_request_body(
            request,
            reasoning=reasoning,
            policy=_REQUEST_POLICY,
        )
        apply_mistral_reasoning_request_shape(body, reasoning=reasoning)
        return body

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Retry once without Mistral reasoning fields when a model rejects them."""
        if not is_mistral_reasoning_rejection(error):
            return None
        retry_body = clone_body_without_mistral_reasoning(body)
        if retry_body is None:
            return None
        logger.warning(
            "MISTRAL_STREAM: retrying without reasoning after upstream rejection"
        )
        return retry_body

    async def _create_stream(self, body: dict) -> tuple[Any, dict]:
        stream, final_body = await super()._create_stream(body)
        return normalize_mistral_stream(stream), final_body
