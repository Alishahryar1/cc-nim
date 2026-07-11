"""Mistral La Plateforme provider implementation (OpenAI-compatible chat completions)."""

from typing import Any

from loguru import logger

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.defaults import MISTRAL_DEFAULT_BASE
from free_claude_code.providers.rate_limit import ProviderRateLimiter
from free_claude_code.providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

from .reasoning import (
    apply_mistral_reasoning_request_shape,
    clone_body_without_mistral_reasoning,
    is_mistral_reasoning_rejection,
    normalize_mistral_stream,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(provider_name="MISTRAL")


class MistralProvider(OpenAIChatTransport):
    """Mistral API using ``https://api.mistral.ai/v1/chat/completions``."""

    def __init__(self, config: ProviderConfig, *, rate_limiter: ProviderRateLimiter):
        super().__init__(
            config,
            provider_name="MISTRAL",
            base_url=config.base_url or MISTRAL_DEFAULT_BASE,
            api_key=config.api_key,
            rate_limiter=rate_limiter,
        )

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        effective_thinking_enabled = self._is_thinking_enabled(
            request, thinking_enabled
        )
        body = build_openai_chat_request_body(
            request,
            thinking_enabled=effective_thinking_enabled,
            policy=_REQUEST_POLICY,
        )
        apply_mistral_reasoning_request_shape(
            body, thinking_enabled=effective_thinking_enabled
        )
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
