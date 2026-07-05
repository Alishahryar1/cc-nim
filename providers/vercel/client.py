"""Vercel AI Gateway provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import VERCEL_AI_GATEWAY_DEFAULT_BASE
from providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="VERCEL",
    include_extra_body=True,
)


class VercelProvider(OpenAIChatTransport):
    """Vercel AI Gateway at ``https://ai-gateway.vercel.sh/v1``."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="VERCEL",
            base_url=config.base_url or VERCEL_AI_GATEWAY_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_openai_chat_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )
