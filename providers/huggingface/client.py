"""Hugging Face Inference Providers implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import HUGGINGFACE_DEFAULT_BASE
from providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="HUGGINGFACE",
    include_extra_body=True,
)


class HuggingFaceProvider(OpenAIChatTransport):
    """Hugging Face Inference Providers router at ``https://router.huggingface.co/v1``."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="HUGGINGFACE",
            base_url=config.base_url or HUGGINGFACE_DEFAULT_BASE,
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
