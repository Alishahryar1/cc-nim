"""Azure AI Foundry provider (OpenAI-compatible ``/openai/v1`` chat completions).

Azure AI Foundry exposes deployed models (e.g. Moonshot ``Kimi-K2.6``) through an
OpenAI-compatible surface at ``https://<resource>.services.ai.azure.com/openai/v1``.
That route accepts ``Authorization: Bearer <key>`` and a standard
``POST /chat/completions`` body, so the shared OpenAI chat transport drives it
directly. The base URL is resource-specific and supplied via
``AZURE_FOUNDRY_BASE_URL`` (no universal default).
"""

from __future__ import annotations

from typing import Any

from providers.base import ProviderConfig
from providers.exceptions import InvalidRequestError
from providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="AZURE_FOUNDRY",
    include_extra_body=True,
    max_tokens_field="max_tokens",
)


class AzureFoundryProvider(OpenAIChatTransport):
    """Azure AI Foundry models via the OpenAI-compatible v1 surface."""

    def __init__(self, config: ProviderConfig, *, max_tokens: int | None = None):
        base_url = (config.base_url or "").strip()
        if not base_url:
            raise InvalidRequestError(
                "AZURE_FOUNDRY_BASE_URL is not set. Add your Azure AI Foundry "
                "OpenAI-compatible endpoint, e.g. "
                "https://<resource>.services.ai.azure.com/openai/v1"
            )
        super().__init__(
            config,
            provider_name="AZURE_FOUNDRY",
            base_url=base_url,
            api_key=config.api_key,
        )
        self._max_tokens = max_tokens

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_openai_chat_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
        )

    def _prepare_create_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Clamp the requested output budget to ``AZURE_FOUNDRY_MAX_TOKENS``.

        Claude clients routinely request a far larger ``max_tokens`` than an Azure
        deployment allows, which Azure rejects with a 400. When a cap is configured
        we lower any over-budget value (without mutating the caller's body).
        """
        if self._max_tokens is None:
            return body
        clamped = {**body}
        for field_name in ("max_tokens", "max_completion_tokens"):
            value = clamped.get(field_name)
            if isinstance(value, int) and value > self._max_tokens:
                clamped[field_name] = self._max_tokens
        return clamped
