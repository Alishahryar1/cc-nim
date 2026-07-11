"""Kimi (Moonshot) provider using OpenAI-compatible Chat Completions."""

from typing import Any

from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.defaults import KIMI_DEFAULT_BASE
from free_claude_code.providers.rate_limit import ProviderRateLimiter
from free_claude_code.providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="KIMI",
    reject_extra_body_message=(
        "Kimi Chat Completions API does not support caller extra_body on requests."
    ),
    default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
)


class KimiProvider(OpenAIChatTransport):
    """Kimi provider using ``https://api.moonshot.ai/v1/chat/completions``."""

    def __init__(self, config: ProviderConfig, *, rate_limiter: ProviderRateLimiter):
        super().__init__(
            config,
            provider_name="KIMI",
            base_url=config.base_url or KIMI_DEFAULT_BASE,
            api_key=config.api_key,
            rate_limiter=rate_limiter,
        )

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        effective_thinking_enabled = self._is_thinking_enabled(
            request, thinking_enabled
        )
        return build_openai_chat_request_body(
            request,
            thinking_enabled=effective_thinking_enabled,
            policy=_REQUEST_POLICY,
            postprocessors=(_apply_kimi_thinking_policy,),
        )


def _apply_kimi_thinking_policy(
    body: dict[str, Any], _request: MessagesRequest, thinking_enabled: bool
) -> None:
    if thinking_enabled:
        return
    extra_body = body.setdefault("extra_body", {})
    if isinstance(extra_body, dict):
        extra_body["thinking"] = {"type": "disabled"}
