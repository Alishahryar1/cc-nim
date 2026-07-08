"""Z.ai provider implementation (OpenAI-compatible Coding Plan Chat Completions)."""

from typing import Any

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from providers.base import ProviderConfig
from providers.defaults import ZAI_DEFAULT_BASE
from providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="ZAI",
    reject_extra_body_message=(
        "Z.ai Chat Completions API does not support caller extra_body on requests."
    ),
    default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
)


class ZaiProvider(OpenAIChatTransport):
    """Z.ai Coding Plan via ``https://api.z.ai/api/coding/paas/v4``."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="ZAI",
            base_url=config.base_url or ZAI_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        effective_thinking_enabled = self._is_thinking_enabled(
            request, thinking_enabled
        )
        return build_openai_chat_request_body(
            request,
            thinking_enabled=effective_thinking_enabled,
            policy=_REQUEST_POLICY,
            postprocessors=(_apply_zai_thinking_policy,),
        )


def _apply_zai_thinking_policy(
    body: dict[str, Any], _request: Any, thinking_enabled: bool
) -> None:
    extra_body = body.setdefault("extra_body", {})
    if not isinstance(extra_body, dict):
        return
    extra_body["thinking"] = (
        {"type": "enabled", "clear_thinking": False}
        if thinking_enabled
        else {"type": "disabled"}
    )
