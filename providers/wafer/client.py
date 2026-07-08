"""Wafer provider implementation (OpenAI-compatible Chat Completions)."""

from typing import Any

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from providers.base import ProviderConfig
from providers.defaults import WAFER_DEFAULT_BASE
from providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="WAFER",
    default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
)


class WaferProvider(OpenAIChatTransport):
    """Wafer using ``https://pass.wafer.ai/v1/chat/completions``."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="WAFER",
            base_url=config.base_url or WAFER_DEFAULT_BASE,
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
            postprocessors=(_apply_wafer_thinking_policy,),
        )


def _apply_wafer_thinking_policy(
    body: dict[str, Any], _request: Any, thinking_enabled: bool
) -> None:
    extra_body = body.setdefault("extra_body", {})
    if isinstance(extra_body, dict):
        extra_body["thinking"] = (
            {"type": "enabled"} if thinking_enabled else {"type": "disabled"}
        )
