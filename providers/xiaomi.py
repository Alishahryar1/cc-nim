from __future__ import annotations
from typing import Any
from providers.openai_compat import OpenAIChatTransport
from providers.base import ProviderConfig
from providers.model_listing import ProviderModelInfo
from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.exceptions import InvalidRequestError

class XiaomiProvider(OpenAIChatTransport):
    """Xiaomi MiMo provider using OpenAI-compatible chat completions."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="XIAOMI",
            base_url=config.base_url,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        effective_thinking = self._is_thinking_enabled(request, thinking_enabled)

        try:
            body = build_base_request_body(
                request,
                reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
                if effective_thinking
                else ReasoningReplayMode.DISABLED,
            )
        except OpenAIConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc

        return body

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Hardcode Xiaomi models to bypass potential list_models API issues."""
        return frozenset([
            ProviderModelInfo(model_id="mimo-v2.5-pro", supports_thinking=True),
            ProviderModelInfo(model_id="mimo-v2.5", supports_thinking=True),
            ProviderModelInfo(model_id="mimo-v2-flash", supports_thinking=False),
        ])
