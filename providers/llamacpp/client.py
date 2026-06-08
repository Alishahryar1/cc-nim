from typing import Any

from core.anthropic import ReasoningReplayMode, build_base_request_body
from providers.base import ProviderConfig
from providers.defaults import LLAMACPP_DEFAULT_BASE
from providers.openai_compat import OpenAIChatTransport


class LlamaCppProvider(OpenAIChatTransport):
    """Llama.cpp provider using OpenAI-compatible chat completions endpoint."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="LLAMACPP",
            base_url=config.base_url or LLAMACPP_DEFAULT_BASE,
            api_key=config.api_key or "llamacpp",
        )

    def _prepare_create_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """Strip fields that Llama.cpp might reject with 400 Bad Request."""
        clean = dict(body)
        clean.pop("reasoning_effort", None)
        return clean

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        replay = (
            ReasoningReplayMode.THINK_TAGS
            if self._is_thinking_enabled(request, thinking_enabled)
            else ReasoningReplayMode.DISABLED
        )
        return build_base_request_body(request, reasoning_replay=replay)
