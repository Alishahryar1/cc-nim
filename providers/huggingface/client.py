"""Hugging Face Inference Providers implementation."""

from copy import deepcopy
from typing import Any

from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.base import ProviderConfig
from providers.defaults import HUGGINGFACE_DEFAULT_BASE
from providers.exceptions import InvalidRequestError
from providers.transports.openai_chat import OpenAIChatTransport


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
        """Build a Hugging Face Chat Completions body.

        Hugging Face's router documents reasoning controls for supported models
        but not prior-turn reasoning replay through ``messages[].reasoning_content``.
        Keep replay disabled while still allowing new streamed reasoning output
        to map back to Claude thinking in the shared OpenAI-chat stream adapter.
        """
        try:
            body = build_base_request_body(
                request,
                reasoning_replay=ReasoningReplayMode.DISABLED,
            )
        except OpenAIConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc

        request_extra = getattr(request, "extra_body", None)
        if isinstance(request_extra, dict) and request_extra:
            body["extra_body"] = deepcopy(request_extra)

        return body
