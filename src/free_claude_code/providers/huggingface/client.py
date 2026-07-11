"""Hugging Face Inference Providers implementation."""

from copy import deepcopy

from free_claude_code.core.anthropic import ReasoningReplayMode, build_base_request_body
from free_claude_code.core.anthropic.conversion import OpenAIConversionError
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.defaults import HUGGINGFACE_DEFAULT_BASE
from free_claude_code.providers.exceptions import InvalidRequestError
from free_claude_code.providers.rate_limit import ProviderRateLimiter
from free_claude_code.providers.transports.openai_chat import OpenAIChatTransport


class HuggingFaceProvider(OpenAIChatTransport):
    """Hugging Face Inference Providers router at ``https://router.huggingface.co/v1``."""

    def __init__(self, config: ProviderConfig, *, rate_limiter: ProviderRateLimiter):
        super().__init__(
            config,
            provider_name="HUGGINGFACE",
            base_url=config.base_url or HUGGINGFACE_DEFAULT_BASE,
            api_key=config.api_key,
            rate_limiter=rate_limiter,
        )

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
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

        request_extra = request.extra_body
        if isinstance(request_extra, dict) and request_extra:
            body["extra_body"] = deepcopy(request_extra)

        return body
