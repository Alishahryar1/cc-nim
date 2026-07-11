"""Cohere provider implementation (OpenAI-compatible chat completions)."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.defaults import COHERE_DEFAULT_BASE
from free_claude_code.providers.exceptions import InvalidRequestError
from free_claude_code.providers.rate_limit import ProviderRateLimiter
from free_claude_code.providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

_ALLOWED_EXTRA_BODY_KEYS = frozenset(
    {
        "frequency_penalty",
        "presence_penalty",
        "response_format",
        "seed",
    }
)
_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="COHERE",
    strip_message_names=True,
    unsupported_body_keys=frozenset(
        {
            "audio",
            "logit_bias",
            "metadata",
            "modalities",
            "n",
            "parallel_tool_calls",
            "prediction",
            "service_tier",
            "store",
            "top_logprobs",
        }
    ),
)


class CohereProvider(OpenAIChatTransport):
    """Cohere Compatibility API at ``https://api.cohere.ai/compatibility/v1``."""

    def __init__(self, config: ProviderConfig, *, rate_limiter: ProviderRateLimiter):
        super().__init__(
            config,
            provider_name="COHERE",
            base_url=config.base_url or COHERE_DEFAULT_BASE,
            api_key=config.api_key,
            rate_limiter=rate_limiter,
        )

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        return build_openai_chat_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
            postprocessors=(_apply_cohere_request_quirks,),
        )


def _apply_cohere_request_quirks(
    body: dict[str, Any], request_data: MessagesRequest, thinking_enabled: bool
) -> None:
    _merge_allowed_extra_body(body, request_data.extra_body)
    body["reasoning_effort"] = "high" if thinking_enabled else "none"


def _merge_allowed_extra_body(body: dict[str, Any], extra_body: Any) -> None:
    if extra_body in (None, {}):
        return
    if not isinstance(extra_body, Mapping):
        raise InvalidRequestError("Cohere extra_body must be an object when provided.")

    unsupported = sorted(
        str(key) for key in extra_body if key not in _ALLOWED_EXTRA_BODY_KEYS
    )
    if unsupported:
        raise InvalidRequestError(
            "Cohere extra_body supports only these keys: "
            f"{sorted(_ALLOWED_EXTRA_BODY_KEYS)}. Unsupported: {unsupported}"
        )

    body.update({str(key): deepcopy(value) for key, value in extra_body.items()})
