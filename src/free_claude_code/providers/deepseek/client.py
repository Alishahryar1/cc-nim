"""DeepSeek provider implementation (OpenAI-compatible Chat Completions)."""

from typing import Any

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import (
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
    usage_int,
)
from free_claude_code.providers.rate_limit import ProviderRateLimiter

from .compat import build_deepseek_request_body

_PROFILE = OpenAIChatProfile(OpenAIChatRequestPolicy(provider_name="DEEPSEEK"))


class DeepSeekProvider(OpenAIChatProvider):
    """DeepSeek using ``https://api.deepseek.com`` Chat Completions."""

    def __init__(self, config: ProviderConfig, *, rate_limiter: ProviderRateLimiter):
        super().__init__(
            config,
            profile=_PROFILE,
            rate_limiter=rate_limiter,
        )

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        return build_deepseek_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    def _anthropic_usage_fields(self, usage_info: Any) -> dict[str, int]:
        usage_fields: dict[str, int] = {}
        cache_hit_tokens = usage_int(usage_info, "prompt_cache_hit_tokens")
        if cache_hit_tokens is not None:
            usage_fields["cache_read_input_tokens"] = cache_hit_tokens
        cache_miss_tokens = usage_int(usage_info, "prompt_cache_miss_tokens")
        if cache_miss_tokens is not None:
            usage_fields["cache_creation_input_tokens"] = cache_miss_tokens
        return usage_fields
