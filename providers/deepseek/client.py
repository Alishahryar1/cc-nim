"""DeepSeek provider implementation (OpenAI-compatible Chat Completions)."""

from collections.abc import Mapping
from typing import Any

from providers.base import ProviderConfig
from providers.defaults import DEEPSEEK_DEFAULT_BASE
from providers.transports.openai_chat import OpenAIChatTransport

from .compat import build_deepseek_request_body


class DeepSeekProvider(OpenAIChatTransport):
    """DeepSeek using ``https://api.deepseek.com`` Chat Completions."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="DEEPSEEK",
            base_url=config.base_url or DEEPSEEK_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_deepseek_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    def _anthropic_usage_fields(self, usage_info: Any) -> dict[str, int]:
        usage_fields: dict[str, int] = {}
        cache_hit_tokens = _usage_int(usage_info, "prompt_cache_hit_tokens")
        if cache_hit_tokens is not None:
            usage_fields["cache_read_input_tokens"] = cache_hit_tokens
        cache_miss_tokens = _usage_int(usage_info, "prompt_cache_miss_tokens")
        if cache_miss_tokens is not None:
            usage_fields["cache_creation_input_tokens"] = cache_miss_tokens
        return usage_fields


def _usage_int(usage_info: Any, key: str) -> int | None:
    if usage_info is None:
        return None
    if isinstance(usage_info, Mapping):
        value = usage_info.get(key)
    else:
        value = getattr(usage_info, key, None)
        if value is None:
            extra = getattr(usage_info, "model_extra", None)
            if isinstance(extra, Mapping):
                value = extra.get(key)
    return value if isinstance(value, int) else None
