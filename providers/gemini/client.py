"""Google AI Studio Gemini provider (OpenAI-compatible chat completions)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from providers.base import ProviderConfig
from providers.defaults import GEMINI_DEFAULT_BASE
from providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    OpenAIChatTransport,
    build_openai_chat_request_body,
)

from .quirks import apply_gemini_request_quirks

_MAX_TOOL_CALL_EXTRA_CONTENT_CACHE = 4096
_REQUEST_POLICY = OpenAIChatRequestPolicy(provider_name="GEMINI")


class GeminiProvider(OpenAIChatTransport):
    """Gemini API using ``https://generativelanguage.googleapis.com/v1beta/openai/``."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="GEMINI",
            base_url=config.base_url or GEMINI_DEFAULT_BASE,
            api_key=config.api_key,
        )
        self._tool_call_extra_content_by_id: dict[str, dict[str, Any]] = {}

    def _record_tool_call_extra_content(
        self, tool_call_id: str, extra_content: dict[str, Any]
    ) -> None:
        if (
            tool_call_id not in self._tool_call_extra_content_by_id
            and len(self._tool_call_extra_content_by_id)
            >= _MAX_TOOL_CALL_EXTRA_CONTENT_CACHE
        ):
            self._tool_call_extra_content_by_id.pop(
                next(iter(self._tool_call_extra_content_by_id))
            )
        self._tool_call_extra_content_by_id[tool_call_id] = deepcopy(extra_content)

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return build_openai_chat_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
            policy=_REQUEST_POLICY,
            postprocessors=(
                lambda body, request_data, enabled: apply_gemini_request_quirks(
                    body,
                    request_data,
                    enabled,
                    tool_call_extra_content_by_id=self._tool_call_extra_content_by_id,
                ),
            ),
        )
