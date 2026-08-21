"""Shared Google behavior for OpenAI-compatible Gemini endpoints."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningPolicy,
)
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import (
    OpenAIAsyncCredentialProvider,
    OpenAIChatProfile,
    OpenAIChatProvider,
    build_openai_chat_request_body,
)

from .thought_signatures import apply_google_thought_signatures

_MAX_TOOL_CALL_EXTRA_CONTENT_CACHE = 4096


class GoogleOpenAIProvider(OpenAIChatProvider):
    """Shared thought-signature and request behavior for Google Gemini APIs."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        profile: OpenAIChatProfile,
        admission: ProviderAdmissionController,
        api_key_provider: OpenAIAsyncCredentialProvider | None = None,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            config,
            profile=profile,
            admission=admission,
            api_key_provider=api_key_provider,
            default_headers=default_headers,
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
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> dict[str, Any]:
        return build_openai_chat_request_body(
            request,
            reasoning=reasoning,
            policy=self._profile.request_policy,
            postprocessors=(
                lambda body, _request_data, _policy: apply_google_thought_signatures(
                    body,
                    tool_call_extra_content_by_id=(self._tool_call_extra_content_by_id),
                ),
                *self._profile.request_postprocessors,
            ),
        )
