"""Google AI Studio Gemini provider (OpenAI-compatible chat completions)."""

from copy import deepcopy
from typing import Any

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import (
    NamedEffortReasoning,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
    build_openai_chat_request_body,
)
from free_claude_code.providers.rate_limit import ProviderRateLimiter

from .quirks import apply_gemini_request_quirks

_MAX_TOOL_CALL_EXTRA_CONTENT_CACHE = 4096
_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="GEMINI",
    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
)
_PROFILE = OpenAIChatProfile(
    _REQUEST_POLICY,
    NamedEffortReasoning(
        (
            (ReasoningEffort.MINIMAL, "minimal"),
            (ReasoningEffort.LOW, "low"),
            (ReasoningEffort.MEDIUM, "medium"),
            (ReasoningEffort.HIGH, "high"),
            (ReasoningEffort.XHIGH, "high"),
            (ReasoningEffort.MAX, "high"),
        ),
        disabled_value="none",
    ),
)


class GeminiProvider(OpenAIChatProvider):
    """Gemini API using ``https://generativelanguage.googleapis.com/v1beta/openai/``."""

    def __init__(self, config: ProviderConfig, *, rate_limiter: ProviderRateLimiter):
        super().__init__(
            config,
            profile=_PROFILE,
            rate_limiter=rate_limiter,
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
    ) -> dict:
        return build_openai_chat_request_body(
            request,
            reasoning=reasoning,
            policy=_REQUEST_POLICY,
            postprocessors=(
                lambda body, request_data, policy: apply_gemini_request_quirks(
                    body,
                    request_data,
                    policy,
                    tool_call_extra_content_by_id=self._tool_call_extra_content_by_id,
                ),
                _PROFILE.apply_reasoning,
            ),
        )
