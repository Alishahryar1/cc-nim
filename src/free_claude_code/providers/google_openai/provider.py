"""Shared Google behavior for OpenAI-compatible Gemini endpoints."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningControl,
    ReasoningEffort,
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

from .quirks import (
    apply_google_request_quirks,
    clear_google_thinking_config,
    google_thinking_config,
)

_MAX_TOOL_CALL_EXTRA_CONTENT_CACHE = 4096

_GEMINI_EFFORT_VALUES = {
    ReasoningEffort.MINIMAL: "minimal",
    ReasoningEffort.LOW: "low",
    ReasoningEffort.MEDIUM: "medium",
    ReasoningEffort.HIGH: "high",
    ReasoningEffort.XHIGH: "high",
    ReasoningEffort.MAX: "high",
}


@dataclass(frozen=True, slots=True)
class GoogleThinkingBudgetReasoning:
    """Encode FCC reasoning intent in Google's model-neutral thinking budget."""

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        if policy.control is ReasoningControl.OFF:
            thinking = google_thinking_config(body)
            thinking["thinking_budget"] = 0
            thinking["include_thoughts"] = False
            return
        budget = policy.numeric_budget_tokens
        if budget is None:
            return
        thinking = google_thinking_config(body)
        thinking.setdefault("thinking_budget", budget)
        thinking.setdefault("include_thoughts", True)


@dataclass(frozen=True, slots=True)
class GeminiReasoning:
    """Encode Gemini with either reasoning_effort or thinking_config, never both."""

    disabled_value: str = "none"
    _thinking_budget: GoogleThinkingBudgetReasoning = field(
        default_factory=GoogleThinkingBudgetReasoning
    )

    def encode(self, body: dict[str, Any], policy: ReasoningPolicy) -> None:
        if policy.control is ReasoningControl.OFF:
            body["reasoning_effort"] = self.disabled_value
            clear_google_thinking_config(body)
            return

        if policy.budget_tokens is not None:
            body.pop("reasoning_effort", None)
            self._thinking_budget.encode(body, policy)
            return

        effort = _GEMINI_EFFORT_VALUES.get(policy.effort)
        if effort is not None:
            body["reasoning_effort"] = effort
            clear_google_thinking_config(body)
            return

        body.pop("reasoning_effort", None)
        if policy.requests_reasoning:
            self._thinking_budget.encode(body, policy)


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
                lambda body, request_data, policy: apply_google_request_quirks(
                    body,
                    request_data,
                    policy,
                    tool_call_extra_content_by_id=(self._tool_call_extra_content_by_id),
                ),
                self._profile.apply_reasoning,
            ),
        )
