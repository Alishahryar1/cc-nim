"""OpenRouter provider implementation."""

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.reasoning import ReasoningEffort
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.model_listing import extract_tool_capable_model_infos
from free_claude_code.providers.openai_chat import (
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
    ReasoningObject,
    ReasoningReplayMode,
    validate_extra_body_does_not_override_canonical_fields,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="OPENROUTER",
    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    include_extra_body=True,
    extra_body_validator=validate_extra_body_does_not_override_canonical_fields,
    default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
)


class OpenRouterProvider(OpenAIChatProvider):
    """OpenRouter provider using the OpenAI-compatible Chat Completions API."""

    def __init__(
        self, config: ProviderConfig, *, admission: ProviderAdmissionController
    ):
        super().__init__(
            config,
            profile=_PROFILE,
            admission=admission,
        )

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Advertise OpenRouter tool models with reasoning capability metadata."""
        payload = await self._list_models_payload()
        return extract_tool_capable_model_infos(
            payload, provider_name=self._provider_name
        )


_PROFILE = OpenAIChatProfile(
    _REQUEST_POLICY,
    ReasoningObject(tuple((effort, effort.value) for effort in ReasoningEffort)),
    structured_reasoning_details=True,
)
