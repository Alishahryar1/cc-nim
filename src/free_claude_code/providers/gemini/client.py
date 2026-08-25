"""Google AI Studio Gemini provider (OpenAI-compatible chat completions)."""

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.google_openai import (
    GeminiReasoningEncoder,
    GoogleOpenAIProvider,
    validate_google_extra_body,
)
from free_claude_code.providers.openai_chat import (
    OpenAIChatProfile,
    OpenAIChatRequestPolicy,
)

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="GEMINI",
    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    include_extra_body=True,
    extra_body_validator=validate_google_extra_body,
)
_PROFILE = OpenAIChatProfile(
    _REQUEST_POLICY,
    GeminiReasoningEncoder(),
)


class GeminiProvider(GoogleOpenAIProvider):
    """Gemini API using ``https://generativelanguage.googleapis.com/v1beta/openai/``."""

    def __init__(
        self, config: ProviderConfig, *, admission: ProviderAdmissionController
    ):
        super().__init__(
            config,
            profile=_PROFILE,
            admission=admission,
        )
