"""Google AI Studio Gemini provider (OpenAI-compatible chat completions)."""

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.reasoning import ReasoningEffort
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.google_openai import GoogleOpenAIProvider
from free_claude_code.providers.openai_chat import (
    NamedEffortReasoning,
    OpenAIChatProfile,
    OpenAIChatRequestPolicy,
)
from free_claude_code.providers.rate_limit import ProviderRateLimiter

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


class GeminiProvider(GoogleOpenAIProvider):
    """Gemini API using ``https://generativelanguage.googleapis.com/v1beta/openai/``."""

    def __init__(self, config: ProviderConfig, *, rate_limiter: ProviderRateLimiter):
        super().__init__(
            config,
            profile=_PROFILE,
            rate_limiter=rate_limiter,
        )
