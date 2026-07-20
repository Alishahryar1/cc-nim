"""Provider test helpers with explicit admission ownership."""

from free_claude_code.application.reasoning import client_reasoning_policy
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import (
    OpenAIChatProvider,
    create_openai_chat_provider,
)

REASONING_DEFAULT = ReasoningPolicy.provider_default()
REASONING_ON = ReasoningPolicy.on()
REASONING_OFF = ReasoningPolicy.off()


def immediate_admission(
    *,
    provider_name: str = "TEST",
    max_attempts: int = 5,
) -> ProviderAdmissionController:
    """Return a real controller with deterministic zero-delay recovery."""
    return ProviderAdmissionController(
        provider_name=provider_name,
        rate_limit=1_000_000,
        rate_window=1.0,
        max_concurrency=1_000,
        max_attempts=max_attempts,
        base_delay=0.0,
        max_delay=0.0,
        jitter=0.0,
    )


def profiled_provider(
    provider_id: str,
    config: ProviderConfig,
    *,
    admission: ProviderAdmissionController | None = None,
) -> OpenAIChatProvider:
    """Construct one declarative provider for a focused behavior test."""
    return create_openai_chat_provider(
        provider_id,
        config,
        admission or immediate_admission(provider_name=provider_id),
    )


def reasoning_for(request: MessagesRequest) -> ReasoningPolicy:
    """Resolve provider-test input through the production client boundary."""

    return client_reasoning_policy(request)
