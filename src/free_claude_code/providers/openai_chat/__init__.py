"""OpenAI-compatible provider family."""

from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig

from .base_url import openai_v1_base_url
from .extra_body import (
    validate_extra_body_does_not_override_canonical_fields,
    validate_extra_body_does_not_override_reasoning_fields,
)
from .profiles import OPENAI_CHAT_PROFILES, OpenAIChatProfile
from .provider import OpenAIAsyncCredentialProvider, OpenAIChatProvider
from .reasoning import (
    NO_REASONING,
    ChatTemplateReasoning,
    NamedEffortReasoning,
    ReasoningObject,
)
from .request_policy import OpenAIChatRequestPolicy, build_openai_chat_request_body
from .usage import usage_int


def create_openai_chat_provider(
    provider_id: str,
    config: ProviderConfig,
    admission: ProviderAdmissionController,
) -> OpenAIChatProvider:
    """Construct one profile-driven provider."""
    profile = OPENAI_CHAT_PROFILES.get(provider_id)
    if profile is None:
        raise KeyError(f"No declarative OpenAI-chat profile for {provider_id!r}")
    return OpenAIChatProvider(
        config,
        profile=profile,
        admission=admission,
        default_headers=(
            {"User-Agent": profile.user_agent} if profile.user_agent else None
        ),
    )


__all__ = [
    "NO_REASONING",
    "OPENAI_CHAT_PROFILES",
    "ChatTemplateReasoning",
    "NamedEffortReasoning",
    "OpenAIAsyncCredentialProvider",
    "OpenAIChatProfile",
    "OpenAIChatProvider",
    "OpenAIChatRequestPolicy",
    "ReasoningObject",
    "build_openai_chat_request_body",
    "create_openai_chat_provider",
    "openai_v1_base_url",
    "usage_int",
    "validate_extra_body_does_not_override_canonical_fields",
    "validate_extra_body_does_not_override_reasoning_fields",
]
