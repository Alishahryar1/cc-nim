"""Declarative profiles for ordinary OpenAI-compatible providers."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy

from .base_url import openai_v1_base_url
from .extra_body import (
    validate_extra_body_does_not_override_canonical_fields,
    validate_extra_body_does_not_override_reasoning_fields,
)
from .reasoning import (
    LLAMACPP_REASONING,
    NO_REASONING,
    SPLIT_REASONING_OUTPUT,
    NamedEffortReasoning,
    ReasoningEncoder,
    ReasoningObject,
    ThinkingObjectReasoning,
)
from .request_policy import OpenAIChatPostprocessor, OpenAIChatRequestPolicy

_ALL_EFFORTS = tuple((effort, effort.value) for effort in ReasoningEffort)
_LOW_MEDIUM_HIGH = (
    (ReasoningEffort.MINIMAL, "low"),
    (ReasoningEffort.LOW, "low"),
    (ReasoningEffort.MEDIUM, "medium"),
    (ReasoningEffort.HIGH, "high"),
    (ReasoningEffort.XHIGH, "high"),
    (ReasoningEffort.MAX, "high"),
)
_LOW_TO_MAX = (
    (ReasoningEffort.MINIMAL, "low"),
    (ReasoningEffort.LOW, "low"),
    (ReasoningEffort.MEDIUM, "medium"),
    (ReasoningEffort.HIGH, "high"),
    (ReasoningEffort.XHIGH, "max"),
    (ReasoningEffort.MAX, "max"),
)
_KIMI_CODE_EFFORTS = (
    (ReasoningEffort.MINIMAL, "low"),
    (ReasoningEffort.LOW, "low"),
    (ReasoningEffort.MEDIUM, "high"),
    (ReasoningEffort.HIGH, "high"),
    (ReasoningEffort.XHIGH, "max"),
    (ReasoningEffort.MAX, "max"),
)


@dataclass(frozen=True, slots=True)
class OpenAIChatProfile:
    """Immutable transport and reasoning behavior for one provider."""

    request_policy: OpenAIChatRequestPolicy
    reasoning: ReasoningEncoder
    postprocessors: tuple[OpenAIChatPostprocessor, ...] = ()
    normalize_base_url: bool = False
    reasoning_delta_field: Literal["reasoning_content", "reasoning"] = (
        "reasoning_content"
    )
    user_agent: str | None = None

    @property
    def provider_name(self) -> str:
        return self.request_policy.provider_name

    def base_url(self, configured: str) -> str:
        return openai_v1_base_url(configured) if self.normalize_base_url else configured

    def reasoning_delta(self, delta: Any) -> str | None:
        value = getattr(delta, self.reasoning_delta_field, None)
        return value if isinstance(value, str) else None

    def apply_reasoning(
        self,
        body: dict[str, Any],
        _request: MessagesRequest,
        policy: ReasoningPolicy,
    ) -> None:
        self.reasoning.encode(body, policy)

    @property
    def request_postprocessors(self) -> tuple[OpenAIChatPostprocessor, ...]:
        return (*self.postprocessors, self.apply_reasoning)


def _apply_cohere_request_quirks(
    body: dict[str, Any], request: MessagesRequest, _policy: ReasoningPolicy
) -> None:
    _merge_allowed_cohere_extra_body(body, request.extra_body)


_COHERE_EXTRA_BODY_KEYS = frozenset(
    {
        "frequency_penalty",
        "presence_penalty",
        "response_format",
        "seed",
    }
)


def _merge_allowed_cohere_extra_body(body: dict[str, Any], extra_body: Any) -> None:
    if extra_body in (None, {}):
        return
    if not isinstance(extra_body, Mapping):
        raise InvalidRequestError("Cohere extra_body must be an object when provided.")

    unsupported = sorted(
        str(key) for key in extra_body if key not in _COHERE_EXTRA_BODY_KEYS
    )
    if unsupported:
        raise InvalidRequestError(
            "Cohere extra_body supports only these keys: "
            f"{sorted(_COHERE_EXTRA_BODY_KEYS)}. Unsupported: {unsupported}"
        )
    body.update({str(key): deepcopy(value) for key, value in extra_body.items()})


def _policy(
    provider_name: str,
    replay: ReasoningReplayMode,
    **kwargs: Any,
) -> OpenAIChatRequestPolicy:
    return OpenAIChatRequestPolicy(
        provider_name=provider_name,
        reasoning_replay=replay,
        **kwargs,
    )


OPENAI_CHAT_PROFILES: dict[str, OpenAIChatProfile] = {
    "mistral_codestral": OpenAIChatProfile(
        _policy("CODESTRAL", ReasoningReplayMode.THINK_TAGS),
        NO_REASONING,
    ),
    "opencode": OpenAIChatProfile(
        _policy("OPENCODE", ReasoningReplayMode.THINK_TAGS),
        NO_REASONING,
    ),
    "opencode_go": OpenAIChatProfile(
        _policy("OPENCODE_GO", ReasoningReplayMode.THINK_TAGS),
        NO_REASONING,
    ),
    "vercel": OpenAIChatProfile(
        _policy(
            "VERCEL",
            ReasoningReplayMode.THINK_TAGS,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
        ),
        ReasoningObject(_ALL_EFFORTS),
    ),
    "bedrock": OpenAIChatProfile(
        _policy("BEDROCK", ReasoningReplayMode.THINK_TAGS),
        NO_REASONING,
        normalize_base_url=True,
    ),
    "huggingface": OpenAIChatProfile(
        _policy(
            "HUGGINGFACE",
            ReasoningReplayMode.DISABLED,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
        ),
        NO_REASONING,
    ),
    "cohere": OpenAIChatProfile(
        _policy(
            "COHERE",
            ReasoningReplayMode.REASONING_CONTENT,
            strip_message_names=True,
            unsupported_body_keys=frozenset(
                {
                    "audio",
                    "logit_bias",
                    "metadata",
                    "modalities",
                    "n",
                    "parallel_tool_calls",
                    "prediction",
                    "service_tier",
                    "store",
                    "top_logprobs",
                }
            ),
        ),
        NamedEffortReasoning(
            tuple((effort, "high") for effort in ReasoningEffort),
            disabled_value="none",
            enabled_value="high",
        ),
        postprocessors=(_apply_cohere_request_quirks,),
    ),
    "wafer": OpenAIChatProfile(
        _policy(
            "WAFER",
            ReasoningReplayMode.REASONING_CONTENT,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NamedEffortReasoning(
            _LOW_TO_MAX,
            disabled_value="none",
            enabled_value="high",
        ),
    ),
    "kimi": OpenAIChatProfile(
        _policy(
            "KIMI",
            ReasoningReplayMode.REASONING_CONTENT,
            reject_extra_body_message=(
                "Kimi Chat Completions API does not support caller extra_body on requests."
            ),
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        ThinkingObjectReasoning(
            enabled={"type": "enabled"},
            disabled={"type": "disabled"},
        ),
    ),
    "kimi_code": OpenAIChatProfile(
        _policy(
            "KIMI_CODE",
            ReasoningReplayMode.REASONING_CONTENT,
            reject_extra_body_message=(
                "Kimi Code Chat Completions API does not support caller "
                "extra_body on requests."
            ),
            max_tokens_field="max_completion_tokens",
        ),
        NamedEffortReasoning(
            _KIMI_CODE_EFFORTS,
            disabled_value="none",
            enabled_value="max",
        ),
        user_agent="free-claude-code",
    ),
    "minimax": OpenAIChatProfile(
        _policy(
            "MINIMAX",
            ReasoningReplayMode.REASONING_CONTENT,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
            max_tokens_field="max_completion_tokens",
        ),
        SPLIT_REASONING_OUTPUT,
    ),
    "cerebras": OpenAIChatProfile(
        _policy(
            "CEREBRAS",
            ReasoningReplayMode.THINK_TAGS,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
            max_tokens_field="max_completion_tokens",
        ),
        NamedEffortReasoning(
            _LOW_MEDIUM_HIGH,
            disabled_value="none",
            enabled_value="medium",
        ),
        reasoning_delta_field="reasoning",
    ),
    "groq": OpenAIChatProfile(
        _policy(
            "GROQ",
            ReasoningReplayMode.REASONING_CONTENT,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
            max_tokens_field="max_completion_tokens",
            strip_message_names=True,
            unsupported_body_keys=frozenset({"logprobs", "logit_bias", "top_logprobs"}),
            normalize_n_to_one=True,
        ),
        NamedEffortReasoning(
            _LOW_MEDIUM_HIGH,
            disabled_value="none",
            enabled_value="medium",
        ),
    ),
    "sambanova": OpenAIChatProfile(
        _policy(
            "SAMBANOVA",
            ReasoningReplayMode.REASONING_CONTENT,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_reasoning_fields,
        ),
        NamedEffortReasoning(
            _LOW_MEDIUM_HIGH,
            enabled_value="medium",
        ),
    ),
    "fireworks": OpenAIChatProfile(
        _policy(
            "FIREWORKS",
            ReasoningReplayMode.REASONING_CONTENT,
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_canonical_fields,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NamedEffortReasoning(
            (
                (ReasoningEffort.MINIMAL, "low"),
                (ReasoningEffort.LOW, "low"),
                (ReasoningEffort.MEDIUM, "medium"),
                (ReasoningEffort.HIGH, "high"),
                (ReasoningEffort.XHIGH, "xhigh"),
                (ReasoningEffort.MAX, "max"),
            ),
            disabled_value="none",
            enabled_value="high",
            budget_field="reasoning_effort",
        ),
    ),
    "zai": OpenAIChatProfile(
        _policy(
            "ZAI",
            ReasoningReplayMode.REASONING_CONTENT,
            reject_extra_body_message=(
                "Z.ai Chat Completions API does not support caller extra_body on requests."
            ),
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        ThinkingObjectReasoning(
            enabled={"type": "enabled", "clear_thinking": False},
            disabled={"type": "disabled"},
        ),
    ),
    "ollama_cloud": OpenAIChatProfile(
        _policy(
            "OLLAMA_CLOUD",
            ReasoningReplayMode.REASONING,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NamedEffortReasoning(
            _LOW_TO_MAX,
            disabled_value="none",
            enabled_value="high",
        ),
        reasoning_delta_field="reasoning",
    ),
    "llamacpp": OpenAIChatProfile(
        _policy(
            "LLAMACPP",
            ReasoningReplayMode.THINK_TAGS,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        LLAMACPP_REASONING,
        normalize_base_url=True,
    ),
    "ollama": OpenAIChatProfile(
        _policy(
            "OLLAMA",
            ReasoningReplayMode.REASONING,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        NamedEffortReasoning(
            _LOW_TO_MAX,
            disabled_value="none",
            enabled_value="high",
        ),
        normalize_base_url=True,
        reasoning_delta_field="reasoning",
    ),
}
