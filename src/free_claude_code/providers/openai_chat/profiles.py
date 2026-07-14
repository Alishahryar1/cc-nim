"""Declarative profiles for providers with no adapter-specific runtime behavior."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest

from .base_url import openai_v1_base_url
from .extra_body import validate_extra_body_does_not_override_canonical_fields
from .request_policy import OpenAIChatPostprocessor, OpenAIChatRequestPolicy


@dataclass(frozen=True, slots=True)
class OpenAIChatProfile:
    """Immutable behavior differences for one ordinary OpenAI-chat provider."""

    request_policy: OpenAIChatRequestPolicy
    postprocessors: tuple[OpenAIChatPostprocessor, ...] = ()
    normalize_base_url: bool = False
    reasoning_delta_field: Literal["reasoning_content", "reasoning"] = (
        "reasoning_content"
    )

    @property
    def provider_name(self) -> str:
        return self.request_policy.provider_name

    def base_url(self, configured: str) -> str:
        return openai_v1_base_url(configured) if self.normalize_base_url else configured

    def reasoning_delta(self, delta: Any) -> str | None:
        value = getattr(delta, self.reasoning_delta_field, None)
        return value if isinstance(value, str) else None


def _apply_cohere_request_quirks(
    body: dict[str, Any], request: MessagesRequest, thinking_enabled: bool
) -> None:
    _merge_allowed_cohere_extra_body(body, request.extra_body)
    body["reasoning_effort"] = "high" if thinking_enabled else "none"


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


def _apply_kimi_thinking_policy(
    body: dict[str, Any], _request: MessagesRequest, thinking_enabled: bool
) -> None:
    if thinking_enabled:
        return
    extra_body = body.setdefault("extra_body", {})
    if isinstance(extra_body, dict):
        extra_body["thinking"] = {"type": "disabled"}


def _apply_minimax_thinking_policy(
    body: dict[str, Any], _request: MessagesRequest, thinking_enabled: bool
) -> None:
    extra_body = body.setdefault("extra_body", {})
    if not isinstance(extra_body, dict):
        return
    extra_body["reasoning_split"] = True
    extra_body["thinking"] = (
        {"type": "adaptive"} if thinking_enabled else {"type": "disabled"}
    )


def _apply_ollama_thinking_policy(
    body: dict[str, Any], _request: MessagesRequest, thinking_enabled: bool
) -> None:
    body["reasoning_effort"] = "high" if thinking_enabled else "none"


def _apply_wafer_thinking_policy(
    body: dict[str, Any], _request: MessagesRequest, thinking_enabled: bool
) -> None:
    extra_body = body.setdefault("extra_body", {})
    if isinstance(extra_body, dict):
        extra_body["thinking"] = (
            {"type": "enabled"} if thinking_enabled else {"type": "disabled"}
        )


def _apply_zai_thinking_policy(
    body: dict[str, Any], _request: MessagesRequest, thinking_enabled: bool
) -> None:
    extra_body = body.setdefault("extra_body", {})
    if not isinstance(extra_body, dict):
        return
    extra_body["thinking"] = (
        {"type": "enabled", "clear_thinking": False}
        if thinking_enabled
        else {"type": "disabled"}
    )


OPENAI_CHAT_PROFILES: dict[str, OpenAIChatProfile] = {
    "mistral_codestral": OpenAIChatProfile(
        OpenAIChatRequestPolicy(provider_name="CODESTRAL")
    ),
    "opencode": OpenAIChatProfile(OpenAIChatRequestPolicy(provider_name="OPENCODE")),
    "opencode_go": OpenAIChatProfile(
        OpenAIChatRequestPolicy(provider_name="OPENCODE_GO")
    ),
    "vercel": OpenAIChatProfile(
        OpenAIChatRequestPolicy(provider_name="VERCEL", include_extra_body=True)
    ),
    "huggingface": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="HUGGINGFACE",
            include_extra_body=True,
            reasoning_replay=ReasoningReplayMode.DISABLED,
        )
    ),
    "cohere": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="COHERE",
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
        postprocessors=(_apply_cohere_request_quirks,),
    ),
    "wafer": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="WAFER",
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        postprocessors=(_apply_wafer_thinking_policy,),
    ),
    "kimi": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="KIMI",
            reject_extra_body_message=(
                "Kimi Chat Completions API does not support caller extra_body on requests."
            ),
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        postprocessors=(_apply_kimi_thinking_policy,),
    ),
    "minimax": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="MINIMAX",
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
            max_tokens_field="max_completion_tokens",
        ),
        postprocessors=(_apply_minimax_thinking_policy,),
    ),
    "cerebras": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="CEREBRAS",
            include_extra_body=True,
            max_tokens_field="max_completion_tokens",
        )
    ),
    "groq": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="GROQ",
            include_extra_body=True,
            max_tokens_field="max_completion_tokens",
            strip_message_names=True,
            unsupported_body_keys=frozenset({"logprobs", "logit_bias", "top_logprobs"}),
            normalize_n_to_one=True,
        )
    ),
    "sambanova": OpenAIChatProfile(
        OpenAIChatRequestPolicy(provider_name="SAMBANOVA", include_extra_body=True)
    ),
    "fireworks": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="FIREWORKS",
            include_extra_body=True,
            extra_body_validator=validate_extra_body_does_not_override_canonical_fields,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        )
    ),
    "zai": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="ZAI",
            reject_extra_body_message=(
                "Z.ai Chat Completions API does not support caller extra_body on requests."
            ),
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        postprocessors=(_apply_zai_thinking_policy,),
    ),
    "ollama_cloud": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="OLLAMA_CLOUD",
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
            reasoning_replay=ReasoningReplayMode.REASONING,
        ),
        postprocessors=(_apply_ollama_thinking_policy,),
        reasoning_delta_field="reasoning",
    ),
    "llamacpp": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="LLAMACPP",
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        normalize_base_url=True,
    ),
    "ollama": OpenAIChatProfile(
        OpenAIChatRequestPolicy(
            provider_name="OLLAMA",
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        normalize_base_url=True,
    ),
}
