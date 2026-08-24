"""Request-body policy for OpenAI-compatible chat providers."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.inference import (
    InferenceRequest,
    ReplayCompatibilityScope,
    thaw_json_object,
)
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.openai_compat import OpenAIToolNameCodec

from .request_codec import (
    OpenAIConversionError,
    ReasoningReplayMode,
    build_base_request_body,
)

MaxTokensField = Literal["max_tokens", "max_completion_tokens"]
OpenAIChatPostprocessor = Callable[
    [JsonObject, InferenceRequest, ReasoningPolicy], None
]
ExtraBodyValidator = Callable[[JsonObject], None]


@dataclass(frozen=True, slots=True)
class OpenAIChatRequestPolicy:
    """Provider policy for canonical-to-OpenAI Chat request encoding."""

    provider_name: str
    reasoning_replay: ReasoningReplayMode
    include_extra_body: bool = False
    postprocessor_consumes_extra_body: bool = False
    extra_body_validator: ExtraBodyValidator | None = None
    reject_extra_body_message: str | None = None
    default_max_tokens: int | None = None
    max_tokens_field: MaxTokensField = "max_tokens"
    strip_message_names: bool = False
    unsupported_body_keys: frozenset[str] = field(default_factory=frozenset)
    normalize_n_to_one: bool = False


def build_openai_chat_request_body(
    request_data: InferenceRequest,
    *,
    provider_model: str,
    reasoning: ReasoningPolicy,
    policy: OpenAIChatRequestPolicy,
    tool_names: OpenAIToolNameCodec,
    replay_scope: ReplayCompatibilityScope | None,
    postprocessors: Iterable[OpenAIChatPostprocessor] = (),
) -> JsonObject:
    """Build an OpenAI-compatible Chat request body from canonical input."""
    logger.debug(
        "{}_REQUEST: conversion start model={} msgs={}",
        policy.provider_name,
        provider_model,
        request_data.message_count,
    )
    try:
        body = build_base_request_body(
            request_data,
            provider_model=provider_model,
            tool_names=tool_names,
            replay_scope=replay_scope,
            default_max_tokens=policy.default_max_tokens,
            reasoning_replay=policy.reasoning_replay,
        )
    except OpenAIConversionError as exc:
        raise InvalidRequestError(str(exc)) from exc

    extension = request_data.openai_chat_extension
    if extension is not None and extension.extra_body:
        if policy.reject_extra_body_message:
            raise InvalidRequestError(policy.reject_extra_body_message)
        if policy.include_extra_body:
            extra_body = thaw_json_object(extension.extra_body)
            if policy.extra_body_validator is not None:
                try:
                    policy.extra_body_validator(extra_body)
                except ValueError as exc:
                    raise InvalidRequestError(str(exc)) from exc
            body["extra_body"] = extra_body
        elif not policy.postprocessor_consumes_extra_body:
            raise InvalidRequestError(
                f"{policy.provider_name} does not support caller extra_body."
            )

    _apply_common_openai_chat_policy(body, policy)

    for postprocess in postprocessors:
        postprocess(body, request_data, reasoning)

    logger.debug(
        "{}_REQUEST: conversion done model={} msgs={} tools={}",
        policy.provider_name,
        body.get("model"),
        _sequence_length(body.get("messages")),
        _sequence_length(body.get("tools")),
    )
    return body


def _sequence_length(value: object) -> int:
    return len(value) if isinstance(value, list | tuple) else 0


def _apply_common_openai_chat_policy(
    body: JsonObject, policy: OpenAIChatRequestPolicy
) -> None:
    if policy.strip_message_names:
        _strip_message_names(body.get("messages"))

    for key in policy.unsupported_body_keys:
        body.pop(key, None)

    if policy.max_tokens_field == "max_completion_tokens":
        _normalize_max_completion_tokens(body)

    if policy.normalize_n_to_one and body.get("n") is not None:
        body["n"] = 1


def _strip_message_names(messages: JsonValue) -> None:
    if not isinstance(messages, list):
        return
    for message in messages:
        if isinstance(message, dict):
            message.pop("name", None)


def _normalize_max_completion_tokens(body: JsonObject) -> None:
    if "max_completion_tokens" in body:
        body.pop("max_tokens", None)
        return
    if "max_tokens" in body and body["max_tokens"] is not None:
        body["max_completion_tokens"] = body.pop("max_tokens")
