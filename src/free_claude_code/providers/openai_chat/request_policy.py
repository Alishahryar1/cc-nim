"""Request-body policy for OpenAI-compatible chat providers."""

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic import ReasoningReplayMode, build_base_request_body
from free_claude_code.core.anthropic.conversion import OpenAIConversionError
from free_claude_code.core.anthropic.models import MessagesRequest

MaxTokensField = Literal["max_tokens", "max_completion_tokens"]
OpenAIChatPostprocessor = Callable[[dict[str, Any], MessagesRequest, bool], None]
ExtraBodyValidator = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class OpenAIChatRequestPolicy:
    """Provider policy for Anthropic-to-OpenAI chat request conversion."""

    provider_name: str
    include_extra_body: bool = False
    extra_body_validator: ExtraBodyValidator | None = None
    reject_extra_body_message: str | None = None
    default_max_tokens: int | None = None
    max_tokens_field: MaxTokensField = "max_tokens"
    reasoning_replay: ReasoningReplayMode | None = None
    strip_message_names: bool = False
    unsupported_body_keys: frozenset[str] = field(default_factory=frozenset)
    normalize_n_to_one: bool = False


def build_openai_chat_request_body(
    request_data: MessagesRequest,
    *,
    thinking_enabled: bool,
    policy: OpenAIChatRequestPolicy,
    postprocessors: Iterable[OpenAIChatPostprocessor] = (),
) -> dict[str, Any]:
    """Build an OpenAI-compatible chat request body from an Anthropic request."""
    logger.debug(
        "{}_REQUEST: conversion start model={} msgs={}",
        policy.provider_name,
        request_data.model,
        len(request_data.messages),
    )
    try:
        if not thinking_enabled:
            reasoning_replay = ReasoningReplayMode.DISABLED
        else:
            reasoning_replay = (
                policy.reasoning_replay or ReasoningReplayMode.REASONING_CONTENT
            )
        body = build_base_request_body(
            request_data,
            default_max_tokens=policy.default_max_tokens,
            reasoning_replay=reasoning_replay,
        )
    except OpenAIConversionError as exc:
        raise InvalidRequestError(str(exc)) from exc

    request_extra = request_data.extra_body
    if isinstance(request_extra, dict) and request_extra:
        if policy.reject_extra_body_message:
            raise InvalidRequestError(policy.reject_extra_body_message)
        if policy.include_extra_body:
            extra_body = deepcopy(request_extra)
            if policy.extra_body_validator is not None:
                try:
                    policy.extra_body_validator(extra_body)
                except ValueError as exc:
                    raise InvalidRequestError(str(exc)) from exc
            body["extra_body"] = extra_body

    _apply_common_openai_chat_policy(body, policy)

    for postprocess in postprocessors:
        postprocess(body, request_data, thinking_enabled)

    logger.debug(
        "{}_REQUEST: conversion done model={} msgs={} tools={}",
        policy.provider_name,
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body


def _apply_common_openai_chat_policy(
    body: dict[str, Any], policy: OpenAIChatRequestPolicy
) -> None:
    if policy.strip_message_names:
        _strip_message_names(body.get("messages"))

    for key in policy.unsupported_body_keys:
        body.pop(key, None)

    if policy.max_tokens_field == "max_completion_tokens":
        _normalize_max_completion_tokens(body)

    if policy.normalize_n_to_one and body.get("n") is not None:
        body["n"] = 1


def _strip_message_names(messages: Any) -> None:
    if not isinstance(messages, list):
        return
    for message in messages:
        if isinstance(message, dict):
            message.pop("name", None)


def _normalize_max_completion_tokens(body: dict[str, Any]) -> None:
    if "max_completion_tokens" in body:
        body.pop("max_tokens", None)
        return
    if "max_tokens" in body and body["max_tokens"] is not None:
        body["max_completion_tokens"] = body.pop("max_tokens")
