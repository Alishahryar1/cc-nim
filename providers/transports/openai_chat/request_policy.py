"""Request-body policy for OpenAI-compatible chat providers."""

from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.exceptions import InvalidRequestError

MaxTokensField = Literal["max_tokens", "max_completion_tokens"]
OpenAIChatPostprocessor = Callable[[dict[str, Any], Any, bool], None]


@dataclass(frozen=True, slots=True)
class OpenAIChatRequestPolicy:
    """Provider policy for Anthropic-to-OpenAI chat request conversion."""

    provider_name: str
    include_extra_body: bool = False
    max_tokens_field: MaxTokensField = "max_tokens"
    strip_message_names: bool = False
    unsupported_body_keys: frozenset[str] = field(default_factory=frozenset)
    normalize_n_to_one: bool = False


def build_openai_chat_request_body(
    request_data: Any,
    *,
    thinking_enabled: bool,
    policy: OpenAIChatRequestPolicy,
    postprocessors: Iterable[OpenAIChatPostprocessor] = (),
) -> dict[str, Any]:
    """Build an OpenAI-compatible chat request body from an Anthropic request."""
    logger.debug(
        "{}_REQUEST: conversion start model={} msgs={}",
        policy.provider_name,
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    try:
        body = build_base_request_body(
            request_data,
            reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
            if thinking_enabled
            else ReasoningReplayMode.DISABLED,
        )
    except OpenAIConversionError as exc:
        raise InvalidRequestError(str(exc)) from exc

    if policy.include_extra_body:
        request_extra = getattr(request_data, "extra_body", None)
        if isinstance(request_extra, dict) and request_extra:
            body["extra_body"] = deepcopy(request_extra)

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
