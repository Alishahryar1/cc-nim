"""Request-body policy for native Anthropic-compatible providers."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.native_messages_request import (
    OpenRouterExtraBodyError,
    build_base_native_anthropic_request_body,
    build_openrouter_native_request_body,
    validate_openrouter_extra_body,
)
from providers.exceptions import InvalidRequestError

NativeExtraBodyPolicy = Literal["drop", "reject", "merge_validated", "openrouter"]
NativeMessagesPostprocessor = Callable[[dict[str, Any], Any, bool], None]


@dataclass(frozen=True, slots=True)
class NativeMessagesRequestPolicy:
    """Provider policy for native Anthropic Messages request construction."""

    provider_name: str
    extra_body: NativeExtraBodyPolicy = "drop"
    default_max_tokens: int = ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    force_stream: bool = True
    reject_extra_body_message: str | None = None


def build_native_messages_request_body(
    request_data: Any,
    *,
    thinking_enabled: bool,
    policy: NativeMessagesRequestPolicy,
    postprocessors: Iterable[NativeMessagesPostprocessor] = (),
) -> dict[str, Any]:
    """Build a native Anthropic-compatible Messages request body."""
    logger.debug(
        "{}_REQUEST: native build model={} msgs={}",
        policy.provider_name,
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    if policy.extra_body == "openrouter":
        body = _build_openrouter_body(
            request_data,
            thinking_enabled=thinking_enabled,
            policy=policy,
        )
    else:
        body = build_base_native_anthropic_request_body(
            request_data,
            default_max_tokens=policy.default_max_tokens,
            thinking_enabled=thinking_enabled,
        )
        _apply_extra_body_policy(body, request_data, policy)
        if policy.force_stream:
            body["stream"] = True

    for postprocess in postprocessors:
        postprocess(body, request_data, thinking_enabled)

    logger.debug(
        "{}_REQUEST: build done model={} msgs={} tools={}",
        policy.provider_name,
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body


def _build_openrouter_body(
    request_data: Any,
    *,
    thinking_enabled: bool,
    policy: NativeMessagesRequestPolicy,
) -> dict[str, Any]:
    try:
        return build_openrouter_native_request_body(
            request_data,
            thinking_enabled=thinking_enabled,
            default_max_tokens=policy.default_max_tokens,
        )
    except OpenRouterExtraBodyError as exc:
        raise InvalidRequestError(str(exc)) from exc


def _apply_extra_body_policy(
    body: dict[str, Any],
    request_data: Any,
    policy: NativeMessagesRequestPolicy,
) -> None:
    extra = getattr(request_data, "extra_body", None)
    if not extra:
        return

    if policy.extra_body == "drop":
        return
    if policy.extra_body == "reject":
        message = (
            policy.reject_extra_body_message
            or f"{policy.provider_name} native Messages API does not support extra_body on requests."
        )
        raise InvalidRequestError(message)
    if policy.extra_body == "merge_validated":
        if isinstance(extra, dict):
            try:
                validate_openrouter_extra_body(extra)
            except OpenRouterExtraBodyError as exc:
                raise InvalidRequestError(str(exc)) from exc
            body.update(extra)
        return

    raise AssertionError(f"Unhandled native extra_body policy: {policy.extra_body}")
