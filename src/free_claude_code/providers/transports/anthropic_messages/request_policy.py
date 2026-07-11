"""Request-body policy for native Anthropic-compatible providers."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger

from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.native_messages_request import (
    build_base_native_anthropic_request_body,
)
from free_claude_code.providers.exceptions import InvalidRequestError

NativeExtraBodyPolicy = Literal["drop", "reject"]
NativeMessagesPostprocessor = Callable[[dict[str, Any], MessagesRequest, bool], None]


@dataclass(frozen=True, slots=True)
class NativeMessagesRequestPolicy:
    """Provider policy for native Anthropic Messages request construction."""

    provider_name: str
    extra_body: NativeExtraBodyPolicy = "drop"
    default_max_tokens: int = ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    force_stream: bool = True
    reject_extra_body_message: str | None = None


def build_native_messages_request_body(
    request_data: MessagesRequest,
    *,
    thinking_enabled: bool,
    policy: NativeMessagesRequestPolicy,
    postprocessors: Iterable[NativeMessagesPostprocessor] = (),
) -> dict[str, Any]:
    """Build a native Anthropic-compatible Messages request body."""
    logger.debug(
        "{}_REQUEST: native build model={} msgs={}",
        policy.provider_name,
        request_data.model,
        len(request_data.messages),
    )

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


def _apply_extra_body_policy(
    body: dict[str, Any],
    request_data: MessagesRequest,
    policy: NativeMessagesRequestPolicy,
) -> None:
    extra = request_data.extra_body
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
    raise AssertionError(f"Unhandled native extra_body policy: {policy.extra_body}")
