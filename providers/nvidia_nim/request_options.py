"""NVIDIA NIM request option injection."""

from __future__ import annotations

from typing import Any

from config.nim import NimSettings
from core.anthropic import set_if_not_none
from providers.transports.openai_chat import (
    OpenAIChatRequestPolicy,
    build_openai_chat_request_body,
)

from .tool_schema import sanitize_nim_tool_schemas

_REQUEST_POLICY = OpenAIChatRequestPolicy(provider_name="NIM")


def build_nim_request_body(
    request_data: Any, nim: NimSettings, *, thinking_enabled: bool
) -> dict[str, Any]:
    """Build OpenAI-format request body from Anthropic request plus NIM settings."""
    return build_openai_chat_request_body(
        request_data,
        thinking_enabled=thinking_enabled,
        policy=_REQUEST_POLICY,
        postprocessors=(
            lambda body, request, enabled: apply_nim_request_options(
                body,
                request,
                enabled,
                nim=nim,
            ),
        ),
    )


def apply_nim_request_options(
    body: dict[str, Any],
    request_data: Any,
    thinking_enabled: bool,
    *,
    nim: NimSettings,
) -> None:
    """Apply NIM schema repairs and configured request defaults."""
    sanitize_nim_tool_schemas(body)

    max_tokens = body.get("max_tokens") or getattr(request_data, "max_tokens", None)
    if max_tokens is None:
        max_tokens = nim.max_tokens
    elif nim.max_tokens:
        max_tokens = min(max_tokens, nim.max_tokens)
    set_if_not_none(body, "max_tokens", max_tokens)

    if body.get("temperature") is None and nim.temperature is not None:
        body["temperature"] = nim.temperature
    if body.get("top_p") is None and nim.top_p is not None:
        body["top_p"] = nim.top_p

    if "stop" not in body and nim.stop:
        body["stop"] = nim.stop

    if nim.presence_penalty != 0.0:
        body["presence_penalty"] = nim.presence_penalty
    if nim.frequency_penalty != 0.0:
        body["frequency_penalty"] = nim.frequency_penalty
    if nim.seed is not None:
        body["seed"] = nim.seed

    body["parallel_tool_calls"] = nim.parallel_tool_calls

    extra_body: dict[str, Any] = {}
    request_extra = getattr(request_data, "extra_body", None)
    if request_extra:
        extra_body.update(request_extra)

    if thinking_enabled:
        chat_template_kwargs = extra_body.setdefault(
            "chat_template_kwargs", {"thinking": True, "enable_thinking": True}
        )
        if isinstance(chat_template_kwargs, dict):
            chat_template_kwargs.setdefault("reasoning_budget", max_tokens)

    req_top_k = getattr(request_data, "top_k", None)
    top_k = req_top_k if req_top_k is not None else nim.top_k
    _set_extra(extra_body, "top_k", top_k, ignore_value=-1)
    _set_extra(extra_body, "min_p", nim.min_p, ignore_value=0.0)
    _set_extra(
        extra_body, "repetition_penalty", nim.repetition_penalty, ignore_value=1.0
    )
    _set_extra(extra_body, "min_tokens", nim.min_tokens, ignore_value=0)
    _set_extra(extra_body, "chat_template", nim.chat_template)
    _set_extra(extra_body, "request_id", nim.request_id)
    _set_extra(extra_body, "ignore_eos", nim.ignore_eos)

    if extra_body:
        body["extra_body"] = extra_body


def _set_extra(
    extra_body: dict[str, Any], key: str, value: Any, ignore_value: Any = None
) -> None:
    if key in extra_body:
        return
    if value is None:
        return
    if ignore_value is not None and value == ignore_value:
        return
    extra_body[key] = value
