"""OpenRouter provider implementation."""

import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic.models import MessagesRequest, ThinkingConfig
from free_claude_code.core.anthropic.streaming import AnthropicStreamLedger
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.model_listing import (
    extract_openrouter_tool_model_ids,
    extract_openrouter_tool_model_infos,
)
from free_claude_code.providers.openai_chat import (
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
    build_openai_chat_request_body,
    validate_extra_body_does_not_override_canonical_fields,
)
from free_claude_code.providers.rate_limit import ProviderRateLimiter

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="OPENROUTER",
    include_extra_body=True,
    extra_body_validator=validate_extra_body_does_not_override_canonical_fields,
    default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
)
_PROFILE = OpenAIChatProfile(_REQUEST_POLICY)


class OpenRouterProvider(OpenAIChatProvider):
    """OpenRouter provider using the OpenAI-compatible Chat Completions API."""

    def __init__(self, config: ProviderConfig, *, rate_limiter: ProviderRateLimiter):
        super().__init__(
            config,
            profile=_PROFILE,
            rate_limiter=rate_limiter,
        )

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        effective_thinking_enabled = self._is_thinking_enabled(
            request, thinking_enabled
        )
        return build_openai_chat_request_body(
            request,
            thinking_enabled=effective_thinking_enabled,
            policy=_REQUEST_POLICY,
            postprocessors=(
                _apply_openrouter_reasoning_policy,
                _apply_openrouter_reasoning_details_replay,
            ),
        )

    async def list_model_ids(self) -> frozenset[str]:
        """Only advertise OpenRouter models that can run Claude Code tools."""
        payload = await self._client.models.list()
        return extract_openrouter_tool_model_ids(
            payload, provider_name=self._provider_name
        )

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Advertise OpenRouter tool models with reasoning capability metadata."""
        payload = await self._client.models.list()
        return extract_openrouter_tool_model_infos(
            payload, provider_name=self._provider_name
        )

    def _handle_extra_reasoning(
        self, delta: Any, ledger: AnthropicStreamLedger, *, thinking_enabled: bool
    ) -> Iterator[str]:
        """Map OpenRouter reasoning details onto Anthropic thinking blocks."""
        if not thinking_enabled:
            return iter(())
        return _iter_openrouter_reasoning_detail_events(delta, ledger)


def _apply_openrouter_reasoning_policy(
    body: dict[str, Any], request: MessagesRequest, thinking_enabled: bool
) -> None:
    if not thinking_enabled:
        return
    extra_body = body.setdefault("extra_body", {})
    if not isinstance(extra_body, dict):
        return
    reasoning = extra_body.setdefault("reasoning", {"enabled": True})
    if not isinstance(reasoning, dict):
        return
    reasoning.setdefault("enabled", True)
    budget_tokens = _thinking_budget_tokens(request.thinking)
    if isinstance(budget_tokens, int):
        reasoning.setdefault("max_tokens", budget_tokens)


def _apply_openrouter_reasoning_details_replay(
    body: dict[str, Any], request: MessagesRequest, thinking_enabled: bool
) -> None:
    if not thinking_enabled:
        return
    assistant_details = _assistant_reasoning_details(request.messages)
    if not assistant_details:
        return
    messages = body.get("messages")
    if not isinstance(messages, list):
        return

    cursor = 0
    for details in assistant_details:
        for index in range(cursor, len(messages)):
            message = messages[index]
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            existing = message.get("reasoning_details")
            if isinstance(existing, list):
                existing.extend(details)
            else:
                message["reasoning_details"] = list(details)
            cursor = index + 1
            break


def _assistant_reasoning_details(messages: Any) -> list[list[dict[str, Any]]]:
    if not _is_sequence(messages):
        return []
    result: list[list[dict[str, Any]]] = []
    for message in messages:
        if _field(message, "role") != "assistant":
            continue
        details = _redacted_reasoning_details(_field(message, "content"))
        if details:
            result.append(details)
    return result


def _redacted_reasoning_details(content: Any) -> list[dict[str, Any]]:
    if not _is_sequence(content):
        return []
    details: list[dict[str, Any]] = []
    for block in content:
        if _field(block, "type") != "redacted_thinking":
            continue
        data = _field(block, "data")
        if not isinstance(data, str) or not data:
            continue
        parsed = _json_payload(data)
        if isinstance(parsed, list):
            details.extend(item for item in parsed if isinstance(item, dict))
        elif isinstance(parsed, dict):
            details.append(parsed)
        else:
            details.append({"type": "reasoning.encrypted", "data": data})
    return details


def _thinking_budget_tokens(thinking: ThinkingConfig | None) -> int | None:
    value = thinking.budget_tokens if thinking is not None else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _iter_openrouter_reasoning_detail_events(
    delta: Any, ledger: AnthropicStreamLedger
) -> Iterator[str]:
    details = _field(delta, "reasoning_details")
    if details is None:
        extra = _field(delta, "model_extra")
        if isinstance(extra, Mapping):
            details = extra.get("reasoning_details")
    if not _is_sequence(details):
        return

    native_reasoning = _field(delta, "reasoning_content")
    has_native_reasoning = isinstance(native_reasoning, str) and bool(native_reasoning)
    for detail in details:
        encrypted = _reasoning_detail_encrypted(detail)
        if encrypted:
            yield from ledger.close_content_blocks()
            index = ledger.blocks.allocate_index()
            yield ledger.content_block_start(index, "redacted_thinking", data=encrypted)
            yield ledger.content_block_stop(index)
            continue
        if has_native_reasoning:
            continue
        text = _reasoning_detail_text(detail)
        if not text:
            continue
        yield from ledger.ensure_thinking_block()
        yield ledger.emit_thinking_delta(text)


def _reasoning_detail_text(detail: Any) -> str | None:
    kind = str(_field(detail, "type") or "").lower()
    if "encrypted" in kind or "redacted" in kind:
        return None
    for key in ("text", "content", "reasoning"):
        value = _field(detail, key)
        if isinstance(value, str) and value:
            return value
    return None


def _reasoning_detail_encrypted(detail: Any) -> str | None:
    kind = str(_field(detail, "type") or "").lower()
    if "encrypted" not in kind and "redacted" not in kind and "summary" not in kind:
        return None
    if isinstance(detail, Mapping):
        return json.dumps(dict(detail), separators=(",", ":"))
    return None


def _json_payload(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _field(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )
