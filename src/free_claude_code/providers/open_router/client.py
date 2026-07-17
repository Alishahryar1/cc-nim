"""OpenRouter provider implementation."""

import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.streaming import AnthropicStreamLedger
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.model_listing import (
    extract_openrouter_tool_model_ids,
    extract_openrouter_tool_model_infos,
)
from free_claude_code.providers.openai_chat import (
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
    ReasoningObject,
    validate_extra_body_does_not_override_canonical_fields,
)
from free_claude_code.providers.rate_limit import ProviderRateLimiter

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="OPENROUTER",
    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    include_extra_body=True,
    extra_body_validator=validate_extra_body_does_not_override_canonical_fields,
    default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
)


class OpenRouterProvider(OpenAIChatProvider):
    """OpenRouter provider using the OpenAI-compatible Chat Completions API."""

    def __init__(self, config: ProviderConfig, *, rate_limiter: ProviderRateLimiter):
        super().__init__(
            config,
            profile=_PROFILE,
            rate_limiter=rate_limiter,
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
        self, delta: Any, ledger: AnthropicStreamLedger, *, output_reasoning: bool
    ) -> Iterator[str]:
        """Map OpenRouter reasoning details onto Anthropic thinking blocks."""
        if not output_reasoning:
            return iter(())
        return _iter_openrouter_reasoning_detail_events(delta, ledger)


def _apply_openrouter_reasoning_details_replay(
    body: dict[str, Any], request: MessagesRequest, _policy: ReasoningPolicy
) -> None:
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


_PROFILE = OpenAIChatProfile(
    _REQUEST_POLICY,
    ReasoningObject(tuple((effort, effort.value) for effort in ReasoningEffort)),
    postprocessors=(_apply_openrouter_reasoning_details_replay,),
)


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
