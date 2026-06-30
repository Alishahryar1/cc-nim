"""Native Anthropic Messages recovery event construction."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from core.anthropic.stream_contracts import parse_sse_text
from core.anthropic.streaming import (
    MIDSTREAM_RECOVERY_ATTEMPTS,
    AnthropicStreamLedger,
    TruncatedProviderStreamError,
    accept_tool_json_repair,
    continuation_suffix,
    is_retryable_stream_error,
    make_text_recovery_body,
    make_tool_repair_body,
    parse_complete_tool_input,
    tool_schemas_by_name,
)
from core.trace import trace_event
from providers.transports.http import maybe_await_aclose

IterStreamChunks = Callable[..., AsyncIterator[str]]


class AnthropicMessagesRecovery:
    """Construct recovery events for interrupted native Anthropic streams."""

    def __init__(
        self,
        transport: Any,
        *,
        iter_stream_chunks: IterStreamChunks,
    ) -> None:
        self._transport = transport
        self._iter_stream_chunks = iter_stream_chunks

    async def collect_text(
        self,
        body: dict[str, Any],
        *,
        req_tag: str,
        thinking_enabled: bool,
    ) -> tuple[str, str]:
        """Collect text/thinking from an internal native recovery request."""
        last_error: Exception | None = None
        for attempt in range(MIDSTREAM_RECOVERY_ATTEMPTS):
            response: httpx.Response | None = None
            try:
                response = (
                    await self._transport._global_rate_limiter.execute_with_retry(
                        self._transport._validated_stream_send, body, req_tag=req_tag
                    )
                )
                state = self._transport._new_stream_state(
                    None, thinking_enabled=thinking_enabled
                )
                chunks = [
                    chunk
                    async for chunk in self._iter_stream_chunks(
                        response,
                        state=state,
                        thinking_enabled=thinking_enabled,
                    )
                ]
                text_parts: list[str] = []
                thinking_parts: list[str] = []
                terminal_seen = False
                for event in parse_sse_text("".join(chunks)):
                    if event.event == "message_stop":
                        terminal_seen = True
                    content_block = event.data.get("content_block")
                    if isinstance(content_block, dict):
                        text = content_block.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
                        thinking = content_block.get("thinking")
                        if isinstance(thinking, str):
                            thinking_parts.append(thinking)
                    delta = event.data.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    text = delta.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
                    thinking = delta.get("thinking")
                    if isinstance(thinking, str):
                        thinking_parts.append(thinking)
                if not terminal_seen:
                    raise TruncatedProviderStreamError(
                        "Recovery stream ended without message_stop."
                    )
                return "".join(text_parts), "".join(thinking_parts)
            except Exception as error:
                last_error = error
                if not is_retryable_stream_error(error):
                    raise
                trace_event(
                    stage="provider",
                    event="provider.recovery.retry",
                    source="provider",
                    provider=self._transport._provider_name,
                    recovery_kind="native_text",
                    attempt=attempt + 1,
                    max_attempts=MIDSTREAM_RECOVERY_ATTEMPTS,
                    exc_type=type(error).__name__,
                )
            finally:
                if response is not None and not response.is_closed:
                    await maybe_await_aclose(response)
        if last_error is not None:
            raise last_error
        return "", ""

    async def events(
        self,
        *,
        body: dict[str, Any],
        request: Any,
        ledger: AnthropicStreamLedger,
        error: Exception,
        request_id: str | None,
        req_tag: str,
        thinking_enabled: bool,
    ) -> list[str] | None:
        """Build recovery events, or return None when recovery is impossible."""
        if not is_retryable_stream_error(error):
            return None

        schemas = tool_schemas_by_name(request)
        if ledger.tool_blocks():
            repair_events: list[str] = []
            for index, block in enumerate(ledger.tool_blocks()):
                if (
                    block.tool_id
                    and block.name
                    and parse_complete_tool_input(block.content, block.name, schemas)
                    is not None
                ):
                    continue
                schema = schemas.get(block.name)
                recovery_body = make_tool_repair_body(
                    body,
                    tool_name=block.name,
                    prefix=block.content,
                    input_schema=schema.input_schema if schema is not None else None,
                )
                accepted_suffix: str | None = None
                for attempt in range(MIDSTREAM_RECOVERY_ATTEMPTS):
                    text, _ = await self.collect_text(
                        recovery_body,
                        req_tag=req_tag,
                        thinking_enabled=thinking_enabled,
                    )
                    repair = accept_tool_json_repair(
                        block.content,
                        text,
                        tool_name=block.name,
                        schemas=schemas,
                    )
                    if repair is not None:
                        accepted_suffix = repair.suffix
                        trace_event(
                            stage="provider",
                            event="provider.recovery.tool_repaired",
                            source="provider",
                            provider=self._transport._provider_name,
                            tool_name=block.name,
                            attempt=attempt + 1,
                        )
                        break
                if accepted_suffix is None:
                    return None
                repair_events.extend(
                    ledger.append_tool_repair_suffix(index, accepted_suffix)
                )

            if not ledger.can_salvage_tool_use(schemas):
                return None
            events = list(repair_events)
            events.extend(ledger.success_tail("end_turn"))
            trace_event(
                stage="provider",
                event="provider.recovery.tool_salvaged",
                source="provider",
                provider=self._transport._provider_name,
                request_id=request_id,
            )
            return events

        partial_text = ledger.accumulated_text
        partial_thinking = ledger.accumulated_reasoning
        if not partial_text and not partial_thinking:
            return None
        if not ledger.can_append_content():
            return None
        recovery_body = make_text_recovery_body(body, partial_text, partial_thinking)
        text, thinking = await self.collect_text(
            recovery_body,
            req_tag=req_tag,
            thinking_enabled=thinking_enabled,
        )
        text_suffix = continuation_suffix(partial_text, text)
        thinking_suffix = continuation_suffix(partial_thinking, thinking)
        events: list[str] = []
        if thinking_suffix:
            events.extend(ledger.append_thinking_suffix(thinking_suffix))
        if text_suffix:
            events.extend(ledger.append_text_suffix(text_suffix))
        if not events:
            return None
        events.extend(ledger.success_tail("end_turn"))
        trace_event(
            stage="provider",
            event="provider.recovery.continued",
            source="provider",
            provider=self._transport._provider_name,
            request_id=request_id,
        )
        return events
