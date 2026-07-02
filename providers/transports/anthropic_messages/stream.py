"""Native Anthropic Messages upstream adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from core.anthropic.stream_contracts import parse_sse_text
from core.anthropic.streaming import (
    AnthropicStreamLedger,
    RecoveryController,
    RecoveryFailureAction,
    TruncatedProviderStreamError,
    tool_schemas_by_name,
)
from core.trace import provider_native_messages_body_snapshot, trace_event
from providers.transports.http import maybe_await_aclose

from .recovery import AnthropicMessagesRecovery


async def iter_sse_events(response: httpx.Response) -> AsyncIterator[str]:
    """Group line-delimited SSE responses into full SSE events."""
    event_lines: list[str] = []
    async for line in response.aiter_lines():
        if line:
            event_lines.append(line)
            continue
        if event_lines:
            yield "\n".join(event_lines) + "\n\n"
            event_lines.clear()
    if event_lines:
        yield "\n".join(event_lines) + "\n\n"


class AnthropicMessagesStreamAdapter:
    """Convert one native Anthropic upstream stream into normalized Anthropic SSE."""

    def __init__(
        self,
        transport: Any,
        *,
        request: Any,
        input_tokens: int,
        request_id: str | None,
        thinking_enabled: bool | None,
    ) -> None:
        self._transport = transport
        self._request = request
        self._input_tokens = input_tokens
        self._request_id = request_id
        self._thinking_enabled = thinking_enabled
        self._recovery = AnthropicMessagesRecovery(
            transport,
            iter_stream_chunks=self.iter_stream_chunks,
        )

    async def run(self) -> AsyncIterator[str]:
        """Stream response via a native Anthropic-compatible messages endpoint."""
        tag = self._transport._provider_name
        req_tag = f" request_id={self._request_id}" if self._request_id else ""
        body = self._transport._build_request_body(
            self._request, thinking_enabled=self._thinking_enabled
        )
        thinking_enabled = self._transport._is_thinking_enabled(
            self._request, self._thinking_enabled
        )

        trace_event(
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider=tag,
            gateway_model=self._request.model,
            downstream_model=body.get("model"),
            message_count=len(body.get("messages", [])),
            tool_count=len(body.get("tools", [])),
            body=provider_native_messages_body_snapshot(body),
        )

        response: httpx.Response | None = None
        sent_any_event = False
        state = self._transport._new_stream_state(
            self._request, thinking_enabled=thinking_enabled
        )
        ledger = self._new_ledger()
        recovery = RecoveryController(provider_name=tag, request_id=self._request_id)

        async with self._transport._global_rate_limiter.concurrency_slot():
            while True:
                stream_opened = False
                try:
                    response = (
                        await self._transport._global_rate_limiter.execute_with_retry(
                            self._transport._validated_stream_send,
                            body,
                            req_tag=req_tag,
                        )
                    )
                    stream_opened = True
                    chunk_count = 0
                    chunk_bytes = 0

                    async for chunk in self.iter_stream_chunks(
                        response,
                        state=state,
                        thinking_enabled=thinking_enabled,
                    ):
                        chunk_count += 1
                        chunk_bytes += len(chunk.encode("utf-8", errors="replace"))
                        for parsed in parse_sse_text(chunk):
                            emitted = ledger.ingest_native_event(parsed)
                            if emitted is None:
                                continue
                            for event in recovery.push(emitted):
                                sent_any_event = True
                                yield event

                    if not ledger.has_terminal_message():
                        raise TruncatedProviderStreamError(
                            "Provider stream ended without message_stop."
                        )

                    trace_event(
                        stage="provider",
                        event="provider.response.completed",
                        source="provider",
                        provider=tag,
                        gateway_model=self._request.model,
                        sse_chunks_out=chunk_count,
                        sse_bytes_out=chunk_bytes,
                    )
                    for event in recovery.flush():
                        sent_any_event = True
                        yield event
                    return

                except Exception as error:
                    generated_output = ledger.has_content_block()
                    complete_tool_salvageable = generated_output and (
                        ledger.can_salvage_tool_use(tool_schemas_by_name(self._request))
                    )
                    decision = recovery.advance_failure(
                        error,
                        stream_opened=stream_opened,
                        generated_output=generated_output,
                        complete_tool_salvageable=complete_tool_salvageable,
                    )
                    if decision.action == RecoveryFailureAction.EARLY_RETRY:
                        if response is not None and not response.is_closed:
                            await maybe_await_aclose(response)
                        response = None
                        state = self._transport._new_stream_state(
                            self._request, thinking_enabled=thinking_enabled
                        )
                        ledger = self._new_ledger()
                        sent_any_event = False
                        continue

                    if decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY:
                        try:
                            recovery_events = await self._recovery.events(
                                body=body,
                                request=self._request,
                                ledger=ledger,
                                error=error,
                                request_id=self._request_id,
                                req_tag=req_tag,
                                thinking_enabled=thinking_enabled,
                            )
                        except Exception as recovery_error:
                            trace_event(
                                stage="provider",
                                event="provider.recovery.failed",
                                source="provider",
                                provider=tag,
                                request_id=self._request_id,
                                exc_type=type(recovery_error).__name__,
                            )
                            recovery_events = None
                        if recovery_events is not None:
                            for event in recovery.flush_uncommitted(decision):
                                sent_any_event = True
                                yield event
                            for event in recovery_events:
                                yield event
                            return

                    if not isinstance(error, httpx.HTTPStatusError):
                        self._transport._log_stream_transport_error(
                            tag, req_tag, error, request_id=self._request_id
                        )
                    error_message = self._transport._get_error_message(
                        error, self._request_id
                    )

                    if response is not None and not response.is_closed:
                        await maybe_await_aclose(response)

                    trace_event(
                        stage="provider",
                        event="provider.response.error",
                        source="provider",
                        provider=tag,
                        error_message=error_message,
                        exc_type=type(error).__name__,
                        mid_stream=(
                            sent_any_event
                            or decision.committed
                            or decision.has_buffered
                        ),
                    )
                    if decision.committed or decision.has_buffered:
                        if not decision.committed:
                            for event in recovery.flush():
                                sent_any_event = True
                                yield event
                        for event in ledger.midstream_error_tail(error_message):
                            yield event
                    else:
                        recovery.discard()
                        for event in self._transport._emit_error_events(
                            request=self._request,
                            input_tokens=self._input_tokens,
                            error_message=error_message,
                            sent_any_event=False,
                        ):
                            yield event
                    return
                finally:
                    if response is not None and not response.is_closed:
                        await maybe_await_aclose(response)

    async def iter_stream_chunks(
        self,
        response: httpx.Response,
        *,
        state: Any,
        thinking_enabled: bool,
    ) -> AsyncIterator[str]:
        """Yield normalized grouped SSE events from the provider stream."""
        async for event in iter_sse_events(response):
            output_event = self._transport._transform_stream_event(
                event,
                state,
                thinking_enabled=thinking_enabled,
            )
            if output_event is not None:
                yield output_event

    def _new_ledger(self) -> AnthropicStreamLedger:
        return AnthropicStreamLedger(
            None,
            self._request.model,
            self._input_tokens,
            log_raw_events=self._transport._config.log_raw_sse_events,
        )
