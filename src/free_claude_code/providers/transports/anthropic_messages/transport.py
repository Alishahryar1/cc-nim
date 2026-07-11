"""Shared transport for providers with native Anthropic Messages endpoints."""

import sys
from collections.abc import AsyncIterator
from typing import Any

import httpx

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic import execution_failure_from_anthropic_error
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.native_sse_block_policy import (
    NativeSseBlockPolicyState,
    transform_native_sse_block_event,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.anthropic.streaming import (
    AnthropicStreamLedger,
    accept_tool_json_repair,
    continuation_suffix,
    make_text_recovery_body,
    make_tool_repair_body,
    parse_complete_tool_input,
    tool_schemas_by_name,
)
from free_claude_code.core.trace import (
    close_stream_input,
    provider_native_messages_body_snapshot,
    trace_event,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.failure_policy import classify_provider_failure
from free_claude_code.providers.model_listing import (
    extract_openai_model_ids,
    model_infos_from_ids,
)
from free_claude_code.providers.rate_limit import ProviderRateLimiter
from free_claude_code.providers.stream_recovery import (
    MIDSTREAM_RECOVERY_ATTEMPTS,
    RecoveryController,
    RecoveryFailureAction,
    TruncatedProviderStreamError,
    is_retryable_stream_error,
)
from free_claude_code.providers.transports.http import (
    close_provider_stream,
    maybe_await_aclose,
)

from .http import model_list_json, raise_for_status_with_body
from .request_policy import (
    NativeMessagesRequestPolicy,
    build_native_messages_request_body,
)


class AnthropicMessagesTransport(BaseProvider):
    """Base class for providers that stream from an Anthropic-compatible endpoint."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider_name: str,
        default_base_url: str,
        rate_limiter: ProviderRateLimiter,
    ):
        super().__init__(config)
        self._provider_name = provider_name
        self._api_key = config.api_key
        self._base_url = (config.base_url or default_base_url).rstrip("/")
        self._request_policy = NativeMessagesRequestPolicy(provider_name=provider_name)
        self._rate_limiter = rate_limiter
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            proxy=config.proxy or None,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
        )

    async def cleanup(self) -> None:
        """Release HTTP client resources."""
        await self._client.aclose()

    async def list_model_ids(self) -> frozenset[str]:
        """Return model ids from an OpenAI-compatible ``/models`` endpoint."""
        return frozenset(info.model_id for info in await self.list_model_infos())

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Return model ids plus optional metadata from a ``/models`` endpoint."""
        response = await self._send_model_list_request()
        try:
            payload = model_list_json(response, provider_name=self._provider_name)
            return self._extract_model_infos_from_model_list_payload(payload)
        finally:
            await maybe_await_aclose(response)

    async def _send_model_list_request(self) -> httpx.Response:
        """Query the provider endpoint that advertises available model ids."""
        return await self._client.get(
            "/models",
            headers=self._model_list_headers(),
        )

    def _model_list_headers(self) -> dict[str, str]:
        """Return headers for model-list requests."""
        return {}

    def _extract_model_ids_from_model_list_payload(
        self, payload: Any
    ) -> frozenset[str]:
        """Parse the provider model-list response body."""
        return extract_openai_model_ids(payload, provider_name=self._provider_name)

    def _extract_model_infos_from_model_list_payload(
        self, payload: Any
    ) -> frozenset[ProviderModelInfo]:
        """Parse provider model metadata; default to unknown capabilities."""
        return model_infos_from_ids(
            self._extract_model_ids_from_model_list_payload(payload)
        )

    def _request_headers(self) -> dict[str, str]:
        """Return headers for the native messages request."""
        return {"Content-Type": "application/json"}

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        """Build a native Anthropic request body."""
        thinking_enabled = self._is_thinking_enabled(request, thinking_enabled)
        return self._build_request_body_with_resolved_thinking(
            request,
            thinking_enabled=thinking_enabled,
        )

    def preflight_stream(
        self, request: MessagesRequest, *, thinking_enabled: bool | None = None
    ) -> None:
        """Validate native Messages request construction before streaming."""
        self._build_request_body(request, thinking_enabled=thinking_enabled)

    def _build_request_body_with_resolved_thinking(
        self, request: MessagesRequest, *, thinking_enabled: bool
    ) -> dict:
        """Build a native Anthropic request body after thinking is resolved."""
        return build_native_messages_request_body(
            request,
            thinking_enabled=thinking_enabled,
            policy=self._request_policy,
        )

    async def _send_stream_request(self, body: dict) -> httpx.Response:
        """Create a streaming messages response."""
        # This transport always parses the upstream response as SSE, so the
        # upstream request must always be streaming — forwarding a client's
        # stream=false makes native providers return plain JSON that the SSE
        # reader misreads as a truncated stream.
        request = self._client.build_request(
            "POST",
            "/messages",
            json={**body, "stream": True},
            headers=self._request_headers(),
        )
        return await self._client.send(request, stream=True)

    async def _raise_for_status(
        self, response: httpx.Response, *, req_tag: str
    ) -> None:
        """Raise for non-200 responses after attaching safe error metadata."""
        await raise_for_status_with_body(
            response,
            provider_name=self._provider_name,
            req_tag=req_tag,
            log_api_error_tracebacks=self._config.log_api_error_tracebacks,
        )

    async def _validated_stream_send(
        self,
        body: dict,
        *,
        req_tag: str,
        request_id: str | None = None,
    ) -> httpx.Response:
        """Send request and raise mapped HTTP errors before yielding body chunks."""
        send_response = await self._send_stream_request(body)
        if send_response.status_code != 200:
            try:
                await self._raise_for_status(send_response, req_tag=req_tag)
            finally:
                if not send_response.is_closed:
                    await close_provider_stream(
                        send_response,
                        active_error=sys.exception(),
                        provider_name=self._provider_name,
                        request_id=request_id,
                    )
        return send_response

    def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream response via a native Anthropic-compatible messages endpoint."""
        runner = _AnthropicMessagesStreamRunner(
            self,
            request=request,
            input_tokens=input_tokens,
            request_id=request_id,
            thinking_enabled=thinking_enabled,
        )
        return runner.run()


async def _iter_sse_events(response: httpx.Response) -> AsyncIterator[str]:
    """Group line-delimited upstream data into complete SSE events."""
    event_lines: list[str] = []
    lines = response.aiter_lines()
    try:
        async for line in lines:
            if line:
                event_lines.append(line)
                continue
            if event_lines:
                yield "\n".join(event_lines) + "\n\n"
                event_lines.clear()
        if event_lines:
            yield "\n".join(event_lines) + "\n\n"
    finally:
        await close_stream_input(
            lines,
            owner="anthropic_messages.sse_lines",
            source="provider",
            preserved_error=sys.exception(),
        )


class _AnthropicMessagesStreamRunner:
    """Own one native Anthropic Messages request and all of its recovery work."""

    def __init__(
        self,
        transport: AnthropicMessagesTransport,
        *,
        request: MessagesRequest,
        input_tokens: int,
        request_id: str | None,
        thinking_enabled: bool | None,
    ) -> None:
        self._transport = transport
        self._request = request
        self._input_tokens = input_tokens
        self._request_id = request_id
        self._thinking_enabled = thinking_enabled

    async def run(self) -> AsyncIterator[str]:
        """Stream and recover one native Anthropic-compatible request."""
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
            request_id=self._request_id,
            gateway_model=self._request.model,
            downstream_model=body.get("model"),
            message_count=len(body.get("messages", [])),
            tool_count=len(body.get("tools", [])),
            body=provider_native_messages_body_snapshot(body),
        )

        response: httpx.Response | None = None
        sent_any_event = False
        state = NativeSseBlockPolicyState()
        ledger = self._new_ledger()
        recovery = RecoveryController(provider_name=tag, request_id=self._request_id)

        async with self._transport._rate_limiter.concurrency_slot():
            while True:
                stream_opened = False
                chunks: AsyncIterator[str] | None = None
                try:
                    response = await self._transport._rate_limiter.execute_with_retry(
                        self._transport._validated_stream_send,
                        body,
                        req_tag=req_tag,
                        request_id=self._request_id,
                    )
                    stream_opened = True
                    chunk_count = 0
                    chunk_bytes = 0

                    chunks = self._iter_stream_chunks(
                        response,
                        state=state,
                        thinking_enabled=thinking_enabled,
                    )
                    async for chunk in chunks:
                        chunk_count += 1
                        chunk_bytes += len(chunk.encode("utf-8", errors="replace"))
                        for parsed in parse_sse_text(chunk):
                            if parsed.event == "error":
                                raise execution_failure_from_anthropic_error(
                                    parsed.data
                                )
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
                        request_id=self._request_id,
                        gateway_model=self._request.model,
                        sse_chunks_out=chunk_count,
                        sse_bytes_out=chunk_bytes,
                    )
                    for event in recovery.flush():
                        sent_any_event = True
                        yield event
                    return

                except Exception as error:
                    if ledger.has_terminal_message():
                        trace_event(
                            stage="provider",
                            event="provider.response.completed",
                            source="provider",
                            provider=tag,
                            request_id=self._request_id,
                            gateway_model=self._request.model,
                            sse_chunks_out=chunk_count,
                            sse_bytes_out=chunk_bytes,
                            late_exc_type=type(error).__name__,
                        )
                        for event in recovery.flush():
                            sent_any_event = True
                            yield event
                        return

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
                            await close_provider_stream(
                                response,
                                active_error=error,
                                provider_name=tag,
                                request_id=self._request_id,
                            )
                        response = None
                        state = NativeSseBlockPolicyState()
                        ledger = self._new_ledger()
                        sent_any_event = False
                        continue

                    if decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY:
                        try:
                            recovery_events = await self._recovery_events(
                                body=body,
                                ledger=ledger,
                                error=error,
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
                    failure = classify_provider_failure(
                        error,
                        provider_name=tag,
                        read_timeout_s=self._transport._config.http_read_timeout,
                        request_id=self._request_id,
                        mark_rate_limited=(
                            self._transport._rate_limiter.extend_reactive_block
                        ),
                    )

                    error_trace: dict[str, Any] = {
                        "stage": "provider",
                        "event": "provider.response.error",
                        "source": "provider",
                        "provider": tag,
                        "request_id": self._request_id,
                        "exc_type": type(error).__name__,
                        "failure_kind": failure.kind.value,
                        "status_code": failure.status_code,
                        "provider_retryable": failure.retryable,
                        "mid_stream": sent_any_event or decision.committed,
                    }
                    if self._transport._config.log_api_error_tracebacks:
                        error_trace["error_message"] = failure.message
                    trace_event(**error_trace)
                    if decision.committed:
                        for event in ledger.close_unclosed_blocks():
                            yield event
                    elif decision.has_buffered and complete_tool_salvageable:
                        for event in recovery.flush():
                            sent_any_event = True
                            yield event
                        for event in ledger.close_unclosed_blocks():
                            yield event
                    else:
                        recovery.discard()
                    raise failure from error
                finally:
                    if chunks is not None:
                        await close_stream_input(
                            chunks,
                            owner="anthropic_messages.runner",
                            source="provider",
                            preserved_error=sys.exception(),
                        )
                    if response is not None and not response.is_closed:
                        await close_provider_stream(
                            response,
                            active_error=sys.exception(),
                            provider_name=tag,
                            request_id=self._request_id,
                        )

    async def _iter_stream_chunks(
        self,
        response: httpx.Response,
        *,
        state: NativeSseBlockPolicyState,
        thinking_enabled: bool,
    ) -> AsyncIterator[str]:
        """Yield normalized grouped SSE events from the provider stream."""
        events = _iter_sse_events(response)
        try:
            async for event in events:
                output_event = transform_native_sse_block_event(
                    event,
                    state,
                    thinking_enabled=thinking_enabled,
                )
                if output_event is not None:
                    yield output_event
        finally:
            await close_stream_input(
                events,
                owner="anthropic_messages.block_policy",
                source="provider",
                preserved_error=sys.exception(),
            )

    def _new_ledger(self) -> AnthropicStreamLedger:
        return AnthropicStreamLedger(
            None,
            self._request.model,
            self._input_tokens,
            log_raw_events=self._transport._config.log_raw_sse_events,
        )

    async def _collect_recovery_text(
        self,
        body: dict[str, Any],
        *,
        req_tag: str,
        thinking_enabled: bool,
    ) -> tuple[str, str]:
        """Collect text and thinking from an internal continuation request."""
        last_error: Exception | None = None
        for attempt in range(MIDSTREAM_RECOVERY_ATTEMPTS):
            response: httpx.Response | None = None
            try:
                response = await self._transport._rate_limiter.execute_with_retry(
                    self._transport._validated_stream_send,
                    body,
                    req_tag=req_tag,
                    request_id=self._request_id,
                )
                state = NativeSseBlockPolicyState()
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

    async def _recovery_events(
        self,
        *,
        body: dict[str, Any],
        ledger: AnthropicStreamLedger,
        error: Exception,
        req_tag: str,
        thinking_enabled: bool,
    ) -> list[str] | None:
        """Build recovery events, or return None when recovery is impossible."""
        if not is_retryable_stream_error(error):
            return None

        schemas = tool_schemas_by_name(self._request)
        if ledger.tool_blocks():
            repair_events: list[str] = []
            for index, block in enumerate(ledger.tool_blocks()):
                if (
                    block.tool_id
                    and block.name
                    and parse_complete_tool_input(
                        block.content,
                        block.name,
                        schemas,
                    )
                    is not None
                ):
                    continue
                schema = schemas.get(block.name)
                recovery_body = make_tool_repair_body(
                    body,
                    tool_name=block.name,
                    prefix=block.content,
                    input_schema=(schema.input_schema if schema is not None else None),
                )
                accepted_suffix: str | None = None
                for attempt in range(MIDSTREAM_RECOVERY_ATTEMPTS):
                    text, _ = await self._collect_recovery_text(
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
                request_id=self._request_id,
            )
            return events

        partial_text = ledger.accumulated_text
        partial_thinking = ledger.accumulated_reasoning
        if not partial_text and not partial_thinking:
            return None
        if not ledger.can_append_content():
            return None
        recovery_body = make_text_recovery_body(
            body,
            partial_text,
            partial_thinking,
        )
        text, thinking = await self._collect_recovery_text(
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
            request_id=self._request_id,
        )
        return events
