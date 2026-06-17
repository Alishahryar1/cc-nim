"""Shared transport for providers with native Anthropic Messages endpoints."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

import httpx
from loguru import logger

from config.constants import (
    ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
    NATIVE_MESSAGES_ERROR_BODY_LOG_CAP_BYTES,
    PROVIDER_ERROR_BODY_DISPLAY_CAP_BYTES,
)
from core.anthropic import iter_provider_stream_error_sse_events
from core.anthropic.emitted_sse_tracker import EmittedNativeSseTracker
from core.anthropic.native_messages_request import (
    build_base_native_anthropic_request_body,
)
from core.anthropic.native_sse_block_policy import (
    NativeSseBlockPolicyState,
    transform_native_sse_block_event,
)
from core.anthropic.stream_contracts import parse_sse_text
from core.anthropic.stream_recovery import (
    MIDSTREAM_RECOVERY_ATTEMPTS,
    TruncatedProviderStreamError,
    accept_tool_json_repair,
    continuation_suffix,
    is_retryable_stream_error,
    make_native_text_recovery_body,
    make_native_tool_repair_body,
    parse_complete_tool_input,
    tool_schemas_by_name,
)
from core.anthropic.stream_recovery_session import (
    StreamFailureAction,
    StreamRecoverySession,
)
from core.trace import provider_native_messages_body_snapshot, trace_event
from providers.base import BaseProvider, ProviderConfig
from providers.error_mapping import (
    attach_provider_error_body,
    extract_provider_error_detail,
    map_error,
    user_visible_message_for_mapped_provider_error,
)
from providers.exceptions import ModelListResponseError
from providers.model_listing import (
    ProviderModelInfo,
    extract_openai_model_ids,
    model_infos_from_ids,
)
from providers.rate_limit import GlobalRateLimiter

StreamChunkMode = Literal["line", "event"]


async def _maybe_await_aclose(response: Any) -> None:
    """Call ``aclose`` on httpx-like responses; ignore non-async test doubles."""
    close = getattr(response, "aclose", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        await result


def _model_list_json(response: httpx.Response, *, provider_name: str) -> Any:
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:
        raise ModelListResponseError(
            f"{provider_name} model-list response is malformed: invalid JSON"
        ) from exc


class AnthropicMessagesTransport(BaseProvider):
    """Base class for providers that stream from an Anthropic-compatible endpoint."""

    stream_chunk_mode: StreamChunkMode = "line"

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider_name: str,
        default_base_url: str,
    ):
        super().__init__(config)
        self._provider_name = provider_name
        self._api_key = config.api_key
        self._base_url = (config.base_url or default_base_url).rstrip("/")
        self._global_rate_limiter = GlobalRateLimiter.get_scoped_instance(
            provider_name.lower(),
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
            max_concurrency=config.max_concurrency,
        )
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
            payload = _model_list_json(response, provider_name=self._provider_name)
            return self._extract_model_infos_from_model_list_payload(payload)
        finally:
            await _maybe_await_aclose(response)

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
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Build a native Anthropic request body."""
        thinking_enabled = self._is_thinking_enabled(request, thinking_enabled)
        return build_base_native_anthropic_request_body(
            request,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
            thinking_enabled=thinking_enabled,
        )

    async def _send_stream_request(self, body: dict) -> httpx.Response:
        """Create a streaming messages response."""
        request = self._client.build_request(
            "POST",
            "/messages",
            json=body,
            headers=self._request_headers(),
        )
        return await self._client.send(request, stream=True)

    async def _raise_for_status(
        self, response: httpx.Response, *, req_tag: str
    ) -> None:
        """Raise for non-200 responses after logging safe metadata (or capped body if opted in)."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            preview, truncated = await self._read_error_body_preview(
                response, PROVIDER_ERROR_BODY_DISPLAY_CAP_BYTES
            )
            attach_provider_error_body(error, preview, truncated=truncated)
            if self._config.log_api_error_tracebacks:
                log_preview = preview[:NATIVE_MESSAGES_ERROR_BODY_LOG_CAP_BYTES]
                log_truncated = truncated or len(preview) > len(log_preview)
                if log_preview:
                    text = log_preview.decode("utf-8", errors="replace")
                    logger.error(
                        "{}_ERROR:{} HTTP {} body_preview_bytes={} truncated={}: {}",
                        self._provider_name,
                        req_tag,
                        response.status_code,
                        len(log_preview),
                        log_truncated,
                        text,
                    )
                else:
                    logger.error(
                        "{}_ERROR:{} HTTP {} (empty error body)",
                        self._provider_name,
                        req_tag,
                        response.status_code,
                    )
            else:
                cl = response.headers.get("content-length", "").strip()
                extra = f" content_length_declared={cl}" if cl.isdigit() else ""
                body_extra = (
                    " empty_error_body"
                    if not preview
                    else f" error_body_bytes_read={len(preview)}"
                )
                logger.error(
                    "{}_ERROR:{} HTTP {}{}{}",
                    self._provider_name,
                    req_tag,
                    response.status_code,
                    extra,
                    body_extra,
                )
            raise error

    async def _read_error_body_preview(
        self, response: httpx.Response, max_bytes: int
    ) -> tuple[bytes, bool]:
        """Read at most ``max_bytes`` from the error body for logging. Returns (preview, truncated)."""
        if max_bytes <= 0:
            return b"", False
        received = 0
        parts: list[bytes] = []
        truncated = False
        async for chunk in response.aiter_bytes(chunk_size=65_536):
            if received >= max_bytes:
                truncated = True
                break
            remaining = max_bytes - received
            take = chunk if len(chunk) <= remaining else chunk[:remaining]
            if take:
                parts.append(take)
            received += len(take)
            if len(chunk) > len(take):
                truncated = True
                break
            if received >= max_bytes:
                break
        return (b"".join(parts), truncated)

    async def _iter_sse_lines(self, response: httpx.Response) -> AsyncIterator[str]:
        """Yield raw SSE line chunks preserving local provider behavior."""
        async for line in response.aiter_lines():
            if line:
                yield f"{line}\n"
            else:
                yield "\n"

    async def _iter_sse_events(self, response: httpx.Response) -> AsyncIterator[str]:
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

    def _new_stream_state(self, request: Any, *, thinking_enabled: bool) -> Any:
        """Return per-stream provider state for event transformation."""
        if self.stream_chunk_mode == "line":
            return NativeSseBlockPolicyState()
        return None

    def _transform_stream_event(
        self,
        event: str,
        state: Any,
        *,
        thinking_enabled: bool,
    ) -> str | None:
        """Transform or drop a grouped SSE event before yielding it downstream."""
        if isinstance(state, NativeSseBlockPolicyState):
            return transform_native_sse_block_event(
                event, state, thinking_enabled=thinking_enabled
            )
        return event

    def _get_error_message(self, error: Exception, request_id: str | None) -> str:
        """Map an exception into a user-facing provider error message."""
        mapped_error = map_error(error, rate_limiter=self._global_rate_limiter)
        base_message = user_visible_message_for_mapped_provider_error(
            mapped_error,
            provider_name=self._provider_name,
            read_timeout_s=self._config.http_read_timeout,
            detail=extract_provider_error_detail(error),
            request_id=request_id,
        )
        return base_message

    async def _validated_stream_send(
        self, body: dict, *, req_tag: str
    ) -> httpx.Response:
        """Send request and raise mapped HTTP errors before yielding body chunks."""
        send_response = await self._send_stream_request(body)
        if send_response.status_code != 200:
            try:
                await self._raise_for_status(send_response, req_tag=req_tag)
            finally:
                if not send_response.is_closed:
                    await _maybe_await_aclose(send_response)
        return send_response

    def _emit_error_events(
        self,
        *,
        request: Any,
        input_tokens: int,
        error_message: str,
        sent_any_event: bool,
    ) -> Iterator[str]:
        """Emit the same Anthropic message lifecycle used by OpenAI-compat providers."""
        yield from iter_provider_stream_error_sse_events(
            request=request,
            input_tokens=input_tokens,
            error_message=error_message,
            sent_any_event=sent_any_event,
            log_raw_sse_events=self._config.log_raw_sse_events,
        )

    async def _iter_stream_chunks(
        self,
        response: httpx.Response,
        *,
        state: Any,
        thinking_enabled: bool,
    ) -> AsyncIterator[str]:
        """Yield stream chunks according to the provider's observable chunk shape."""
        if self.stream_chunk_mode == "line" and isinstance(
            state, NativeSseBlockPolicyState
        ):
            async for event in self._iter_sse_events(response):
                output_event = self._transform_stream_event(
                    event,
                    state,
                    thinking_enabled=thinking_enabled,
                )
                if output_event is None:
                    continue
                for line in output_event.splitlines(keepends=True):
                    yield line
            return

        if self.stream_chunk_mode == "line":
            async for chunk in self._iter_sse_lines(response):
                yield chunk
            return

        async for event in self._iter_sse_events(response):
            output_event = self._transform_stream_event(
                event,
                state,
                thinking_enabled=thinking_enabled,
            )
            if output_event is not None:
                yield output_event

    async def _collect_native_recovery_text(
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
                response = await self._global_rate_limiter.execute_with_retry(
                    self._validated_stream_send, body, req_tag=req_tag
                )
                state = self._new_stream_state(None, thinking_enabled=thinking_enabled)
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
                for event in parse_sse_text("".join(chunks)):
                    delta = event.data.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    text = delta.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
                    thinking = delta.get("thinking")
                    if isinstance(thinking, str):
                        thinking_parts.append(thinking)
                return "".join(text_parts), "".join(thinking_parts)
            except Exception as error:
                last_error = error
                if not is_retryable_stream_error(error):
                    raise
                trace_event(
                    stage="provider",
                    event="provider.recovery.retry",
                    source="provider",
                    provider=self._provider_name,
                    recovery_kind="native_text",
                    attempt=attempt + 1,
                    max_attempts=MIDSTREAM_RECOVERY_ATTEMPTS,
                    exc_type=type(error).__name__,
                )
            finally:
                if response is not None and not response.is_closed:
                    await _maybe_await_aclose(response)
        if last_error is not None:
            raise last_error
        return "", ""

    async def _native_recovery_events(
        self,
        *,
        body: dict[str, Any],
        request: Any,
        tracker: EmittedNativeSseTracker,
        error: Exception,
        request_id: str | None,
        req_tag: str,
        thinking_enabled: bool,
    ) -> list[str] | None:
        if not is_retryable_stream_error(error):
            return None

        schemas = tool_schemas_by_name(request)
        if tracker.has_tool_block():
            repair_events: list[str] = []
            for index, block in enumerate(tracker.tool_blocks()):
                if (
                    block.tool_id
                    and block.name
                    and parse_complete_tool_input(block.content, block.name, schemas)
                    is not None
                ):
                    continue
                schema = schemas.get(block.name)
                recovery_body = make_native_tool_repair_body(
                    body,
                    tool_name=block.name,
                    prefix=block.content,
                    input_schema=schema.input_schema if schema is not None else None,
                )
                accepted_suffix: str | None = None
                for attempt in range(MIDSTREAM_RECOVERY_ATTEMPTS):
                    text, _ = await self._collect_native_recovery_text(
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
                            provider=self._provider_name,
                            tool_name=block.name,
                            attempt=attempt + 1,
                        )
                        break
                if accepted_suffix is None:
                    return None
                repair_events.extend(
                    tracker.append_tool_repair_suffix(index, accepted_suffix)
                )

            if not tracker.can_salvage_tool_use(schemas):
                return None
            events = list(repair_events)
            events.extend(tracker.iter_success_tail("tool_use"))
            trace_event(
                stage="provider",
                event="provider.recovery.tool_salvaged",
                source="provider",
                provider=self._provider_name,
                request_id=request_id,
            )
            return events

        partial_text = tracker.emitted_text()
        partial_thinking = tracker.emitted_thinking()
        if not partial_text and not partial_thinking:
            return None
        recovery_body = make_native_text_recovery_body(body, partial_text)
        text, thinking = await self._collect_native_recovery_text(
            recovery_body,
            req_tag=req_tag,
            thinking_enabled=thinking_enabled,
        )
        text_suffix = continuation_suffix(partial_text, text)
        thinking_suffix = continuation_suffix(partial_thinking, thinking)
        events: list[str] = []
        if thinking_suffix:
            events.extend(tracker.append_thinking_suffix(thinking_suffix))
        if text_suffix:
            events.extend(tracker.append_text_suffix(text_suffix))
        if not events:
            return None
        events.extend(tracker.iter_success_tail("end_turn"))
        trace_event(
            stage="provider",
            event="provider.recovery.continued",
            source="provider",
            provider=self._provider_name,
            request_id=request_id,
        )
        return events

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream response via a native Anthropic-compatible messages endpoint."""
        tag = self._provider_name
        req_tag = f" request_id={request_id}" if request_id else ""
        body = self._build_request_body(request, thinking_enabled=thinking_enabled)
        thinking_enabled = self._is_thinking_enabled(request, thinking_enabled)

        trace_event(
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider=self._provider_name,
            gateway_model=request.model,
            downstream_model=body.get("model"),
            message_count=len(body.get("messages", [])),
            tool_count=len(body.get("tools", [])),
            body=provider_native_messages_body_snapshot(body),
        )

        response: httpx.Response | None = None
        sent_any_event = False
        state = self._new_stream_state(request, thinking_enabled=thinking_enabled)
        emitted_tracker = EmittedNativeSseTracker()
        recovery_session = StreamRecoverySession(
            provider_name=self._provider_name,
            request_id=request_id,
        )

        async with self._global_rate_limiter.concurrency_slot():
            while True:
                stream_opened = False
                try:
                    response = await self._global_rate_limiter.execute_with_retry(
                        self._validated_stream_send, body, req_tag=req_tag
                    )
                    stream_opened = True

                    chunk_count = 0
                    chunk_bytes = 0

                    async for chunk in self._iter_stream_chunks(
                        response,
                        state=state,
                        thinking_enabled=thinking_enabled,
                    ):
                        chunk_count += 1
                        chunk_bytes += len(chunk.encode("utf-8", errors="replace"))
                        emitted_tracker.feed(chunk)
                        for event in recovery_session.push(chunk):
                            sent_any_event = True
                            yield event

                    if not emitted_tracker.has_terminal_message():
                        raise TruncatedProviderStreamError(
                            "Provider stream ended without message_stop."
                        )

                    trace_event(
                        stage="provider",
                        event="provider.response.completed",
                        source="provider",
                        provider=self._provider_name,
                        gateway_model=request.model,
                        sse_chunks_out=chunk_count,
                        sse_bytes_out=chunk_bytes,
                    )
                    for event in recovery_session.flush():
                        sent_any_event = True
                        yield event
                    return

                except Exception as error:
                    generated_output = emitted_tracker.has_content_block()
                    complete_tool_salvageable = (
                        generated_output
                        and emitted_tracker.can_salvage_tool_use(
                            tool_schemas_by_name(request)
                        )
                    )
                    decision = recovery_session.advance_failure(
                        error,
                        stream_opened=stream_opened,
                        generated_output=generated_output,
                        complete_tool_salvageable=complete_tool_salvageable,
                    )
                    if decision.action == StreamFailureAction.EARLY_RETRY:
                        if response is not None and not response.is_closed:
                            await _maybe_await_aclose(response)
                        response = None
                        state = self._new_stream_state(
                            request, thinking_enabled=thinking_enabled
                        )
                        emitted_tracker = EmittedNativeSseTracker()
                        sent_any_event = False
                        continue

                    if decision.action == StreamFailureAction.MIDSTREAM_RECOVERY:
                        try:
                            recovery_events = await self._native_recovery_events(
                                body=body,
                                request=request,
                                tracker=emitted_tracker,
                                error=error,
                                request_id=request_id,
                                req_tag=req_tag,
                                thinking_enabled=thinking_enabled,
                            )
                        except Exception as recovery_error:
                            trace_event(
                                stage="provider",
                                event="provider.recovery.failed",
                                source="provider",
                                provider=self._provider_name,
                                request_id=request_id,
                                exc_type=type(recovery_error).__name__,
                            )
                            recovery_events = None
                        if recovery_events is not None:
                            for event in recovery_session.flush_uncommitted(decision):
                                sent_any_event = True
                                yield event
                            for event in recovery_events:
                                yield event
                            return

                    if not isinstance(error, httpx.HTTPStatusError):
                        self._log_stream_transport_error(
                            tag, req_tag, error, request_id=request_id
                        )
                    error_message = self._get_error_message(error, request_id)

                    if response is not None and not response.is_closed:
                        await _maybe_await_aclose(response)

                    trace_event(
                        stage="provider",
                        event="provider.response.error",
                        source="provider",
                        provider=self._provider_name,
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
                            for event in recovery_session.flush():
                                sent_any_event = True
                                yield event
                        for event in emitted_tracker.iter_close_unclosed_blocks():
                            yield event
                        for event in emitted_tracker.iter_midstream_error_tail(
                            error_message,
                            request=request,
                            input_tokens=input_tokens,
                            log_raw_sse_events=self._config.log_raw_sse_events,
                        ):
                            yield event
                    else:
                        recovery_session.discard()
                        for event in self._emit_error_events(
                            request=request,
                            input_tokens=input_tokens,
                            error_message=error_message,
                            sent_any_event=False,
                        ):
                            yield event
                    return
                finally:
                    if response is not None and not response.is_closed:
                        await _maybe_await_aclose(response)
