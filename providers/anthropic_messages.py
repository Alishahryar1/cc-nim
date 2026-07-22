"""Shared transport for providers with native Anthropic Messages endpoints."""

from __future__ import annotations

import asyncio
import json
import ssl
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

import certifi
import httpx
from loguru import logger

from config.constants import (
    ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
    NATIVE_MESSAGES_ERROR_BODY_LOG_CAP_BYTES,
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
from providers.base import BaseProvider, ProviderConfig
from providers.error_mapping import (
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


def _build_relaxed_ssl_context() -> ssl.SSLContext:
    """SSL context matching httpx's default, with X.509 strict-mode relaxed.

    Some antivirus HTTPS-scanning proxies (observed with AVG) install a root
    CA whose Basic Constraints extension isn't marked critical. Windows'
    trust store accepts it, but OpenSSL's VERIFY_X509_STRICT flag (on by
    default in ssl.create_default_context() since Python 3.13) rejects it,
    surfacing as CERTIFICATE_VERIFY_FAILED and, mid-stream, as connections
    that stall until read-timeout with no error ever raised. Hostname
    verification and chain-of-trust checks stay fully enabled; only that one
    RFC-5280-strictness check is relaxed.
    """
    context = ssl.create_default_context(cafile=certifi.where())
    context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return context


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
            verify=_build_relaxed_ssl_context(),
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
            await response.aclose()

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
            if self._config.log_api_error_tracebacks:
                preview, truncated = await self._read_error_body_preview(
                    response, NATIVE_MESSAGES_ERROR_BODY_LOG_CAP_BYTES
                )
                if preview:
                    text = preview.decode("utf-8", errors="replace")
                    logger.error(
                        "{}_ERROR:{} HTTP {} body_preview_bytes={} truncated={}: {}",
                        self._provider_name,
                        req_tag,
                        response.status_code,
                        len(preview),
                        truncated,
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
                logger.error(
                    "{}_ERROR:{} HTTP {}{}",
                    self._provider_name,
                    req_tag,
                    response.status_code,
                    extra,
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

    def _format_error_message(self, base_message: str, request_id: str | None) -> str:
        """Apply provider-specific request-id formatting to an error message."""
        if request_id:
            return f"{base_message}\nRequest ID: {request_id}"
        return base_message

    def _get_error_message(self, error: Exception, request_id: str | None) -> str:
        """Map an exception into a user-facing provider error message."""
        mapped_error = map_error(error, rate_limiter=self._global_rate_limiter)
        base_message = user_visible_message_for_mapped_provider_error(
            mapped_error,
            provider_name=self._provider_name,
            read_timeout_s=self._config.http_read_timeout,
        )
        return self._format_error_message(base_message, request_id)

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

        logger.info(
            "{}_STREAM:{} natively passing Anthropic request model={} msgs={} tools={}",
            tag,
            req_tag,
            body.get("model"),
            len(body.get("messages", [])),
            len(body.get("tools", [])),
        )

        response: httpx.Response | None = None
        sent_any_event = False
        state = self._new_stream_state(request, thinking_enabled=thinking_enabled)
        emitted_tracker = EmittedNativeSseTracker()

        async with self._global_rate_limiter.concurrency_slot():
            try:

                async def _validated_stream_send() -> httpx.Response:
                    """Send request; raise inside retry loop on 429 so rate limiter can backoff."""
                    send_response = await self._send_stream_request(body)
                    if send_response.status_code == 429:
                        await send_response.aclose()
                        send_response.raise_for_status()
                    if send_response.status_code != 200:
                        try:
                            await self._raise_for_status(send_response, req_tag=req_tag)
                        finally:
                            if not send_response.is_closed:
                                await send_response.aclose()
                    return send_response

                response = await self._global_rate_limiter.execute_with_retry(
                    _validated_stream_send
                )

                # Accumulate thinking text and tool_use IDs for the thinking cache.
                # Chunks are individual SSE lines; buffer them until an empty line
                # marks the end of a complete event, then extract what we need.
                _ev_lines: list[str] = []
                _thinking_parts: list[str] = []
                _tool_ids: set[str] = set()

                def _proc_buf() -> None:
                    ev_name = None
                    ev_data = None
                    for ln in _ev_lines:
                        if ln.startswith("event:"):
                            ev_name = ln[6:].strip()
                        elif ln.startswith("data:"):
                            ev_data = ln[5:].lstrip()
                    _ev_lines.clear()
                    if not ev_name or not ev_data:
                        return
                    try:
                        p = json.loads(ev_data)
                    except Exception:
                        return
                    if ev_name == "content_block_delta":
                        d = p.get("delta") or {}
                        if isinstance(d, dict) and d.get("type") == "thinking_delta":
                            t = d.get("thinking")
                            if isinstance(t, str):
                                _thinking_parts.append(t)
                    elif ev_name == "content_block_start":
                        blk = p.get("content_block") or {}
                        if isinstance(blk, dict) and blk.get("type") == "tool_use":
                            tid = blk.get("id")
                            if isinstance(tid, str) and tid:
                                _tool_ids.add(tid)

                async for chunk in self._iter_stream_chunks(
                    response,
                    state=state,
                    thinking_enabled=thinking_enabled,
                ):
                    s = chunk.rstrip("\r\n")
                    if s:
                        _ev_lines.append(s)
                    elif _ev_lines:
                        _proc_buf()
                    sent_any_event = True
                    emitted_tracker.feed(chunk)
                    yield chunk

                if _ev_lines:
                    _proc_buf()
                if _thinking_parts and _tool_ids:
                    from core.thinking_cache import ThinkingCache
                    ThinkingCache.store(frozenset(_tool_ids), "".join(_thinking_parts))

            except asyncio.CancelledError:
                # Client disconnected or gave up waiting (e.g. its own request
                # timeout fired) mid-stream. This must stay a bare re-raise:
                # asyncio.CancelledError is BaseException, not Exception, and
                # swallowing it here would break task cancellation semantics.
                # We only add logging so silent stalls are visible in server.log
                # instead of leaving zero trace of what happened to the request.
                logger.warning(
                    "{}_STREAM:{} cancelled (client disconnected or gave up) "
                    "after receiving upstream data={}",
                    tag,
                    req_tag,
                    sent_any_event,
                )
                if response is not None and not response.is_closed:
                    await response.aclose()
                raise
            except Exception as error:
                if not isinstance(error, httpx.HTTPStatusError):
                    self._log_stream_transport_error(tag, req_tag, error)
                error_message = self._get_error_message(error, request_id)

                if response is not None and not response.is_closed:
                    await response.aclose()

                logger.info(
                    "{}_STREAM: Emitting native SSE error event for {}{}",
                    tag,
                    type(error).__name__,
                    req_tag,
                )
                if sent_any_event:
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
                    await response.aclose()
