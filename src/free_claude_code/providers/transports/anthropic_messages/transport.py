"""Shared transport for providers with native Anthropic Messages endpoints."""

from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

import httpx

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.native_sse_block_policy import (
    NativeSseBlockPolicyState,
    transform_native_sse_block_event,
)
from free_claude_code.core.anthropic.streaming import AnthropicStreamLedger
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.error_mapping import (
    extract_provider_error_detail,
    map_error,
    user_visible_message_for_mapped_provider_error,
)
from free_claude_code.providers.model_listing import (
    ProviderModelInfo,
    extract_openai_model_ids,
    model_infos_from_ids,
)
from free_claude_code.providers.rate_limit import GlobalRateLimiter
from free_claude_code.providers.transports.http import maybe_await_aclose

from .http import model_list_json, raise_for_status_with_body
from .request_policy import (
    NativeMessagesRequestPolicy,
    build_native_messages_request_body,
)
from .stream import AnthropicMessagesStreamAdapter

StreamChunkMode = Literal["line", "event"]


class AnthropicMessagesTransport(BaseProvider):
    """Base class for providers that stream from an Anthropic-compatible endpoint."""

    stream_chunk_mode: StreamChunkMode = "line"

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider_name: str,
        default_base_url: str,
        admission: ProviderAdmissionController | None = None,
    ):
        super().__init__(config, admission=admission)
        self._provider_name = provider_name
        self._api_key = config.api_key
        self._base_url = (config.base_url or default_base_url).rstrip("/")
        self._request_policy = NativeMessagesRequestPolicy(provider_name=provider_name)
        self._global_rate_limiter = GlobalRateLimiter.get_scoped_instance(
            provider_name.lower(),
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
            max_concurrency=config.max_concurrency,
        )
        self._base_url = (config.base_url or default_base_url).rstrip("/")
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

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: Any = None,
    ) -> None:
        """Validate request conversion before streaming."""
        pass

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
        """Return the headers for the native messages request."""
        return {"Content-Type": "application/json"}

    def _request_headers_for_request(
        self, request: MessagesRequest | None
    ) -> dict[str, str]:
        """Return request headers, allowing provider-specific request context."""
        return self._request_headers()

    def _build_request_body(
        self, request: MessagesRequest, thinking_enabled: bool | None = None
    ) -> dict:
        """Build a native Anthropic request body."""
        thinking_enabled = self._is_thinking_enabled(request, thinking_enabled)
        return self._build_request_body_with_resolved_thinking(
            request,
            thinking_enabled=thinking_enabled,
        )

    def _build_request_body_with_resolved_thinking(
        self, request: MessagesRequest, *, thinking_enabled: bool
    ) -> dict:
        """Build a native Anthropic request body after thinking is resolved."""
        return build_native_messages_request_body(
            request,
            thinking_enabled=thinking_enabled,
            policy=self._request_policy,
        )

    def _messages_path(self, request: MessagesRequest | None = None) -> str:
        """Return the upstream Messages path for this provider."""
        return "/messages"

    async def _send_stream_request(
        self, body: dict, *, request_data: MessagesRequest | None = None
    ) -> httpx.Response:
        """Create a streaming messages response."""
        request = self._client.build_request(
            "POST",
            self._messages_path(request_data),
            json=body,
            headers=self._request_headers_for_request(request_data),
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

    def _new_stream_state(
        self, request: MessagesRequest, *, thinking_enabled: bool
    ) -> Any:
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
        return user_visible_message_for_mapped_provider_error(
            mapped_error,
            provider_name=self._provider_name,
            read_timeout_s=self._config.http_read_timeout,
            detail=extract_provider_error_detail(error),
            request_id=request_id,
        )

    async def _validated_stream_send(
        self,
        body: dict,
        *,
        req_tag: str,
        request_data: MessagesRequest | None = None,
    ) -> httpx.Response:
        """Send request and raise mapped HTTP errors before yielding body chunks."""
        send_response = await self._send_stream_request(body, request_data=request_data)
        if send_response.status_code != 200:
            try:
                await self._raise_for_status(send_response, req_tag=req_tag)
            finally:
                if not send_response.is_closed:
                    await maybe_await_aclose(send_response)
        return send_response

    def _emit_error_events(
        self,
        request: MessagesRequest,
        input_tokens: int,
        error_message: str,
        sent_any_event: bool = False,
    ) -> Iterator[str]:
        """Emit Anthropic message lifecycle error events."""
        ledger = AnthropicStreamLedger(
            "msg_err",
            request.model or "claude-3-5-sonnet-20241022",
            input_tokens,
            log_raw_events=self._config.log_raw_sse_events,
        )
        if sent_any_event:
            yield from ledger.midstream_error_tail(error_message)
        else:
            yield from ledger.emit_top_level_error(error_message)

    async def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream response via a native Anthropic-compatible messages endpoint."""
        adapter = AnthropicMessagesStreamAdapter(
            self,
            request=request,
            input_tokens=input_tokens,
            request_id=request_id,
            thinking_enabled=thinking_enabled,
        )
        async for event in adapter.run():
            yield event
