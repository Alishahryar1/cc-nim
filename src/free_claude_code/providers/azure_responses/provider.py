"""Azure OpenAI API backend provider using OpenAI Responses."""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.openai_tool_names import OpenAIToolNameCodec
from free_claude_code.core.diagnostics import (
    ERROR_DETAIL_DISPLAY_CAP_BYTES,
    attach_upstream_error_body,
    extract_upstream_error_detail,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.openai_responses import (
    ResponsesConversionError,
    ResponsesProviderStream,
    ResponsesStreamFailure,
    build_responses_provider_request,
)
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningPolicy,
)
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderAttempt,
)
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.failure_policy import (
    RetryableProviderProtocolError,
    classify_provider_failure,
)
from free_claude_code.providers.stream_recovery import RecoveryController

try:
    FCC_VERSION = version("free-claude-code")
except PackageNotFoundError:
    FCC_VERSION = "dev"


class _TruncatedResponsesStream(RetryableProviderProtocolError):
    """A Responses stream ended without a terminal lifecycle event."""


class AzureResponsesProvider(BaseProvider):
    """Use an Azure OpenAI subscription through the Responses API."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        admission: ProviderAdmissionController,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(config)
        self._admission = admission
        self._client_headers = {
            "api-key": config.api_key or "",
            "User-Agent": f"free-claude-code/{FCC_VERSION}",
        }
        self._client = client or httpx.AsyncClient(
            base_url=f"{config.base_url.rstrip('/')}/",
            proxy=config.proxy,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                write=config.http_write_timeout,
            ),
            headers=self._client_headers,
        )
        self._owns_client = client is None

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        """Validate and adapt the request before upstream I/O."""

        self._build_body(request, reasoning=reasoning)

    async def cleanup(self) -> None:
        """Close only provider-owned transport resources."""

        if self._owns_client:
            await self._client.aclose()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Discover models visible to the currently connected Azure account."""

        async def fetch() -> Any:
            response = await self._client.get(
                "models",
                headers=self._client_headers,
            )
            response.raise_for_status()
            return response.json()

        payload = await self._admission.run_with_retry(fetch)
        return _model_infos(payload)

    def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        """Stream Responses output in Anthropic Messages format."""

        tool_names = OpenAIToolNameCodec.from_request(request)
        body = self._build_body(request, reasoning=reasoning)
        return self._run_stream(
            body,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model or request.model,
            tool_names=tool_names,
        )

    @staticmethod
    def _build_body(
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> dict[str, Any]:
        try:
            body = build_responses_provider_request(request, reasoning=reasoning)
        except ResponsesConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc

        # Remove fields unsupported by Azure Responses API
        body.pop("store", None)
        body.pop("include", None)

        # Azure requires model name to match deployment
        if "model" in body:
            pass  # kept intact

        return body

    async def _run_stream(
        self,
        body: dict[str, Any],
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        tool_names: OpenAIToolNameCodec,
    ) -> AsyncIterator[str]:
        retry_session = self._admission.new_retry_session(request_id=request_id)
        recovery = RecoveryController()
        message_id = f"msg_{uuid.uuid4()}"
        trace_event(
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider="azure_openai",
            request_id=request_id,
            gateway_model=response_model,
            downstream_model=body.get("model"),
            item_count=len(body.get("input", [])),
            tool_count=len(body.get("tools", [])),
        )

        while retry_session.can_attempt:
            stream = ResponsesProviderStream(
                message_id=message_id,
                model=response_model,
                input_tokens=input_tokens,
                log_raw_events=self._config.log_raw_sse_events,
                tool_names=tool_names,
            )
            for event in stream.start():
                for held in recovery.push(event):
                    yield held

            response: httpx.Response | None = None
            attempt: ProviderAttempt | None = None
            stream_opened = False
            try:
                attempt = await self._admission.open_attempt(retry_session)
                response = await self._client.send(
                    self._client.build_request(
                        "POST",
                        "responses",
                        json=body,
                        headers={
                            **self._client_headers,
                            "Accept": "text/event-stream",
                        },
                    ),
                    stream=True,
                )

                if not response.is_success:
                    body_bytes, body_truncated = await _read_bounded_body(response)
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        attach_upstream_error_body(
                            exc,
                            body_bytes,
                            truncated=body_truncated,
                        )
                        raise

                content_type = response.headers.get("content-type")
                if content_type and "text/event-stream" not in content_type.lower():
                    body_bytes, body_truncated = await _read_bounded_body(response)
                    error = _TruncatedResponsesStream(
                        "Azure OpenAI returned a non-streaming Responses payload."
                    )
                    attach_upstream_error_body(
                        error,
                        body_bytes,
                        truncated=body_truncated,
                    )
                    raise error

                stream_opened = True

                async for event_type, payload in _iter_sse(response):
                    if not attempt.accepted:
                        await attempt.succeeded()
                    for event in stream.feed(event_type, payload):
                        for held in recovery.push(event):
                            yield held
                if not stream.completed:
                    raise _TruncatedResponsesStream(
                        "Azure OpenAI Responses stream ended without a terminal event."
                    )
                for event in recovery.flush():
                    yield event
                trace_event(
                    stage="provider",
                    event="provider.response.completed",
                    source="provider",
                    provider="azure_openai",
                    request_id=request_id,
                )
                return
            except asyncio.CancelledError, GeneratorExit:
                raise
            except Exception as raw_error:
                error = _effective_error(raw_error)
                if attempt is not None and not attempt.accepted:
                    await attempt.retry(error)
                retryable = (
                    attempt.failure_retryable
                    if attempt is not None and attempt.failure_retryable is not None
                    else None
                )
                decision = recovery.advance_failure(
                    error,
                    stream_opened=stream_opened,
                    generated_output=recovery.committed,
                    complete_tool_salvageable=False,
                    attempts_remaining=retry_session.attempts_remaining,
                    retryable_override=retryable,
                )
                if (
                    not decision.committed
                    and decision.retryable
                    and retry_session.can_attempt
                ):
                    recovery.discard()
                    trace_event(
                        stage="provider",
                        event="provider.recovery.early_retry",
                        source="provider",
                        provider="azure_openai",
                        request_id=request_id,
                        attempts_started=retry_session.attempts_started,
                        max_attempts=retry_session.max_attempts,
                    )
                    continue

                failure = classify_provider_failure(
                    error,
                    provider_name="Azure OpenAI",
                    read_timeout_s=self._config.http_read_timeout,
                    request_id=request_id,
                )
                self._log_stream_transport_error(
                    "AZURE_OPENAI",
                    f" request_id={request_id}" if request_id else "",
                    error,
                    request_id=request_id,
                )
                if not decision.committed:
                    recovery.discard()
                    raise failure from raw_error
                for event in stream.ledger.close_unclosed_blocks():
                    yield event
                raise failure from raw_error
            finally:
                if response is not None:
                    await response.aclose()
                if attempt is not None:
                    await attempt.aclose()

        raise RuntimeError(
            "Azure OpenAI retry session ended without a terminal result."
        )


async def _iter_sse(
    response: httpx.Response,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    event_type = ""
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if not data_lines:
                event_type = ""
                continue
            raw_data = "\n".join(data_lines)
            data_lines = []
            if raw_data == "[DONE]":
                return
            try:
                payload = json.loads(raw_data)
            except json.JSONDecodeError as exc:
                raise _TruncatedResponsesStream(
                    "Azure OpenAI returned malformed Responses SSE."
                ) from exc
            if not isinstance(payload, dict):
                raise _TruncatedResponsesStream(
                    "Azure OpenAI returned a non-object Responses event."
                )
            resolved_type = event_type or payload.get("type")
            event_type = ""
            if isinstance(resolved_type, str) and resolved_type:
                yield resolved_type, payload
            continue
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        raise _TruncatedResponsesStream(
            "Azure OpenAI Responses stream ended during an SSE event."
        )


async def _read_bounded_body(
    response: httpx.Response,
) -> tuple[bytes, bool]:
    limit = ERROR_DETAIL_DISPLAY_CAP_BYTES
    body = bytearray()
    async for chunk in response.aiter_bytes():
        remaining = limit + 1 - len(body)
        if remaining <= 0:
            break
        body.extend(chunk[:remaining])
        if len(body) > limit:
            break
    truncated = len(body) > limit
    return bytes(body[:limit]), truncated


def _model_infos(payload: Any) -> frozenset[ProviderModelInfo]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Azure OpenAI model-list response is missing the data array.")
    infos: set[ProviderModelInfo] = set()
    for model in payload["data"]:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        infos.add(
            ProviderModelInfo(
                model_id=model_id,
            )
        )
    return frozenset(infos)


def _effective_error(error: Exception) -> Exception:
    if isinstance(error, ResponsesStreamFailure):
        message = (
            extract_upstream_error_detail(error).exception_text
            or "Azure OpenAI response failed."
        )
        code = (error.code or "").lower()
        if "rate" in code or "429" in code:
            return ExecutionFailure(FailureKind.RATE_LIMIT, 429, message, True)
        if any(marker in code for marker in ("overload", "capacity", "529")):
            return ExecutionFailure(FailureKind.OVERLOADED, 529, message, True)
        retryable = any(
            marker in code
            for marker in ("server", "internal", "unavailable", "timeout")
        )
        return ExecutionFailure(FailureKind.UPSTREAM, 502, message, retryable)
    return error
