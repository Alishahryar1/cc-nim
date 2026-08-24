"""Standard API-key OpenAI Responses execution over the official SDK."""

import sys
import uuid
from collections.abc import AsyncIterator
from typing import cast

from openai import AsyncOpenAI, AsyncStream
from openai.types.responses import ResponseStreamEvent
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.diagnostics import extract_upstream_error_detail
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.inference import (
    InferenceEvent,
    InferenceRequest,
    ReplayCompatibilityScope,
)
from free_claude_code.core.openai_responses import (
    ResponsesConversionError,
)
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderExecution,
)
from free_claude_code.providers.failure_policy import (
    RetryableProviderProtocolError,
    classify_provider_failure,
    is_retryable_stream_error,
)
from free_claude_code.providers.http import close_provider_stream
from free_claude_code.providers.openai_compat import (
    OpenAIToolNameCodec,
    openai_replay_scope,
)
from free_claude_code.providers.streaming import (
    PublicationBuffer,
    StreamExecutionSupervisor,
    StreamFeed,
    StreamTraceContext,
)

from .events import ResponsesEventDecoder, ResponsesStreamFailure
from .request_codec import ResponsesRequestEncodingError, build_responses_request_body


class _TruncatedResponsesStream(RetryableProviderProtocolError):
    """A Responses stream ended without a terminal lifecycle event."""


class _ClosableResponsesStream(AsyncIterator[ResponseStreamEvent]):
    """Expose the OpenAI SDK stream through the shared ``aclose`` contract."""

    def __init__(self, stream: AsyncStream[ResponseStreamEvent]) -> None:
        self._stream = stream

    def __aiter__(self) -> AsyncIterator[ResponseStreamEvent]:
        return self

    async def __anext__(self) -> ResponseStreamEvent:
        return await anext(self._stream)

    async def aclose(self) -> None:
        await self._stream.close()


class _ResponsesStreamEpoch(AsyncIterator[ResponseStreamEvent]):
    """One Responses raw stream and its fresh decoder epoch."""

    def __init__(
        self,
        stream: _ClosableResponsesStream,
        *,
        decoder: ResponsesEventDecoder,
        provider_name: str,
        request_id: str | None,
    ) -> None:
        self._stream = stream
        self._decoder = decoder
        self._provider_name = provider_name
        self._request_id = request_id

    def __aiter__(self) -> AsyncIterator[ResponseStreamEvent]:
        return self

    async def __anext__(self) -> ResponseStreamEvent:
        return await anext(self._stream)

    async def aclose(self) -> None:
        await self._stream.aclose()

    @property
    def recovery_snapshot(self) -> None:
        return None

    def start(self) -> StreamFeed:
        return StreamFeed(tuple(self._decoder.start()))

    def feed(self, raw: ResponseStreamEvent) -> StreamFeed:
        events = tuple(self._decoder.feed(raw.type, raw.to_dict(mode="json")))
        return StreamFeed(events, terminal=self._decoder.completed)

    def finish(self) -> StreamFeed:
        if not self._decoder.completed:
            raise _TruncatedResponsesStream(
                "Provider Responses stream ended without a terminal event."
            )
        return StreamFeed(terminal=True)

    def failure_events(self) -> tuple[InferenceEvent, ...]:
        return tuple(self._decoder.ledger.close_unclosed_blocks())

    def trace_completed(self) -> None:
        trace_event(
            stage="provider",
            event="provider.response.completed",
            source="provider",
            provider=self._provider_name,
            request_id=self._request_id,
            transport="responses",
        )


class _ResponsesAttemptSource:
    """Request-scoped standard Responses connector and failure policy."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        body: ResponseCreateParamsStreaming,
        provider_name: str,
        read_timeout_s: float,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        tool_names: OpenAIToolNameCodec,
        replay_scope: ReplayCompatibilityScope,
    ) -> None:
        self._client = client
        self._body = body
        self._provider_name = provider_name
        self._read_timeout_s = read_timeout_s
        self._input_tokens = input_tokens
        self._request_id = request_id
        self._response_model = response_model
        self._tool_names = tool_names
        self._replay_scope = replay_scope
        self._response_id = f"response_{uuid.uuid4().hex}"

    @property
    def trace_context(self) -> StreamTraceContext:
        return StreamTraceContext(
            provider_name=self._provider_name,
            request_id=self._request_id,
            transport="responses",
        )

    @property
    def failure_override(self) -> None:
        return None

    def trace_started(self, execution: ProviderExecution) -> None:
        trace_event(
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider=self._provider_name,
            request_id=self._request_id,
            execution_id=execution.execution_id,
            gateway_model=self._response_model,
            downstream_model=self._body.get("model"),
            transport="responses",
        )

    async def open(self) -> _ResponsesStreamEpoch:
        sdk_stream = await self._client.responses.create(**self._body)
        stream = _ClosableResponsesStream(sdk_stream)
        try:
            return _ResponsesStreamEpoch(
                stream,
                decoder=ResponsesEventDecoder(
                    response_id=self._response_id,
                    model=self._response_model,
                    input_tokens=self._input_tokens,
                    tool_names=self._tool_names,
                    replay_scope=self._replay_scope,
                ),
                provider_name=self._provider_name,
                request_id=self._request_id,
            )
        except Exception:
            await close_provider_stream(
                stream,
                active_error=sys.exception(),
                provider_name=self._provider_name,
                request_id=self._request_id,
            )
            raise

    def apply_correction(self, error: Exception) -> bool:
        del error
        return False

    def attempt_error(self, error: Exception) -> Exception:
        return _effective_error(error)

    def is_retryable(self, error: Exception) -> bool:
        return is_retryable_stream_error(_effective_error(error))

    def classify_failure(self, error: Exception) -> ExecutionFailure:
        effective_error = _effective_error(error)
        failure = classify_provider_failure(
            effective_error,
            provider_name=self._provider_name,
            read_timeout_s=self._read_timeout_s,
            request_id=self._request_id,
        )
        trace_event(
            stage="provider",
            event="provider.response.error",
            source="provider",
            provider=self._provider_name,
            request_id=self._request_id,
            transport="responses",
            exc_type=type(effective_error).__name__,
            failure_kind=failure.kind.value,
            status_code=failure.status_code,
            provider_retryable=failure.retryable,
        )
        return failure


class OpenAIResponsesTransport:
    """Execute public Responses requests with provider-owned retry semantics."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        admission: ProviderAdmissionController,
        provider_name: str,
        read_timeout_s: float,
    ) -> None:
        self._client = client
        self._admission = admission
        self._provider_name = provider_name
        self._read_timeout_s = read_timeout_s
        self._supervisor = StreamExecutionSupervisor(admission)

    def preflight_stream(
        self,
        request: InferenceRequest,
        *,
        provider_model: str,
        reasoning: ReasoningPolicy,
    ) -> None:
        self._build_body(request, provider_model=provider_model, reasoning=reasoning)

    def stream_response(
        self,
        request: InferenceRequest,
        *,
        provider_model: str,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[InferenceEvent]:
        body = self._build_body(
            request,
            provider_model=provider_model,
            reasoning=reasoning,
        )
        tool_names = OpenAIToolNameCodec.from_request(request)
        replay_scope = openai_replay_scope(
            self._provider_name,
            provider_model,
            replay_format="responses",
        )
        source = _ResponsesAttemptSource(
            client=self._client,
            body=body,
            provider_name=self._provider_name,
            read_timeout_s=self._read_timeout_s,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            tool_names=tool_names,
            replay_scope=replay_scope,
        )
        return self._supervisor.stream(source, publication=PublicationBuffer())

    def _build_body(
        self,
        request: InferenceRequest,
        *,
        provider_model: str,
        reasoning: ReasoningPolicy,
    ) -> ResponseCreateParamsStreaming:
        try:
            return cast(
                ResponseCreateParamsStreaming,
                build_responses_request_body(
                    request,
                    provider_model=provider_model,
                    reasoning=reasoning,
                    tool_names=OpenAIToolNameCodec.from_request(request),
                    replay_scope=openai_replay_scope(
                        self._provider_name,
                        provider_model,
                        replay_format="responses",
                    ),
                ),
            )
        except (ResponsesConversionError, ResponsesRequestEncodingError) as exc:
            raise InvalidRequestError(str(exc)) from exc


def _effective_error(error: Exception) -> Exception:
    if not isinstance(error, ResponsesStreamFailure):
        return error
    message = (
        extract_upstream_error_detail(error).exception_text
        or "Provider response failed."
    )
    code = (error.code or "").lower()
    if "rate" in code or "429" in code:
        return ExecutionFailure(FailureKind.RATE_LIMIT, 429, message, True)
    if any(marker in code for marker in ("overload", "capacity", "529")):
        return ExecutionFailure(FailureKind.OVERLOADED, 529, message, True)
    retryable = any(
        marker in code for marker in ("server", "internal", "unavailable", "timeout")
    )
    return ExecutionFailure(FailureKind.UPSTREAM, 502, message, retryable)
