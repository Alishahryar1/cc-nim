"""Standard API-key OpenAI Responses execution over the official SDK."""

import asyncio
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
    ProviderOperationKind,
)
from free_claude_code.providers.failure_policy import (
    RetryableProviderProtocolError,
    classify_provider_failure,
    is_retryable_stream_error,
)
from free_claude_code.providers.http import ProviderAttemptScope, maybe_await_aclose
from free_claude_code.providers.openai_compat import (
    OpenAIToolNameCodec,
    openai_replay_scope,
)
from free_claude_code.providers.stream_recovery import (
    RecoveryController,
    RecoveryFailureAction,
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
        return self._run_stream(
            body,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            tool_names=tool_names,
            replay_scope=replay_scope,
        )

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

    async def _run_stream(
        self,
        body: ResponseCreateParamsStreaming,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        tool_names: OpenAIToolNameCodec,
        replay_scope: ReplayCompatibilityScope,
    ) -> AsyncIterator[InferenceEvent]:
        execution = self._admission.start_execution(request_id=request_id)
        provider_stream = self._run_execution(
            body,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            tool_names=tool_names,
            replay_scope=replay_scope,
            execution=execution,
        )
        try:
            async for event in provider_stream:
                yield event
        except asyncio.CancelledError:
            raise
        except Exception as error:
            execution.fail(error)
            raise
        else:
            execution.succeed()
        finally:
            await maybe_await_aclose(provider_stream)
            execution.abandon()

    async def _run_execution(
        self,
        body: ResponseCreateParamsStreaming,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        tool_names: OpenAIToolNameCodec,
        replay_scope: ReplayCompatibilityScope,
        execution: ProviderExecution,
    ) -> AsyncIterator[InferenceEvent]:
        recovery = RecoveryController()
        response_id = f"response_{uuid.uuid4().hex}"
        trace_event(
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider=self._provider_name,
            request_id=request_id,
            execution_id=execution.execution_id,
            gateway_model=response_model,
            downstream_model=body.get("model"),
            transport="responses",
        )

        while execution.can_attempt:
            stream_adapter = ResponsesEventDecoder(
                response_id=response_id,
                model=response_model,
                input_tokens=input_tokens,
                tool_names=tool_names,
                replay_scope=replay_scope,
            )
            for event in stream_adapter.start():
                for held in recovery.push(event):
                    yield held

            scope: ProviderAttemptScope | None = None
            stream_opened = False
            try:
                attempt = await execution.open_attempt(ProviderOperationKind.GENERATION)
                scope = ProviderAttemptScope(
                    attempt,
                    provider_name=self._provider_name,
                    request_id=request_id,
                )
                sdk_stream = await self._client.responses.create(**body)
                stream = scope.retain(_ClosableResponsesStream(sdk_stream))
                stream_opened = True

                async for upstream_event in stream:
                    if not scope.attempt.accepted:
                        await scope.attempt.accept()
                    for event in stream_adapter.feed(
                        upstream_event.type,
                        upstream_event.to_dict(mode="json"),
                    ):
                        for held in recovery.push(event):
                            yield held
                if not stream_adapter.completed:
                    raise _TruncatedResponsesStream(
                        "Provider Responses stream ended without a terminal event."
                    )
                for event in recovery.flush():
                    yield event
                trace_event(
                    stage="provider",
                    event="provider.response.completed",
                    source="provider",
                    provider=self._provider_name,
                    request_id=request_id,
                    transport="responses",
                )
                return
            except asyncio.CancelledError, GeneratorExit:
                raise
            except Exception as raw_error:
                error = _effective_error(raw_error)
                attempt_failure = None
                if scope is not None and not scope.attempt.accepted:
                    attempt_failure = await scope.attempt.fail(error)
                if attempt_failure is not None and attempt_failure.retry_allowed:
                    recovery.discard()
                    _trace_early_retry(
                        provider_name=self._provider_name,
                        request_id=request_id,
                        execution=execution,
                    )
                    continue

                retryable = (
                    attempt_failure.retryable
                    if attempt_failure is not None
                    else is_retryable_stream_error(error)
                )
                decision = recovery.advance_failure(
                    retryable=retryable,
                    stream_opened=stream_opened,
                    generated_output=recovery.committed,
                    complete_tool_salvageable=False,
                    attempts_remaining=execution.attempts_remaining,
                )
                if decision.action is RecoveryFailureAction.EARLY_RETRY:
                    recovery.discard()
                    _trace_early_retry(
                        provider_name=self._provider_name,
                        request_id=request_id,
                        execution=execution,
                    )
                    continue

                failure = classify_provider_failure(
                    error,
                    provider_name=self._provider_name,
                    read_timeout_s=self._read_timeout_s,
                    request_id=request_id,
                )
                trace_event(
                    stage="provider",
                    event="provider.response.error",
                    source="provider",
                    provider=self._provider_name,
                    request_id=request_id,
                    transport="responses",
                    exc_type=type(error).__name__,
                    failure_kind=failure.kind.value,
                    status_code=failure.status_code,
                    provider_retryable=failure.retryable,
                )
                if not decision.committed:
                    recovery.discard()
                    raise failure from raw_error
                for event in stream_adapter.ledger.close_unclosed_blocks():
                    yield event
                raise failure from raw_error
            finally:
                if scope is not None:
                    await scope.aclose(active_error=sys.exception())

        if execution.last_failure is not None:
            raise execution.last_failure
        raise RuntimeError("Responses execution ended without a terminal result.")


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


def _trace_early_retry(
    *,
    provider_name: str,
    request_id: str | None,
    execution: ProviderExecution,
) -> None:
    trace_event(
        stage="provider",
        event="provider.recovery.early_retry",
        source="provider",
        provider=provider_name,
        request_id=request_id,
        transport="responses",
        attempts_started=execution.attempts_started,
        max_attempts=execution.max_attempts,
    )
