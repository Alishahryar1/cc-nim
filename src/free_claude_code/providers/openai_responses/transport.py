"""Standard API-key OpenAI Responses execution over the official SDK."""

import asyncio
import sys
import uuid
from collections.abc import AsyncIterator
from typing import cast

from openai import AsyncOpenAI, AsyncStream
from openai.types.responses import ResponseInputParam, ResponseStreamEvent
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.diagnostics import extract_upstream_error_detail
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesConversionError,
    ResponsesProviderStream,
    ResponsesStreamFailure,
    build_native_responses_request,
    build_responses_provider_request,
    responses_stream_failure_from_event,
)
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec
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
from free_claude_code.providers.stream_recovery import (
    RecoveryController,
    RecoveryFailureAction,
)

from .presentation import (
    MessagesResponsesPresenter,
    NativeResponsesPresenter,
    ResponsesExecutionOutcome,
    ResponsesPresenterFactory,
)


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
        log_raw_sse_events: bool,
    ) -> None:
        self._client = client
        self._admission = admission
        self._provider_name = provider_name
        self._read_timeout_s = read_timeout_s
        self._log_raw_sse_events = log_raw_sse_events

    def preflight_messages(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> None:
        self._build_messages_body(request, reasoning=reasoning)

    def stream_messages(
        self,
        request: MessagesRequest,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        body = self._build_messages_body(request, reasoning=reasoning)
        tool_names = OpenAIToolNameCodec.from_request(request)
        message_id = f"msg_{uuid.uuid4()}"
        return self._run_stream(
            body,
            request_id=request_id,
            response_model=response_model,
            presenter_factory=lambda: MessagesResponsesPresenter(
                ResponsesProviderStream(
                    message_id=message_id,
                    model=response_model,
                    input_tokens=input_tokens,
                    tool_names=tool_names,
                    log_raw_events=self._log_raw_sse_events,
                )
            ),
        )

    def preflight_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> None:
        self._build_native_body(request, reasoning=reasoning)

    def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        del input_tokens
        body = self._build_native_body(request, reasoning=reasoning)
        return self._run_stream(
            body,
            request_id=request_id,
            response_model=response_model,
            presenter_factory=lambda: NativeResponsesPresenter(
                public_model=response_model
            ),
        )

    @staticmethod
    def _build_messages_body(
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> JsonObject:
        try:
            return cast(
                JsonObject,
                cast(
                    ResponseCreateParamsStreaming,
                    build_responses_provider_request(request, reasoning=reasoning),
                ),
            )
        except ResponsesConversionError as exc:
            raise InvalidRequestError(str(exc)) from exc

    @staticmethod
    def _build_native_body(
        request: OpenAIResponsesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> JsonObject:
        if not request.model.strip():
            raise InvalidRequestError("Responses request model must not be empty.")
        if request.input is None or request.input == "" or request.input == []:
            raise InvalidRequestError("Responses request input must not be empty.")
        return build_native_responses_request(
            request,
            model=request.model,
            reasoning=reasoning,
        )

    async def _run_stream(
        self,
        body: JsonObject,
        *,
        request_id: str | None,
        response_model: str,
        presenter_factory: ResponsesPresenterFactory,
    ) -> AsyncIterator[str]:
        execution = self._admission.start_execution(request_id=request_id)
        outcome = ResponsesExecutionOutcome()
        provider_stream = self._run_execution(
            body,
            request_id=request_id,
            response_model=response_model,
            presenter_factory=presenter_factory,
            execution=execution,
            outcome=outcome,
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
            if outcome.failure is None:
                execution.succeed()
            else:
                execution.fail(outcome.failure)
        finally:
            await maybe_await_aclose(provider_stream)
            execution.abandon()

    async def _run_execution(
        self,
        body: JsonObject,
        *,
        request_id: str | None,
        response_model: str,
        presenter_factory: ResponsesPresenterFactory,
        execution: ProviderExecution,
        outcome: ResponsesExecutionOutcome,
    ) -> AsyncIterator[str]:
        recovery = RecoveryController()
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
            presenter = presenter_factory()
            start_events = tuple(presenter.start())
            presenter_started = False

            scope: ProviderAttemptScope | None = None
            stream_opened = False
            try:
                attempt = await execution.open_attempt(ProviderOperationKind.GENERATION)
                scope = ProviderAttemptScope(
                    attempt,
                    provider_name=self._provider_name,
                    request_id=request_id,
                )
                sdk_stream = await self._create_sdk_stream(body)
                stream = scope.retain(_ClosableResponsesStream(sdk_stream))
                stream_opened = True

                async for upstream_event in stream:
                    if not scope.attempt.accepted:
                        await scope.attempt.accept()
                    if not presenter_started:
                        presenter_started = True
                        for event in start_events:
                            for held in recovery.push(event):
                                yield held
                    payload = cast(
                        JsonObject,
                        upstream_event.to_dict(mode="json"),
                    )
                    if upstream_event.type in {
                        "response.failed",
                        "error",
                        "response.error",
                    }:
                        raise responses_stream_failure_from_event(
                            upstream_event.type,
                            payload,
                        )
                    for event in presenter.feed(upstream_event.type, payload):
                        for held in recovery.push(event):
                            yield held
                    if presenter.completed:
                        break
                if not presenter.completed:
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
                for event in presenter.terminal_failure(raw_error, failure):
                    yield event
                if presenter.terminal_failure_completes_wire:
                    outcome.failure = failure
                    return
                raise failure from raw_error
            finally:
                if scope is not None:
                    await scope.aclose(active_error=sys.exception())

        if execution.last_failure is not None:
            raise execution.last_failure
        raise RuntimeError("Responses execution ended without a terminal result.")

    async def _create_sdk_stream(
        self,
        body: JsonObject,
    ) -> AsyncStream[ResponseStreamEvent]:
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise InvalidRequestError("Responses request model must not be empty.")
        input_value = cast(str | ResponseInputParam, body.get("input"))
        extra_body = {
            key: value
            for key, value in body.items()
            if key not in {"model", "input", "stream", "store"}
        }
        return await self._client.responses.create(
            model=model,
            input=input_value,
            stream=True,
            store=False,
            extra_body=extra_body or None,
        )


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
