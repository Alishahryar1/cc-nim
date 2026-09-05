"""One logical generation lifecycle for Responses upstreams."""

import asyncio
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Protocol

from loguru import logger

from free_claude_code.core.diagnostics import (
    extract_upstream_error_detail,
    redacted_exception_traceback,
)
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    ResponsesStreamFailure,
    responses_stream_failure_from_event,
)
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderCorrectionAction,
    ProviderExecutionState,
    ProviderOperationKind,
)
from free_claude_code.providers.failure_policy import (
    RetryableProviderProtocolError,
    classify_provider_failure,
    context_window_exceeded_provider_failure,
    is_context_window_error_code,
    is_retryable_stream_error,
    provider_authentication_status,
    reports_context_window_incomplete,
)
from free_claude_code.providers.http import ProviderAttemptScope, close_provider_stream
from free_claude_code.providers.stream_recovery import (
    RecoveryController,
    RecoveryFailureAction,
    TruncatedProviderStreamError,
)

from .events import ResponsesEventSource
from .presentation import ResponsesPresenterFactory

type AuthenticationRecovery = Callable[[], Awaitable[None]]


class ResponsesBackend(Protocol):
    """Request-local connection mechanics; generation replay belongs to the runner."""

    async def prepare_attempt(self) -> None: ...

    async def open_attempt(
        self, scope: ProviderAttemptScope
    ) -> ResponsesEventSource: ...

    def authentication_recovery(
        self, error: Exception
    ) -> AuthenticationRecovery | None: ...

    def normalize_error(self, error: Exception) -> Exception: ...

    async def aclose(self) -> None: ...


async def run_responses_stream(
    *,
    backend: ResponsesBackend,
    admission: ProviderAdmissionController,
    provider_name: str,
    request_id: str | None,
    response_model: str,
    body: JsonObject,
    read_timeout_s: float,
    presenter_factory: ResponsesPresenterFactory,
    log_error_tracebacks: bool = False,
) -> AsyncIterator[str]:
    """Own admission, replay, commitment, terminal outcome, and cleanup together."""
    execution = admission.start_execution(request_id=request_id)
    recovery = RecoveryController()
    authentication_recovered = False

    def failure_for(raw_error: Exception) -> ExecutionFailure:
        return classify_provider_failure(
            _effective_error(backend.normalize_error(raw_error)),
            provider_name=provider_name,
            read_timeout_s=read_timeout_s,
            request_id=request_id,
        )

    try:
        trace_event(
            stage="provider",
            event="provider.request.sent",
            source="provider",
            provider=provider_name,
            request_id=request_id,
            execution_id=execution.execution_id,
            gateway_model=response_model,
            downstream_model=body.get("model"),
            transport="responses",
        )
        while execution.can_attempt:
            # Credential I/O is not an inference attempt. Failure here is terminal
            # for this execution, including a failed forced snapshot refresh.
            await backend.prepare_attempt()
            presenter = presenter_factory()
            start_events = tuple(presenter.start())
            presenter_started = False
            scope: ProviderAttemptScope | None = None
            stream_opened = False
            try:
                attempt = await execution.open_attempt(ProviderOperationKind.GENERATION)
                scope = ProviderAttemptScope(
                    attempt, provider_name=provider_name, request_id=request_id
                )
                source = await backend.open_attempt(scope)
                stream_opened = True
                async for event_type, payload in source:
                    if not attempt.accepted:
                        await attempt.accept()
                    if not presenter_started:
                        presenter_started = True
                        for event in start_events:
                            for held in recovery.push(event):
                                yield held
                    if event_type in {"response.failed", "error", "response.error"}:
                        error = responses_stream_failure_from_event(event_type, payload)
                        try:
                            error.payload = source.normalize(event_type, payload)
                        except RetryableProviderProtocolError:
                            # Partial failure identity must not hide the real error.
                            error.payload = None
                        raise error
                    if reports_context_window_incomplete(event_type, payload):
                        raise context_window_exceeded_provider_failure()
                    payload = source.normalize(event_type, payload)
                    for event in presenter.feed(event_type, payload):
                        for held in recovery.push(event):
                            yield held
                    if presenter.completed:
                        break
                if not presenter.completed:
                    raise TruncatedProviderStreamError(
                        "Provider Responses stream ended without a terminal event."
                    )
                for event in recovery.flush():
                    yield event
                execution.succeed()
                trace_event(
                    stage="provider",
                    event="provider.response.completed",
                    source="provider",
                    provider=provider_name,
                    request_id=request_id,
                    transport="responses",
                )
                return
            except asyncio.CancelledError, GeneratorExit:
                raise
            except Exception as raw_error:
                error = _effective_error(backend.normalize_error(raw_error))
                correction = (
                    backend.authentication_recovery(raw_error)
                    if scope is not None
                    and not recovery.committed
                    and not authentication_recovered
                    and execution.can_attempt
                    else None
                )
                if correction is not None and scope is not None:
                    allowed = scope.attempt.accepted or (
                        await scope.attempt.correct(error)
                        is ProviderCorrectionAction.RETRY
                    )
                    if allowed:
                        closing_scope, scope = scope, None
                        await closing_scope.aclose(active_error=raw_error)
                        authentication_recovered = True
                        # Exceptions in recovery escape this handler; they cannot
                        # be reclassified as a retryable generation-open failure.
                        await correction()
                        recovery.discard()
                        continue
                attempt_failure = None
                if scope is not None and not scope.attempt.accepted:
                    attempt_failure = await scope.attempt.fail(error)
                decision = recovery.advance_failure(
                    retryable=(
                        attempt_failure.retryable
                        if attempt_failure is not None
                        else is_retryable_stream_error(error)
                    ),
                    stream_opened=stream_opened,
                    generated_output=recovery.committed,
                    complete_tool_salvageable=False,
                    attempts_remaining=execution.attempts_remaining,
                )
                if (
                    attempt_failure is not None and attempt_failure.retry_allowed
                ) or decision.action is RecoveryFailureAction.EARLY_RETRY:
                    recovery.discard()
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
                    continue
                failure = failure_for(raw_error)
                failure.__cause__ = raw_error
                execution.fail(failure)
                if not recovery.committed:
                    recovery.discard()
                    raise failure from raw_error
                for event in presenter.terminal_failure(raw_error, failure):
                    yield event
                if presenter.terminal_failure_completes_wire:
                    return
                raise failure from raw_error
            finally:
                if scope is not None:
                    await scope.aclose(active_error=sys.exception())
        if execution.last_failure is not None:
            raise execution.last_failure
        raise RuntimeError("Responses execution ended without a terminal result.")
    except asyncio.CancelledError, GeneratorExit:
        raise
    except ExecutionFailure as failure:
        execution.fail(failure)
        raise
    except Exception as raw_error:
        failure = failure_for(raw_error)
        execution.fail(failure)
        raise failure from raw_error
    finally:
        try:
            if (
                execution.state is ProviderExecutionState.FAILED
                and execution.last_failure is not None
            ):
                failure = failure_for(execution.last_failure)
                trace_event(
                    stage="provider",
                    event="provider.response.error",
                    source="provider",
                    provider=provider_name,
                    request_id=request_id,
                    transport="responses",
                    failure_kind=failure.kind.value,
                    status_code=failure.status_code,
                    provider_retryable=failure.retryable,
                )
                logger.error(
                    "{}_ERROR request_id={} failure_kind={} status={}{}",
                    provider_name,
                    request_id,
                    failure.kind.value,
                    failure.status_code,
                    "\n" + redacted_exception_traceback(execution.last_failure)
                    if log_error_tracebacks
                    else "",
                )
            await close_provider_stream(
                backend,
                active_error=sys.exception(),
                provider_name=provider_name,
                request_id=request_id,
            )
        finally:
            await execution.aclose()


def _effective_error(error: Exception) -> Exception:
    if not isinstance(error, ResponsesStreamFailure):
        return error
    message = (
        extract_upstream_error_detail(error).exception_text
        or "Provider response failed."
    )
    if is_context_window_error_code(error.code):
        return context_window_exceeded_provider_failure()
    auth_status = provider_authentication_status(error)
    if auth_status is not None:
        return ExecutionFailure(
            FailureKind.AUTHENTICATION
            if auth_status == 401
            else FailureKind.PERMISSION,
            auth_status,
            message,
            False,
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
