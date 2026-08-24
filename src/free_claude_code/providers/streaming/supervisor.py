"""Shared provider stream attempt, publication, and cleanup lifecycle."""

import asyncio
import sys
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.inference import InferenceEvent
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderCorrectionAction,
    ProviderExecution,
    ProviderOperationKind,
)
from free_claude_code.providers.failure_policy import ProviderFailureOverride
from free_claude_code.providers.http import ProviderAttemptScope

from .publication import PublicationBuffer


@dataclass(frozen=True, slots=True)
class StreamTraceContext:
    """Safe common trace dimensions for one request-scoped attempt source."""

    provider_name: str
    request_id: str | None
    transport: str | None = None
    recovery_kind: str | None = None


@dataclass(frozen=True, slots=True)
class StreamFeed:
    """Canonical events produced by one raw item or end-of-stream transition.

    A terminal feed is the logical end of the response. The supervisor must stop
    pulling the raw transport immediately, even if the transport has not reached
    physical EOF yet.
    """

    events: tuple[InferenceEvent, ...] = ()
    terminal: bool = False


class StreamAttemptEpoch[RawT, SnapshotT](Protocol):
    """One closeable raw stream and its fresh attempt-local decoder state."""

    def __aiter__(self) -> AsyncIterator[RawT]: ...

    async def __anext__(self) -> RawT: ...

    async def aclose(self) -> None: ...

    @property
    def recovery_snapshot(self) -> SnapshotT: ...

    def start(self) -> StreamFeed: ...

    def feed(self, raw: RawT) -> StreamFeed: ...

    def finish(self) -> StreamFeed: ...

    def failure_events(self) -> tuple[InferenceEvent, ...]: ...

    def trace_completed(self) -> None: ...


class BufferedAttemptEpoch[RawT, ResultT](Protocol):
    """One closeable recovery stream whose output stays private until complete."""

    def __aiter__(self) -> AsyncIterator[RawT]: ...

    async def __anext__(self) -> RawT: ...

    async def aclose(self) -> None: ...

    def feed(self, raw: RawT) -> None: ...

    def finish(self) -> ResultT: ...


class AttemptSource[EpochT](Protocol):
    """Request-scoped owner of one-call opening and provider failure semantics."""

    @property
    def trace_context(self) -> StreamTraceContext: ...

    @property
    def failure_override(self) -> ProviderFailureOverride | None: ...

    def trace_started(self, execution: ProviderExecution) -> None: ...

    async def open(self) -> EpochT: ...

    def apply_correction(self, error: Exception) -> bool: ...

    def attempt_error(self, error: Exception) -> Exception: ...

    def is_retryable(self, error: Exception) -> bool: ...

    def classify_failure(self, error: Exception) -> ExecutionFailure: ...


@dataclass(frozen=True, slots=True)
class RecoveryContext[SnapshotT]:
    """Accepted stream failure state offered to optional transport recovery."""

    error: Exception
    retryable: bool
    published: bool
    has_buffered: bool
    attempts_remaining: int
    snapshot: SnapshotT


@dataclass(frozen=True, slots=True)
class RecoveryOutcome:
    """Events and terminal state produced by a transport recovery strategy."""

    events: tuple[InferenceEvent, ...]
    publish_buffer: bool
    completed: bool


class StreamRecoveryStrategy[SnapshotT](Protocol):
    """Optional transport policy for recovery after an accepted stream fails."""

    def prefers_recovery(self, context: RecoveryContext[SnapshotT]) -> bool: ...

    async def resolve(
        self,
        context: RecoveryContext[SnapshotT],
        attempts: BoundAttemptOperations,
    ) -> RecoveryOutcome | None: ...


class BoundAttemptOperations:
    """Run private recovery attempts on one supervisor-owned execution budget."""

    def __init__(
        self,
        supervisor: StreamExecutionSupervisor,
        execution: ProviderExecution,
    ) -> None:
        self._supervisor = supervisor
        self._execution = execution

    @property
    def can_attempt(self) -> bool:
        return self._execution.can_attempt

    @property
    def attempts_remaining(self) -> int:
        return self._execution.attempts_remaining

    async def collect[RawT, ResultT](
        self,
        source: AttemptSource[BufferedAttemptEpoch[RawT, ResultT]],
        *,
        operation_kind: ProviderOperationKind,
    ) -> ResultT:
        """Collect one complete private stream using the shared attempt lifecycle."""
        last_error: Exception | None = None
        while self._execution.can_attempt:
            scope, epoch = await self._supervisor._open_epoch(
                source,
                self._execution,
                operation_kind=operation_kind,
            )
            error: Exception | None = None
            result: ResultT | None = None
            try:
                async for raw in epoch:
                    if not scope.attempt.accepted:
                        await scope.attempt.accept()
                    epoch.feed(raw)
                result = epoch.finish()
            except asyncio.CancelledError, GeneratorExit:
                await scope.aclose(active_error=sys.exception())
                raise
            except Exception as caught:
                error = caught

            if error is None:
                await scope.aclose(active_error=None)
                if result is None:
                    raise RuntimeError("buffered attempt completed without a result")
                return result

            last_error = error
            try:
                effective_error = source.attempt_error(error)
                if not scope.attempt.accepted:
                    decision = await scope.attempt.fail(
                        effective_error,
                        provider_failure_override=source.failure_override,
                    )
                    retryable = decision.retryable
                else:
                    retryable = source.is_retryable(error)
            except BaseException:
                await scope.aclose(active_error=sys.exception())
                raise
            await scope.aclose(active_error=error)
            if not retryable or not self._execution.can_attempt:
                raise error
            _trace_retry(
                source.trace_context,
                self._execution,
                operation_kind=operation_kind,
                error=error,
            )

        if last_error is not None:
            raise last_error
        raise RuntimeError("recovery collection ended without an attempt outcome")


class StreamExecutionSupervisor:
    """Compose admission attempts into one canonical provider stream lifecycle."""

    def __init__(self, admission: ProviderAdmissionController) -> None:
        self._admission = admission

    async def stream[RawT, SnapshotT](
        self,
        source: AttemptSource[StreamAttemptEpoch[RawT, SnapshotT]],
        *,
        publication: PublicationBuffer,
        recovery: StreamRecoveryStrategy[SnapshotT] | None = None,
    ) -> AsyncIterator[InferenceEvent]:
        """Run one request-scoped source through the shared logical lifecycle."""
        execution = self._admission.start_execution(
            request_id=source.trace_context.request_id
        )
        provider_stream: AsyncGenerator[InferenceEvent] | None = None
        try:
            source.trace_started(execution)
            provider_stream = self._run_execution(
                source,
                publication=publication,
                recovery=recovery,
                execution=execution,
            )
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
            if provider_stream is not None:
                await provider_stream.aclose()
            execution.abandon()

    async def _run_execution[RawT, SnapshotT](
        self,
        source: AttemptSource[StreamAttemptEpoch[RawT, SnapshotT]],
        *,
        publication: PublicationBuffer,
        recovery: StreamRecoveryStrategy[SnapshotT] | None,
        execution: ProviderExecution,
    ) -> AsyncGenerator[InferenceEvent]:
        while execution.can_attempt:
            scope, epoch = await self._open_epoch(
                source,
                execution,
                operation_kind=ProviderOperationKind.GENERATION,
            )
            pending_read: asyncio.Task[RawT] | None = None
            terminal_events: list[InferenceEvent] = []
            error: Exception | None = None
            try:
                for event in epoch.start().events:
                    for publishable in publication.push(event):
                        yield publishable
                iterator = aiter(epoch)
                while True:
                    if pending_read is None:
                        pending_read = asyncio.create_task(_read_next(iterator))
                    remaining = publication.seconds_until_deadline
                    if remaining is not None:
                        if remaining == 0:
                            for publishable in publication.flush():
                                yield publishable
                            continue
                        done, _pending = await asyncio.wait(
                            {pending_read},
                            timeout=remaining,
                        )
                        if pending_read not in done:
                            for publishable in publication.flush():
                                yield publishable
                            continue
                    try:
                        raw = await pending_read
                    except StopAsyncIteration:
                        pending_read = None
                        break
                    pending_read = None
                    if not scope.attempt.accepted:
                        await scope.attempt.accept()
                    feed = epoch.feed(raw)
                    if feed.terminal:
                        terminal_events.extend(feed.events)
                        break
                    else:
                        for event in feed.events:
                            for publishable in publication.push(event):
                                yield publishable
                finished = epoch.finish()
                if not finished.terminal:
                    raise RuntimeError("stream epoch finished without terminal events")
                terminal_events.extend(finished.events)
            except asyncio.CancelledError, GeneratorExit:
                if pending_read is not None:
                    if not pending_read.done():
                        pending_read.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await pending_read
                    pending_read = None
                await scope.aclose(active_error=sys.exception())
                raise
            except Exception as caught:
                error = caught
            finally:
                if pending_read is not None:
                    if not pending_read.done():
                        pending_read.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await pending_read

            if error is None:
                await scope.aclose(active_error=None)
                epoch.trace_completed()
                for event in terminal_events:
                    for publishable in publication.push(event):
                        yield publishable
                for publishable in publication.flush():
                    yield publishable
                return

            if not scope.attempt.accepted:
                try:
                    effective_error = source.attempt_error(error)
                    decision = await scope.attempt.fail(
                        effective_error,
                        provider_failure_override=source.failure_override,
                    )
                except BaseException:
                    await scope.aclose(active_error=sys.exception())
                    raise
                if decision.retry_allowed:
                    await scope.aclose(active_error=error)
                    publication.discard()
                    _trace_retry(
                        source.trace_context,
                        execution,
                        operation_kind=ProviderOperationKind.GENERATION,
                        error=error,
                    )
                    continue
                try:
                    failure = source.classify_failure(error)
                except BaseException:
                    await scope.aclose(active_error=sys.exception())
                    raise
                await scope.aclose(active_error=failure)
                publication.discard()
                raise failure from error

            try:
                retryable = source.is_retryable(error)
                context = RecoveryContext(
                    error=error,
                    retryable=retryable,
                    published=publication.published,
                    has_buffered=publication.has_buffered,
                    attempts_remaining=execution.attempts_remaining,
                    snapshot=epoch.recovery_snapshot,
                )
                prefer_recovery = recovery is not None and recovery.prefers_recovery(
                    context
                )
            except BaseException:
                await scope.aclose(active_error=sys.exception())
                raise
            if (
                retryable
                and not publication.published
                and execution.can_attempt
                and not prefer_recovery
            ):
                await scope.aclose(active_error=error)
                publication.discard()
                _trace_retry(
                    source.trace_context,
                    execution,
                    operation_kind=ProviderOperationKind.GENERATION,
                    error=error,
                )
                continue

            failure: ExecutionFailure | None = None
            if prefer_recovery:
                await scope.aclose(active_error=error)
            else:
                try:
                    failure = source.classify_failure(error)
                except BaseException:
                    await scope.aclose(active_error=sys.exception())
                    raise
                await scope.aclose(active_error=failure)

            outcome: RecoveryOutcome | None = None
            if recovery is not None:
                try:
                    outcome = await recovery.resolve(
                        context,
                        BoundAttemptOperations(self, execution),
                    )
                except asyncio.CancelledError, GeneratorExit:
                    raise
                except Exception as recovery_error:
                    trace_event(
                        stage="provider",
                        event="provider.recovery.failed",
                        source="provider",
                        provider=source.trace_context.provider_name,
                        request_id=source.trace_context.request_id,
                        exc_type=type(recovery_error).__name__,
                    )

            if outcome is not None and outcome.completed:
                if outcome.publish_buffer:
                    for publishable in publication.flush():
                        yield publishable
                elif not publication.published:
                    publication.discard()
                for event in outcome.events:
                    yield event
                return

            if failure is None:
                failure = source.classify_failure(error)
            if outcome is not None:
                if outcome.publish_buffer:
                    for publishable in publication.flush():
                        yield publishable
                elif not publication.published:
                    publication.discard()
                for event in outcome.events:
                    yield event
                raise failure from error

            if not publication.published:
                publication.discard()
                raise failure from error
            for event in epoch.failure_events():
                yield event
            raise failure from error

        if execution.last_failure is not None:
            raise execution.last_failure
        raise RuntimeError("provider stream execution ended without a terminal result")

    async def _open_epoch[EpochT](
        self,
        source: AttemptSource[EpochT],
        execution: ProviderExecution,
        *,
        operation_kind: ProviderOperationKind,
    ) -> tuple[ProviderAttemptScope, EpochT]:
        while execution.can_attempt:
            try:
                attempt = await execution.open_attempt(operation_kind)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise source.classify_failure(error) from error
            scope = ProviderAttemptScope(
                attempt,
                provider_name=source.trace_context.provider_name,
                request_id=source.trace_context.request_id,
            )
            try:
                epoch = await source.open()
            except asyncio.CancelledError:
                await scope.aclose(active_error=sys.exception())
                raise
            except Exception as error:
                try:
                    effective_error = source.attempt_error(error)
                    if source.apply_correction(error):
                        correction = await attempt.correct(effective_error)
                        await scope.aclose(active_error=error)
                        if correction is ProviderCorrectionAction.RETRY:
                            continue
                        raise source.classify_failure(error) from error
                    decision = await attempt.fail(
                        effective_error,
                        provider_failure_override=source.failure_override,
                    )
                    await scope.aclose(active_error=error)
                    if decision.retry_allowed:
                        _trace_retry(
                            source.trace_context,
                            execution,
                            operation_kind=operation_kind,
                            error=error,
                        )
                        continue
                    raise source.classify_failure(error) from error
                except BaseException:
                    await scope.aclose(active_error=sys.exception())
                    raise
            return scope, scope.retain(epoch)

        if execution.last_failure is not None:
            raise source.classify_failure(execution.last_failure)
        raise RuntimeError("provider execution ended without a final error")


def _trace_retry(
    context: StreamTraceContext,
    execution: ProviderExecution,
    *,
    operation_kind: ProviderOperationKind,
    error: Exception,
) -> None:
    if operation_kind is ProviderOperationKind.GENERATION:
        if context.transport is None:
            trace_event(
                stage="provider",
                event="provider.recovery.early_retry",
                source="provider",
                provider=context.provider_name,
                request_id=context.request_id,
                attempts_started=execution.attempts_started,
                max_attempts=execution.max_attempts,
                retryable=True,
            )
        else:
            trace_event(
                stage="provider",
                event="provider.recovery.early_retry",
                source="provider",
                provider=context.provider_name,
                request_id=context.request_id,
                transport=context.transport,
                attempts_started=execution.attempts_started,
                max_attempts=execution.max_attempts,
                retryable=True,
            )
        return
    trace_event(
        stage="provider",
        event="provider.recovery.retry",
        source="provider",
        provider=context.provider_name,
        recovery_kind=context.recovery_kind or operation_kind.value,
        attempts_started=execution.attempts_started,
        max_attempts=execution.max_attempts,
        exc_type=type(error).__name__,
    )


async def _read_next[RawT](iterator: AsyncIterator[RawT]) -> RawT:
    return await anext(iterator)
