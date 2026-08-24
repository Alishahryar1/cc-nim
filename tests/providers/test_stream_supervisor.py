"""Transport-neutral contracts for the shared provider stream supervisor."""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.inference import (
    FinishReason,
    InferenceEvent,
    InferenceUsage,
    ResponseCompleted,
    ResponseStarted,
    TextBlockStarted,
    TextDelta,
)
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderExecution,
    ProviderExecutionState,
)
from free_claude_code.providers.failure_policy import (
    classify_provider_failure,
    is_retryable_stream_error,
)
from free_claude_code.providers.streaming import (
    BoundAttemptOperations,
    PublicationBuffer,
    RecoveryContext,
    RecoveryOutcome,
    StreamExecutionSupervisor,
    StreamFeed,
    StreamTraceContext,
)
from tests.providers.support import immediate_admission


class _CorrectionError(ValueError):
    pass


class _Epoch(AsyncIterator[str]):
    """Scripted raw stream with observable read and close ownership."""

    def __init__(
        self,
        *items: str,
        block_after: int | None = None,
        block_error: Exception | None = None,
    ) -> None:
        self._items = items
        self._index = 0
        self._block_after = block_after
        self._block_error = block_error
        self._terminal = False
        self.read_calls = 0
        self.closed = False
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        self.read_calls += 1
        if self._block_after is not None and self._index >= self._block_after:
            self.blocked.set()
            await self.release.wait()
            if self._block_error is not None:
                raise self._block_error
            raise StopAsyncIteration
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item

    async def aclose(self) -> None:
        self.closed = True
        self.release.set()

    @property
    def recovery_snapshot(self) -> str:
        return "content" if self._index else "empty"

    def start(self) -> StreamFeed:
        return StreamFeed((ResponseStarted("response_test", "model"),))

    def feed(self, raw: str) -> StreamFeed:
        if raw == "decode_error":
            raise TimeoutError("decoder failed after first raw item")
        if raw == "noop":
            return StreamFeed()
        if raw == "content":
            return StreamFeed(
                (
                    TextBlockStarted("item_test", "block_test"),
                    TextDelta("block_test", "visible"),
                )
            )
        if raw == "terminal":
            self._terminal = True
            return StreamFeed(
                (
                    ResponseCompleted(
                        FinishReason.END_TURN,
                        InferenceUsage(),
                    ),
                ),
                terminal=True,
            )
        raise AssertionError(f"unknown raw item: {raw}")

    def finish(self) -> StreamFeed:
        if not self._terminal:
            raise TimeoutError("stream ended before its terminal event")
        return StreamFeed(terminal=True)

    def failure_events(self) -> tuple[InferenceEvent, ...]:
        return ()

    def trace_completed(self) -> None:
        pass


class _Source:
    """Scripted request source that records the request revision per call."""

    def __init__(self, *actions: _Epoch | Exception) -> None:
        self._actions = list(actions)
        self.opens = 0
        self.corrected = False
        self.opened_revisions: list[str] = []
        self.classifications = 0

    @property
    def trace_context(self) -> StreamTraceContext:
        return StreamTraceContext("TEST_STREAM", "req_test")

    @property
    def failure_override(self) -> None:
        return None

    def trace_started(self, execution: ProviderExecution) -> None:
        assert execution.state is ProviderExecutionState.ACTIVE

    async def open(self) -> _Epoch:
        self.opens += 1
        self.opened_revisions.append("corrected" if self.corrected else "original")
        action = self._actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action

    def apply_correction(self, error: Exception) -> bool:
        if not isinstance(error, _CorrectionError) or self.corrected:
            return False
        self.corrected = True
        return True

    def attempt_error(self, error: Exception) -> Exception:
        return error

    def is_retryable(self, error: Exception) -> bool:
        return is_retryable_stream_error(error)

    def classify_failure(self, error: Exception) -> ExecutionFailure:
        self.classifications += 1
        return classify_provider_failure(
            error,
            provider_name="TEST_STREAM",
            read_timeout_s=1.0,
            request_id="req_test",
        )


class _ReserveFinalAttempt:
    """Minimal Chat-like policy that recovers instead of replaying partial output."""

    def __init__(self, epoch: _Epoch) -> None:
        self._epoch = epoch
        self.resolved = False

    def prefers_recovery(self, context: RecoveryContext[str]) -> bool:
        return context.has_buffered and context.attempts_remaining == 1

    async def resolve(
        self,
        context: RecoveryContext[str],
        attempts: BoundAttemptOperations,
    ) -> RecoveryOutcome | None:
        assert context.snapshot == "content"
        assert attempts.attempts_remaining == 1
        assert self._epoch.closed
        self.resolved = True
        return RecoveryOutcome(
            events=(ResponseCompleted(FinishReason.END_TURN, InferenceUsage()),),
            publish_buffer=True,
            completed=True,
        )


async def _collect(
    admission: ProviderAdmissionController,
    source: _Source,
    *,
    publication: PublicationBuffer | None = None,
    recovery: _ReserveFinalAttempt | None = None,
) -> tuple[list[InferenceEvent], ProviderExecution]:
    execution = admission.start_execution(request_id="req_test")
    supervisor = StreamExecutionSupervisor(admission)
    with patch.object(admission, "start_execution", return_value=execution):
        events = [
            event
            async for event in supervisor.stream(
                source,
                publication=publication or PublicationBuffer(),
                recovery=recovery,
            )
        ]
    return events, execution


@pytest.mark.asyncio
async def test_preaccept_open_failure_uses_admission_retry() -> None:
    admission = immediate_admission(provider_name="TEST_STREAM", max_attempts=2)
    source = _Source(TimeoutError("open failed"), _Epoch("terminal"))

    with patch("free_claude_code.providers.admission.trace_event") as trace:
        events, execution = await _collect(admission, source)

    outcomes = [
        call.kwargs["outcome"]
        for call in trace.call_args_list
        if call.kwargs.get("event") == "provider.attempt.resolved"
    ]
    assert outcomes == ["retryable_failure", "accepted"]
    assert source.opens == 2
    assert execution.attempts_started == 2
    assert execution.state is ProviderExecutionState.SUCCEEDED
    assert sum(isinstance(event, ResponseStarted) for event in events) == 1
    assert sum(isinstance(event, ResponseCompleted) for event in events) == 1


@pytest.mark.asyncio
async def test_preaccept_eof_uses_admission_retry() -> None:
    admission = immediate_admission(provider_name="TEST_STREAM", max_attempts=2)
    source = _Source(_Epoch(), _Epoch("terminal"))

    with patch("free_claude_code.providers.admission.trace_event") as trace:
        events, execution = await _collect(admission, source)

    outcomes = [
        call.kwargs["outcome"]
        for call in trace.call_args_list
        if call.kwargs.get("event") == "provider.attempt.resolved"
    ]
    assert outcomes == ["retryable_failure", "accepted"]
    assert source.opens == 2
    assert execution.attempts_started == 2
    assert execution.state is ProviderExecutionState.SUCCEEDED
    assert sum(isinstance(event, ResponseStarted) for event in events) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("first_raw", ["noop", "decode_error"])
async def test_postaccept_failure_retries_locally_while_unpublished(
    first_raw: str,
) -> None:
    admission = immediate_admission(provider_name="TEST_STREAM", max_attempts=2)
    source = _Source(_Epoch(first_raw), _Epoch("terminal"))

    with patch("free_claude_code.providers.admission.trace_event") as trace:
        events, execution = await _collect(admission, source)

    outcomes = [
        call.kwargs["outcome"]
        for call in trace.call_args_list
        if call.kwargs.get("event") == "provider.attempt.resolved"
    ]
    assert outcomes == ["accepted", "accepted"]
    assert source.opens == 2
    assert execution.attempts_started == 2
    assert execution.state is ProviderExecutionState.SUCCEEDED
    assert sum(isinstance(event, ResponseStarted) for event in events) == 1
    assert sum(isinstance(event, ResponseCompleted) for event in events) == 1


@pytest.mark.asyncio
async def test_correction_and_early_retry_keep_the_corrected_request() -> None:
    admission = immediate_admission(provider_name="TEST_STREAM", max_attempts=3)
    source = _Source(
        _CorrectionError("unsupported deterministic field"),
        _Epoch("noop"),
        _Epoch("terminal"),
    )

    events, execution = await _collect(admission, source)

    assert source.opened_revisions == ["original", "corrected", "corrected"]
    assert execution.attempts_started == 3
    assert execution.state is ProviderExecutionState.SUCCEEDED
    assert sum(isinstance(event, ResponseStarted) for event in events) == 1


@pytest.mark.asyncio
async def test_standard_stream_consumes_its_final_prepublication_attempt() -> None:
    admission = immediate_admission(provider_name="TEST_STREAM", max_attempts=2)
    source = _Source(_Epoch("content"), _Epoch("terminal"))

    events, execution = await _collect(admission, source)

    assert source.opens == 2
    assert execution.attempts_started == 2
    assert not any(isinstance(event, TextDelta) for event in events)
    assert sum(isinstance(event, ResponseStarted) for event in events) == 1


@pytest.mark.asyncio
async def test_typed_recovery_reserves_final_attempt_after_scope_close() -> None:
    admission = immediate_admission(provider_name="TEST_STREAM", max_attempts=2)
    failed_epoch = _Epoch("content")
    source = _Source(failed_epoch)
    recovery = _ReserveFinalAttempt(failed_epoch)

    events, execution = await _collect(admission, source, recovery=recovery)

    assert recovery.resolved
    assert failed_epoch.closed
    assert source.opens == 1
    assert source.classifications == 0
    assert execution.attempts_started == 1
    assert execution.state is ProviderExecutionState.SUCCEEDED
    assert sum(isinstance(event, TextDelta) for event in events) == 1
    assert sum(isinstance(event, ResponseCompleted) for event in events) == 1


@pytest.mark.asyncio
async def test_terminal_events_publish_only_after_attempt_scope_closes() -> None:
    admission = immediate_admission(provider_name="TEST_STREAM", max_attempts=1)
    epoch = _Epoch("terminal")
    source = _Source(epoch)
    execution = admission.start_execution(request_id="req_test")
    supervisor = StreamExecutionSupervisor(admission)

    with patch.object(admission, "start_execution", return_value=execution):
        stream = supervisor.stream(source, publication=PublicationBuffer())
        first = await anext(stream)
        assert isinstance(first, ResponseStarted)
        assert epoch.closed
        remaining = [event async for event in stream]

    assert any(isinstance(event, ResponseCompleted) for event in remaining)
    assert execution.state is ProviderExecutionState.SUCCEEDED


@pytest.mark.asyncio
async def test_terminal_feed_stops_raw_reads_and_closes_attempt_scope() -> None:
    admission = immediate_admission(provider_name="TEST_STREAM", max_attempts=1)
    epoch = _Epoch("terminal", block_after=1)
    source = _Source(epoch)

    events, execution = await asyncio.wait_for(
        _collect(admission, source),
        timeout=1,
    )

    assert epoch.read_calls == 1
    assert epoch.closed
    assert not epoch.blocked.is_set()
    assert sum(isinstance(event, ResponseCompleted) for event in events) == 1
    assert execution.state is ProviderExecutionState.SUCCEEDED


@pytest.mark.asyncio
async def test_active_deadline_publishes_without_restarting_blocked_read() -> None:
    admission = immediate_admission(provider_name="TEST_STREAM", max_attempts=2)
    epoch = _Epoch("content", block_after=1)
    source = _Source(epoch)
    execution = admission.start_execution(request_id="req_test")
    supervisor = StreamExecutionSupervisor(admission)

    with patch.object(admission, "start_execution", return_value=execution):
        stream = supervisor.stream(
            source,
            publication=PublicationBuffer(holdback_seconds=0),
        )
        first = await anext(stream)
        second = await anext(stream)
        await asyncio.wait_for(epoch.blocked.wait(), timeout=1)
        assert isinstance(first, ResponseStarted)
        assert isinstance(second, TextBlockStarted)
        third = await anext(stream)
        assert isinstance(third, TextDelta)
        assert epoch.read_calls == 2

        resumed = asyncio.Event()

        async def consume_next() -> InferenceEvent:
            resumed.set()
            return await anext(stream)

        pending = asyncio.create_task(consume_next())
        await resumed.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    assert epoch.closed
    assert epoch.read_calls == 2
    assert execution.state is ProviderExecutionState.ABANDONED


@pytest.mark.asyncio
async def test_failure_after_deadline_publication_is_not_replayed() -> None:
    admission = immediate_admission(provider_name="TEST_STREAM", max_attempts=2)
    epoch = _Epoch(
        "content",
        block_after=1,
        block_error=TimeoutError("failed after publication"),
    )
    source = _Source(epoch, _Epoch("terminal"))
    execution = admission.start_execution(request_id="req_test")
    supervisor = StreamExecutionSupervisor(admission)

    with patch.object(admission, "start_execution", return_value=execution):
        stream = supervisor.stream(
            source,
            publication=PublicationBuffer(holdback_seconds=0),
        )
        visible = [await anext(stream), await anext(stream), await anext(stream)]
        await asyncio.wait_for(epoch.blocked.wait(), timeout=1)
        epoch.release.set()
        with pytest.raises(ExecutionFailure):
            await anext(stream)

    assert source.opens == 1
    assert source.classifications == 1
    assert sum(isinstance(event, TextDelta) for event in visible) == 1
    assert execution.state is ProviderExecutionState.FAILED


@pytest.mark.asyncio
async def test_failure_policy_exception_still_closes_attempt_scope() -> None:
    admission = immediate_admission(provider_name="TEST_STREAM", max_attempts=1)
    epoch = _Epoch("decode_error")
    source = _Source(epoch)
    execution = admission.start_execution(request_id="req_test")
    supervisor = StreamExecutionSupervisor(admission)

    with (
        patch.object(admission, "start_execution", return_value=execution),
        patch.object(
            source,
            "is_retryable",
            side_effect=RuntimeError("failure policy failed"),
        ),
        pytest.raises(RuntimeError, match="failure policy failed"),
    ):
        _ = [
            event
            async for event in supervisor.stream(
                source,
                publication=PublicationBuffer(),
            )
        ]

    assert epoch.closed
    assert execution.state is ProviderExecutionState.FAILED
