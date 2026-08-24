"""Provider stream commit-boundary and recovery policy."""

from free_claude_code.core.inference import (
    ResponseStarted,
    TextDelta,
    inference_event_size,
)
from free_claude_code.providers.stream_recovery import (
    RecoveryController,
    RecoveryFailureAction,
)
from free_claude_code.providers.streaming import PublicationBuffer


def _event(label: str) -> TextDelta:
    return TextDelta(block_id="test-block", delta=label)


def test_early_retry_discards_uncommitted_holdback() -> None:
    controller = RecoveryController()

    assert controller.push(_event("hidden")) == []
    decision = controller.advance_failure(
        retryable=True,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=2,
    )

    assert decision.action == RecoveryFailureAction.EARLY_RETRY
    assert decision.retryable
    assert decision.has_buffered
    assert not controller.committed
    assert not controller.has_buffered
    assert controller.flush() == []


def test_early_retry_requires_remaining_execution_budget() -> None:
    controller = RecoveryController()
    assert controller.push(_event("hidden")) == []

    decision = controller.advance_failure(
        retryable=True,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=0,
    )

    assert decision.action == RecoveryFailureAction.FINAL_ERROR
    assert decision.retryable
    assert controller.has_buffered


def test_last_attempt_is_reserved_for_partial_output_recovery() -> None:
    controller = RecoveryController()
    assert controller.push(_event("partial")) == []

    decision = controller.advance_failure(
        retryable=True,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=1,
    )

    assert decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY
    assert decision.has_buffered
    assert controller.has_buffered


def test_create_failure_is_owned_by_admission_not_stream_recovery() -> None:
    decision = RecoveryController().advance_failure(
        retryable=True,
        stream_opened=False,
        generated_output=False,
        complete_tool_salvageable=False,
        attempts_remaining=1,
    )

    assert decision.action == RecoveryFailureAction.FINAL_ERROR
    assert decision.retryable


def test_statusless_transient_api_error_allows_early_retry() -> None:
    decision = RecoveryController().advance_failure(
        retryable=True,
        stream_opened=True,
        generated_output=False,
        complete_tool_salvageable=False,
        attempts_remaining=1,
    )

    assert decision.action == RecoveryFailureAction.EARLY_RETRY
    assert decision.retryable


def test_committed_output_allows_midstream_recovery() -> None:
    controller = RecoveryController()

    event = _event("committed")
    assert controller.push(event) == []
    assert controller.flush() == [event]
    decision = controller.advance_failure(
        retryable=True,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=1,
    )

    assert decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY
    assert decision.retryable
    assert decision.committed
    assert controller.flush_uncommitted(decision) == []


def test_uncommitted_complete_tool_can_be_salvaged() -> None:
    controller = RecoveryController()

    event = _event("salvageable")
    assert controller.push(event) == []
    decision = controller.advance_failure(
        retryable=True,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=True,
        attempts_remaining=0,
    )

    assert decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY
    assert not decision.committed
    assert decision.has_buffered
    assert controller.flush_uncommitted(decision) == [event]
    assert controller.committed
    assert not controller.has_buffered


def test_non_retryable_error_is_final() -> None:
    decision = RecoveryController().advance_failure(
        retryable=False,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=2,
    )

    assert decision.action == RecoveryFailureAction.FINAL_ERROR
    assert not decision.retryable


def test_holdback_records_deadline_until_supervisor_commits() -> None:
    now = [10.0]
    holdback = PublicationBuffer(holdback_seconds=0.75, now=lambda: now[0])

    started = _event("started")
    delta = _event("delta")
    completed = _event("completed")
    terminal = _event("terminal")

    assert holdback.push(started) == []
    now[0] += 0.74
    assert holdback.push(delta) == []
    assert not holdback.published

    now[0] += 0.01
    assert holdback.push(completed) == []
    assert holdback.seconds_until_deadline == 0
    assert holdback.flush() == [started, delta, completed]
    assert holdback.published
    assert holdback.push(terminal) == [terminal]


def test_synthetic_response_start_does_not_age_holdback_before_provider_output() -> (
    None
):
    now = [10.0]
    holdback = PublicationBuffer(holdback_seconds=0.75, now=lambda: now[0])
    started = ResponseStarted("response_test", "test-model")

    assert holdback.push(started) == []
    now[0] += 10.0
    assert holdback.push(_event("first-provider-output")) == []
    assert not holdback.published


def test_holdback_flushes_at_internal_buffer_cap() -> None:
    first = _event("ab")
    second = _event("cde")
    holdback = PublicationBuffer(
        max_bytes=inference_event_size(first) + inference_event_size(second),
        now=lambda: 1.0,
    )

    assert holdback.push(first) == []
    assert holdback.push(second) == [first, second]
    assert holdback.published


def test_holdback_discard_drops_uncommitted_events() -> None:
    holdback = PublicationBuffer(now=lambda: 1.0)

    assert holdback.push(_event("hidden")) == []
    holdback.discard()

    assert holdback.flush() == []
