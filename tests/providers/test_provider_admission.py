"""Provider-owned admission and coordinated recovery contracts."""

import asyncio
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from unittest.mock import patch

import httpx
import openai
import pytest

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers.admission import (
    LONG_EXHAUSTION_THRESHOLD_S,
    TERMINAL_WINDOW_CAP_S,
    UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS,
    ProviderAdmissionController,
    ProviderRetrySession,
    _retry_after_seconds,
    _retry_info_seconds,
)
from free_claude_code.providers.failure_policy import (
    ProviderRecoveryExhausted,
    retryable_upstream_status,
)
from free_claude_code.providers.stream_recovery import TruncatedProviderStreamError
from tests.providers.rate_limit_fixtures import (
    DEEPSEEK,
    DOCUMENTED_RATE_LIMIT_FIXTURES,
    GEMINI_RPM,
    GROQ_RPD,
    HELP_TYPE,
    RETRY_INFO_TYPE,
    RateLimitFixture,
    rate_limit_error,
)


def _controller(
    *,
    provider_name: str = "TEST",
    rate_limit: int = 1_000_000,
    rate_window: float = 1.0,
    max_concurrency: int = 1_000,
    max_attempts: int = UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS,
    base_delay: float = 0.0,
    max_delay: float = 0.0,
    quota_threshold: float = LONG_EXHAUSTION_THRESHOLD_S,
) -> ProviderAdmissionController:
    return ProviderAdmissionController(
        provider_name=provider_name,
        rate_limit=rate_limit,
        rate_window=rate_window,
        max_concurrency=max_concurrency,
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        jitter=0.0,
        quota_threshold=quota_threshold,
    )


def _status_error(
    status: int,
    *,
    retry_after: str | None = None,
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://provider.test/chat/completions")
    headers = {"retry-after": retry_after} if retry_after is not None else None
    response = httpx.Response(status, request=request, headers=headers)
    return httpx.HTTPStatusError(
        f"upstream returned {status}",
        request=request,
        response=response,
    )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: ProviderAdmissionController(provider_name="TEST", rate_limit=0),
            "rate_limit",
        ),
        (
            lambda: ProviderAdmissionController(provider_name="TEST", rate_window=0),
            "rate_window",
        ),
        (
            lambda: ProviderAdmissionController(
                provider_name="TEST", max_concurrency=0
            ),
            "max_concurrency",
        ),
        (
            lambda: ProviderAdmissionController(provider_name="TEST", max_attempts=0),
            "max_attempts",
        ),
        (
            lambda: ProviderAdmissionController(provider_name="TEST", base_delay=-1),
            "base_delay",
        ),
        (
            lambda: ProviderAdmissionController(
                provider_name="TEST", base_delay=2, max_delay=1
            ),
            "max_delay",
        ),
    ],
)
def test_admission_rejects_invalid_configuration(
    factory: Callable[[], ProviderAdmissionController], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_retry_session_exposes_one_bounded_execution_budget() -> None:
    session = ProviderRetrySession(max_attempts=2)

    assert session.max_attempts == 2
    assert session.attempts_started == 0
    assert session.attempts_remaining == 2
    assert session.can_attempt
    assert session._claim_attempt() == 1
    assert session._claim_attempt() == 2
    assert not session.can_attempt
    assert session.attempts_remaining == 0
    with pytest.raises(RuntimeError, match="exhausted"):
        session._claim_attempt()


@pytest.mark.asyncio
async def test_proactive_rate_limit_is_a_strict_rolling_window() -> None:
    controller = _controller(rate_limit=1, rate_window=0.04)

    first = await controller.open_attempt(controller.new_retry_session())
    await first.succeeded()
    await first.aclose()

    started = time.monotonic()
    second = await controller.open_attempt(controller.new_retry_session())
    elapsed = time.monotonic() - started
    await second.succeeded()
    await second.aclose()

    assert elapsed >= 0.03


@pytest.mark.asyncio
async def test_concurrency_bulkhead_limits_active_attempts() -> None:
    controller = _controller(max_concurrency=2)
    first = await controller.open_attempt(controller.new_retry_session())
    second = await controller.open_attempt(controller.new_retry_session())

    third_task = asyncio.create_task(
        controller.open_attempt(controller.new_retry_session())
    )
    await asyncio.sleep(0)
    assert not third_task.done()

    await first.succeeded()
    await first.aclose()
    third = await asyncio.wait_for(third_task, timeout=1)

    await second.succeeded()
    await second.aclose()
    await third.succeeded()
    await third.aclose()


@pytest.mark.asyncio
async def test_attempt_cancellation_releases_concurrency() -> None:
    controller = _controller(max_concurrency=1)
    entered = asyncio.Event()

    async def never_finishes() -> None:
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(controller.run_with_retry(never_finishes))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert (
        await asyncio.wait_for(
            controller.run_with_retry(lambda: asyncio.sleep(0, result="ok")),
            timeout=1,
        )
        == "ok"
    )


@pytest.mark.asyncio
async def test_run_with_retry_uses_one_five_attempt_budget() -> None:
    controller = _controller()
    attempts = 0
    error = _status_error(503)

    async def fail() -> None:
        nonlocal attempts
        attempts += 1
        raise error

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await controller.run_with_retry(fail)

    assert attempts == UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS
    assert exc_info.value is error


@pytest.mark.asyncio
async def test_run_with_retry_succeeds_without_multiplying_attempts() -> None:
    controller = _controller(max_concurrency=1)
    attempts = 0

    async def recover() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise _status_error(429)
        return "recovered"

    assert (
        await asyncio.wait_for(controller.run_with_retry(recover), timeout=1)
        == "recovered"
    )
    assert attempts == 3


@pytest.mark.asyncio
async def test_recovery_traces_keep_the_logical_request_id() -> None:
    controller = _controller(max_attempts=2)
    attempts = 0

    async def recover() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _status_error(503)
        return "recovered"

    with patch("free_claude_code.providers.admission.trace_event") as trace:
        result = await controller.run_with_retry(recover, request_id="req_trace")

    assert result == "recovered"
    recovery_rows = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event", "").startswith("provider.recovery")
        or call.kwargs.get("event") == "provider.retry.scheduled"
    ]
    assert {row["event"] for row in recovery_rows} == {
        "provider.recovery.opened",
        "provider.recovery.probe",
        "provider.recovery.closed",
        "provider.retry.scheduled",
    }
    assert all(row["request_id"] == "req_trace" for row in recovery_rows)
    assert recovery_rows[-1]["outcome"] == "success"


@pytest.mark.asyncio
async def test_direct_exhaustion_trace_keeps_the_logical_request_id() -> None:
    controller = _controller(max_attempts=1)
    error = _status_error(503)

    async def fail() -> None:
        raise error

    with (
        patch("free_claude_code.providers.admission.trace_event") as trace,
        pytest.raises(httpx.HTTPStatusError),
    ):
        await controller.run_with_retry(fail, request_id="req_terminal")

    rows = [call.kwargs for call in trace.call_args_list]
    assert {row["event"] for row in rows} == {
        "provider.recovery.opened",
        "provider.retry.exhausted",
    }
    assert all(row["request_id"] == "req_terminal" for row in rows)


@pytest.mark.asyncio
async def test_non_retryable_error_is_attempted_once() -> None:
    controller = _controller()
    attempts = 0

    async def reject() -> None:
        nonlocal attempts
        attempts += 1
        raise _status_error(400)

    with pytest.raises(httpx.HTTPStatusError):
        await controller.run_with_retry(reject)

    assert attempts == 1


@pytest.mark.asyncio
async def test_pre_response_protocol_failure_opens_coordinated_recovery() -> None:
    controller = _controller()
    session = controller.new_retry_session()
    attempt = await controller.open_attempt(session)

    assert await attempt.retry(
        TruncatedProviderStreamError("stream ended before its first chunk")
    )
    assert attempt.failure_retryable is True
    await attempt.aclose()

    probe = await controller.open_attempt(session)
    await probe.succeeded()
    await probe.aclose()


@pytest.mark.asyncio
async def test_provider_override_can_classify_retryable_semantics() -> None:
    controller = _controller()
    attempts = 0

    async def degraded() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _status_error(400)
        return "healthy"

    def classify(error: Exception) -> ExecutionFailure | None:
        del error
        return ExecutionFailure(
            kind=FailureKind.OVERLOADED,
            status_code=529,
            message="temporarily degraded",
            retryable=True,
        )

    assert (
        await controller.run_with_retry(
            degraded,
            provider_failure_override=classify,
        )
        == "healthy"
    )
    assert attempts == 2


@pytest.mark.asyncio
async def test_one_leader_backs_off_while_followers_coalesce() -> None:
    controller = _controller(base_delay=2.0, max_delay=60.0)
    leader_session = controller.new_retry_session()
    follower_session = controller.new_retry_session()
    leader = await controller.open_attempt(leader_session)
    follower = await controller.open_attempt(follower_session)
    error = _status_error(429)

    assert await leader.retry(error)
    assert await follower.retry(error)
    await leader.aclose()
    await follower.aclose()

    real_sleep = asyncio.sleep
    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()
    delays: list[float] = []

    async def controlled_sleep(delay: float) -> None:
        delays.append(delay)
        sleep_started.set()
        await release_sleep.wait()

    with patch(
        "free_claude_code.providers.admission.asyncio.sleep",
        side_effect=controlled_sleep,
    ):
        leader_probe_task = asyncio.create_task(controller.open_attempt(leader_session))
        await sleep_started.wait()
        follower_attempt_task = asyncio.create_task(
            controller.open_attempt(follower_session)
        )
        await real_sleep(0)

        assert len(delays) == 1
        assert 1.9 <= delays[0] <= 2.0
        assert not follower_attempt_task.done()

        release_sleep.set()
        leader_probe = await asyncio.wait_for(leader_probe_task, timeout=1)
        await real_sleep(0)
        assert not follower_attempt_task.done()

        await leader_probe.succeeded()
        await leader_probe.aclose()
        resumed_follower = await asyncio.wait_for(follower_attempt_task, timeout=1)

    await resumed_follower.succeeded()
    await resumed_follower.aclose()


@pytest.mark.asyncio
async def test_low_quota_threshold_does_not_truncate_the_recovery_backoff() -> None:
    # Regression: a per-provider quota threshold set below the backoff delay
    # must not clamp the leader's recovery sleep. ready_at is already bounded by
    # max_delay, and a Retry-After longer than the threshold opens a terminal
    # episode instead, so the leader must sleep the full backoff curve.
    controller = _controller(base_delay=10.0, max_delay=60.0, quota_threshold=2.0)
    session = controller.new_retry_session()
    attempt = await controller.open_attempt(session)

    assert await attempt.retry(_status_error(503))
    await attempt.aclose()

    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()
    delays: list[float] = []

    async def controlled_sleep(delay: float) -> None:
        delays.append(delay)
        sleep_started.set()
        await release_sleep.wait()

    with patch(
        "free_claude_code.providers.admission.asyncio.sleep",
        side_effect=controlled_sleep,
    ):
        probe_task = asyncio.create_task(controller.open_attempt(session))
        await sleep_started.wait()

        # Full backoff (~base_delay), not clamped to the 2.0s quota threshold.
        assert len(delays) == 1
        assert delays[0] > 2.0
        assert 9.9 <= delays[0] <= 10.0

        release_sleep.set()
        probe = await asyncio.wait_for(probe_task, timeout=1)

    await probe.succeeded()
    await probe.aclose()


@pytest.mark.asyncio
async def test_cancelled_follower_leaves_recovery_episode() -> None:
    controller = _controller(base_delay=1.0, max_delay=1.0)
    leader_session = controller.new_retry_session()
    follower_session = controller.new_retry_session()
    leader = await controller.open_attempt(leader_session)
    follower = await controller.open_attempt(follower_session)

    assert await leader.retry(_status_error(503))
    assert await follower.retry(_status_error(503))
    await leader.aclose()
    await follower.aclose()

    follower_wait = asyncio.create_task(controller.open_attempt(follower_session))
    await asyncio.sleep(0)
    episode = controller._episode
    assert episode is not None
    assert follower_session in episode.waiters

    follower_wait.cancel()
    with pytest.raises(asyncio.CancelledError):
        await follower_wait

    assert follower_session not in episode.waiters


@pytest.mark.asyncio
async def test_cancelled_backoff_leader_transfers_to_a_waiter() -> None:
    controller = _controller(base_delay=2.0, max_delay=60.0)
    leader_session = controller.new_retry_session()
    follower_session = controller.new_retry_session()
    leader = await controller.open_attempt(leader_session)
    follower = await controller.open_attempt(follower_session)

    assert await leader.retry(_status_error(503))
    assert await follower.retry(_status_error(503))
    await leader.aclose()
    await follower.aclose()

    real_sleep = asyncio.sleep
    first_sleep_started = asyncio.Event()
    second_sleep_started = asyncio.Event()
    release_second_sleep = asyncio.Event()
    sleep_calls = 0

    async def controlled_sleep(delay: float) -> None:
        nonlocal sleep_calls
        assert 1.9 <= delay <= 2.0
        sleep_calls += 1
        if sleep_calls == 1:
            first_sleep_started.set()
            await asyncio.Event().wait()
        second_sleep_started.set()
        await release_second_sleep.wait()

    with patch(
        "free_claude_code.providers.admission.asyncio.sleep",
        side_effect=controlled_sleep,
    ):
        leader_task = asyncio.create_task(controller.open_attempt(leader_session))
        await first_sleep_started.wait()
        follower_task = asyncio.create_task(controller.open_attempt(follower_session))
        await real_sleep(0)

        leader_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader_task

        await second_sleep_started.wait()
        release_second_sleep.set()
        probe = await asyncio.wait_for(follower_task, timeout=1)

    await probe.succeeded()
    await probe.aclose()


@pytest.mark.asyncio
async def test_stale_in_flight_success_cannot_close_recovery_episode() -> None:
    controller = _controller()
    leader_session = controller.new_retry_session()
    stale_session = controller.new_retry_session()
    leader = await controller.open_attempt(leader_session)
    stale = await controller.open_attempt(stale_session)

    assert await leader.retry(_status_error(503))
    await leader.aclose()
    await stale.succeeded()
    await stale.aclose()

    waiting = asyncio.create_task(
        controller.open_attempt(controller.new_retry_session())
    )
    await asyncio.sleep(0)
    assert not waiting.done()

    probe = await controller.open_attempt(leader_session)
    await probe.succeeded()
    await probe.aclose()
    resumed = await asyncio.wait_for(waiting, timeout=1)
    await resumed.succeeded()
    await resumed.aclose()


@pytest.mark.asyncio
async def test_abandoned_probe_transfers_leadership() -> None:
    controller = _controller()
    leader_session = controller.new_retry_session()
    follower_session = controller.new_retry_session()
    leader = await controller.open_attempt(leader_session)
    follower = await controller.open_attempt(follower_session)

    assert await leader.retry(_status_error(503))
    assert await follower.retry(_status_error(503))
    await leader.aclose()
    await follower.aclose()

    abandoned_probe = await controller.open_attempt(leader_session)
    follower_probe_task = asyncio.create_task(controller.open_attempt(follower_session))
    await asyncio.sleep(0)
    assert not follower_probe_task.done()

    await abandoned_probe.aclose()
    follower_probe = await asyncio.wait_for(follower_probe_task, timeout=1)
    await follower_probe.succeeded()
    await follower_probe.aclose()


@pytest.mark.asyncio
async def test_cancelled_probe_resolution_remains_abandonable() -> None:
    controller = _controller(max_concurrency=1)
    leader_session = controller.new_retry_session()
    follower_session = controller.new_retry_session()
    leader = await controller.open_attempt(leader_session)

    assert await leader.retry(_status_error(503))
    await leader.aclose()

    probe = await controller.open_attempt(leader_session)
    await controller._condition.acquire()
    resolution = asyncio.create_task(probe.succeeded())
    await asyncio.sleep(0)
    resolution.cancel()
    controller._condition.release()
    with pytest.raises(asyncio.CancelledError):
        await resolution

    follower = asyncio.create_task(controller.open_attempt(follower_session))
    await probe.aclose()
    replacement = await asyncio.wait_for(follower, timeout=1)
    await replacement.succeeded()
    await replacement.aclose()


@pytest.mark.asyncio
async def test_cancelled_probe_close_releases_slot_and_transfers_leadership() -> None:
    controller = _controller(max_concurrency=1)
    leader_session = controller.new_retry_session()
    follower_session = controller.new_retry_session()
    leader = await controller.open_attempt(leader_session)

    assert await leader.retry(_status_error(503))
    await leader.aclose()

    probe = await controller.open_attempt(leader_session)
    await controller._condition.acquire()
    close = asyncio.create_task(probe.aclose())
    await asyncio.sleep(0)
    close.cancel()
    controller._condition.release()
    with pytest.raises(asyncio.CancelledError):
        await close

    replacement = await asyncio.wait_for(
        controller.open_attempt(follower_session),
        timeout=1,
    )
    await replacement.succeeded()
    await replacement.aclose()


@pytest.mark.asyncio
async def test_non_retryable_probe_closes_episode_and_only_rejects_leader() -> None:
    controller = _controller()
    leader_session = controller.new_retry_session()
    follower_session = controller.new_retry_session()
    leader = await controller.open_attempt(leader_session)
    follower = await controller.open_attempt(follower_session)

    assert await leader.retry(_status_error(503))
    assert await follower.retry(_status_error(503))
    await leader.aclose()
    await follower.aclose()

    probe = await controller.open_attempt(leader_session)
    follower_task = asyncio.create_task(controller.open_attempt(follower_session))
    await asyncio.sleep(0)
    assert not follower_task.done()

    assert not await probe.retry(_status_error(400))
    assert probe.failure_retryable is False
    await probe.aclose()
    resumed = await asyncio.wait_for(follower_task, timeout=1)
    await resumed.succeeded()
    await resumed.aclose()


@pytest.mark.asyncio
async def test_probe_exhaustion_fails_waiters_and_opens_after_cooldown() -> None:
    controller = _controller(
        max_attempts=2,
        base_delay=0.02,
        max_delay=0.02,
    )
    leader_session = controller.new_retry_session()
    follower_session = controller.new_retry_session()
    leader = await controller.open_attempt(leader_session)
    follower = await controller.open_attempt(follower_session)
    error = _status_error(429)

    assert await leader.retry(error)
    assert await follower.retry(error)
    await leader.aclose()
    await follower.aclose()

    follower_wait = asyncio.create_task(controller.open_attempt(follower_session))
    probe = await controller.open_attempt(leader_session)
    assert not await probe.retry(error)
    await probe.aclose()

    with pytest.raises(ProviderRecoveryExhausted) as waiter_error:
        await follower_wait
    assert waiter_error.value.last_error is error

    with pytest.raises(ProviderRecoveryExhausted):
        await controller.open_attempt(controller.new_retry_session())

    await asyncio.sleep(0.03)
    recovery = await controller.open_attempt(controller.new_retry_session())
    await recovery.succeeded()
    await recovery.aclose()


@pytest.mark.asyncio
async def test_exhausted_waiter_cannot_join_a_later_recovery_generation() -> None:
    controller = _controller(
        max_attempts=2,
        base_delay=0.01,
        max_delay=0.01,
    )
    leader_session = controller.new_retry_session()
    delayed_waiter_session = controller.new_retry_session()
    leader = await controller.open_attempt(leader_session)
    delayed_waiter = await controller.open_attempt(delayed_waiter_session)
    error = _status_error(503)

    assert await leader.retry(error)
    assert await delayed_waiter.retry(error)
    await leader.aclose()
    await delayed_waiter.aclose()

    probe = await controller.open_attempt(leader_session)
    assert not await probe.retry(error)
    await probe.aclose()

    await asyncio.sleep(0.02)
    later = await controller.open_attempt(controller.new_retry_session())
    await later.succeeded()
    await later.aclose()

    with pytest.raises(ProviderRecoveryExhausted) as exc_info:
        await controller.open_attempt(delayed_waiter_session)
    assert exc_info.value.last_error is error


@pytest.mark.asyncio
async def test_late_in_flight_failure_keeps_exhausted_generation_outcome() -> None:
    controller = _controller(
        max_attempts=2,
        base_delay=0.01,
        max_delay=0.01,
    )
    leader_session = controller.new_retry_session()
    stale_session = controller.new_retry_session()
    leader = await controller.open_attempt(leader_session)
    stale = await controller.open_attempt(stale_session)
    error = _status_error(503)

    assert await leader.retry(error)
    await leader.aclose()
    probe = await controller.open_attempt(leader_session)
    assert not await probe.retry(error)
    await probe.aclose()

    assert await stale.retry(_status_error(503))
    await stale.aclose()
    await asyncio.sleep(0.02)

    with pytest.raises(ProviderRecoveryExhausted) as exc_info:
        await controller.open_attempt(stale_session)
    assert exc_info.value.last_error is error


@pytest.mark.asyncio
async def test_retry_after_is_a_minimum_backoff() -> None:
    controller = _controller(max_attempts=2)
    attempts = 0

    async def recover() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _status_error(429, retry_after="7")
        return "ok"

    with patch(
        "free_claude_code.providers.admission.asyncio.sleep",
        return_value=None,
    ) as sleep:
        assert await controller.run_with_retry(recover) == "ok"

    sleep.assert_awaited_once()
    await_args = sleep.await_args
    assert await_args is not None
    assert 6.9 <= await_args.args[0] <= 7.0


@pytest.mark.parametrize(
    "fixture",
    DOCUMENTED_RATE_LIMIT_FIXTURES,
    ids=[fixture.name for fixture in DOCUMENTED_RATE_LIMIT_FIXTURES],
)
def test_documented_rate_limit_shapes_expose_their_retry_hint(
    fixture: RateLimitFixture,
) -> None:
    error = rate_limit_error(fixture)

    assert retryable_upstream_status(error) == 429
    assert _retry_after_seconds(error) == fixture.retry_after_s


def _retry_info_body(delay: object) -> dict[str, object]:
    return {"error": {"details": [{"@type": RETRY_INFO_TYPE, "retryDelay": delay}]}}


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (_retry_info_body("12s"), 12.0),
        ([_retry_info_body("9s")], 9.0),
        (json.dumps(_retry_info_body("1.5s")), 1.5),
        (json.dumps(_retry_info_body("8s")).encode(), 8.0),
        ({"details": [{"@type": RETRY_INFO_TYPE, "retryDelay": "3s"}]}, 3.0),
        (_retry_info_body("-4s"), 0.0),
        (_retry_info_body("infs"), None),
        (_retry_info_body("nans"), None),
        (_retry_info_body("12"), None),
        (_retry_info_body("s"), None),
        (_retry_info_body({"seconds": 12}), None),
        ({"error": {"details": [{"@type": HELP_TYPE, "links": []}]}}, None),
        ({"error": {"details": "not-a-list"}}, None),
        ({"error": "quota exhausted"}, None),
        ("not json", None),
        (None, None),
    ],
)
def test_retry_info_body_hints_follow_the_protobuf_duration_format(
    body: object, expected: float | None
) -> None:
    assert _retry_info_seconds(body) == expected


def test_retry_after_header_wins_over_a_body_retry_hint() -> None:
    request = httpx.Request("POST", "https://provider.test/v1/chat/completions")
    body = {"error": {"details": [{"@type": RETRY_INFO_TYPE, "retryDelay": "300s"}]}}
    error = openai.RateLimitError(
        "rate limited",
        response=httpx.Response(
            429,
            request=request,
            headers={"retry-after": "5"},
            json=body,
        ),
        body=body,
    )

    assert _retry_after_seconds(error) == 5.0


@pytest.mark.asyncio
async def test_retry_traces_record_a_body_sourced_retry_after() -> None:
    controller = _controller(max_attempts=2)
    attempts = 0

    async def recover() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise rate_limit_error(GEMINI_RPM)
        return "ok"

    with (
        patch("free_claude_code.providers.admission.asyncio.sleep", return_value=None),
        patch("free_claude_code.providers.admission.trace_event") as trace,
    ):
        assert await controller.run_with_retry(recover) == "ok"

    scheduled = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event") == "provider.retry.scheduled"
    ]
    assert [row["retry_after_s"] for row in scheduled] == [GEMINI_RPM.retry_after_s]


@pytest.mark.asyncio
async def test_exhausted_trace_records_a_missing_retry_after() -> None:
    controller = _controller(max_attempts=1)

    async def fail() -> None:
        raise rate_limit_error(DEEPSEEK)

    with (
        patch("free_claude_code.providers.admission.trace_event") as trace,
        pytest.raises(openai.RateLimitError),
    ):
        await controller.run_with_retry(fail)

    exhausted = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event") == "provider.retry.exhausted"
    ]
    assert [row["retry_after_s"] for row in exhausted] == [None]


def test_retry_after_accepts_http_date_and_rejects_invalid_values() -> None:
    future = format_datetime(datetime.now(UTC) + timedelta(seconds=10), usegmt=True)

    parsed = _retry_after_seconds(_status_error(429, retry_after=future))

    assert parsed is not None
    assert 8 <= parsed <= 10
    assert _retry_after_seconds(_status_error(429, retry_after="invalid")) is None
    assert _retry_after_seconds(_status_error(429, retry_after="nan")) is None
    assert _retry_after_seconds(_status_error(429, retry_after="inf")) is None


@pytest.mark.asyncio
async def test_provider_controllers_do_not_share_recovery_state() -> None:
    first = _controller(provider_name="FIRST")
    second = _controller(provider_name="SECOND")
    first_session = first.new_retry_session()
    first_attempt = await first.open_attempt(first_session)

    assert await first_attempt.retry(_status_error(503))
    await first_attempt.aclose()

    independent = await asyncio.wait_for(
        second.open_attempt(second.new_retry_session()),
        timeout=1,
    )
    await independent.succeeded()
    await independent.aclose()

    probe = await first.open_attempt(first_session)
    await probe.succeeded()
    await probe.aclose()


@pytest.mark.parametrize(
    ("retry_after", "terminal"),
    [
        (f"{LONG_EXHAUSTION_THRESHOLD_S - 1}", False),
        (f"{LONG_EXHAUSTION_THRESHOLD_S}", False),
        (f"{LONG_EXHAUSTION_THRESHOLD_S + 0.001}", True),
        (f"{LONG_EXHAUSTION_THRESHOLD_S * 2}", True),
    ],
)
@pytest.mark.asyncio
async def test_only_retry_after_past_the_threshold_opens_a_terminal_episode(
    retry_after: str, terminal: bool
) -> None:
    controller = _controller()
    attempt = await controller.open_attempt(controller.new_retry_session())

    assert (
        await attempt.retry(_status_error(429, retry_after=retry_after)) is not terminal
    )
    await attempt.aclose()

    episode = controller._episode
    assert episode is not None
    assert (episode.terminal_until is not None) is terminal


@pytest.mark.asyncio
async def test_long_retry_after_fails_the_session_and_every_later_arrival() -> None:
    controller = _controller(base_delay=2.0, max_delay=60.0)
    session = controller.new_retry_session()
    attempt = await controller.open_attempt(session)
    error = _status_error(429, retry_after="3600")

    assert not await attempt.retry(error)
    await attempt.aclose()

    with pytest.raises(ProviderRecoveryExhausted) as reused:
        await controller.open_attempt(session)
    with pytest.raises(ProviderRecoveryExhausted) as arrival:
        await controller.open_attempt(controller.new_retry_session())

    assert reused.value.last_error is error
    assert arrival.value.last_error is error
    episode = controller._episode
    assert episode is not None
    assert episode.leader is None
    assert episode.terminal_until is not None
    assert 3590 <= episode.terminal_until - time.monotonic() <= 3600


@pytest.mark.asyncio
async def test_long_retry_after_fails_parked_waiters_immediately() -> None:
    controller = _controller()
    leader_session = controller.new_retry_session()
    waiter_session = controller.new_retry_session()
    leader = await controller.open_attempt(leader_session)
    waiter = await controller.open_attempt(waiter_session)

    assert await leader.retry(_status_error(503))
    assert await waiter.retry(_status_error(503))
    await leader.aclose()
    await waiter.aclose()

    parked = asyncio.create_task(controller.open_attempt(waiter_session))
    await asyncio.sleep(0)
    episode = controller._episode
    assert episode is not None
    assert waiter_session in episode.waiters

    probe = await controller.open_attempt(leader_session)
    quota_error = _status_error(429, retry_after="3600")
    assert not await probe.retry(quota_error)
    await probe.aclose()

    with pytest.raises(ProviderRecoveryExhausted) as exc_info:
        await asyncio.wait_for(parked, timeout=1)

    assert exc_info.value.last_error is quota_error
    assert not episode.waiters


@pytest.mark.asyncio
async def test_terminal_window_is_capped_at_one_day() -> None:
    controller = _controller()
    attempt = await controller.open_attempt(controller.new_retry_session())

    assert not await attempt.retry(_status_error(429, retry_after="172800"))
    await attempt.aclose()

    episode = controller._episode
    assert episode is not None
    assert episode.terminal_until is not None
    remaining = episode.terminal_until - time.monotonic()
    assert TERMINAL_WINDOW_CAP_S - 1 <= remaining <= TERMINAL_WINDOW_CAP_S


@pytest.mark.asyncio
async def test_terminal_window_reopens_half_open_after_it_expires() -> None:
    controller = _controller(quota_threshold=0.01)
    attempt = await controller.open_attempt(controller.new_retry_session())

    assert not await attempt.retry(_status_error(429, retry_after="0.05"))
    await attempt.aclose()

    with pytest.raises(ProviderRecoveryExhausted):
        await controller.open_attempt(controller.new_retry_session())

    await asyncio.sleep(0.06)
    recovered = await asyncio.wait_for(
        controller.open_attempt(controller.new_retry_session()),
        timeout=1,
    )
    await recovered.succeeded()
    await recovered.aclose()

    assert controller._episode is None


@pytest.mark.asyncio
async def test_long_exhaustion_trace_reports_the_window_it_opened() -> None:
    controller = _controller()
    attempt = await controller.open_attempt(
        controller.new_retry_session(request_id="req_quota")
    )

    with patch("free_claude_code.providers.admission.trace_event") as trace:
        assert not await attempt.retry(_status_error(429, retry_after="172800"))
    await attempt.aclose()

    rows = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event") == "provider.recovery.exhausted_long"
    ]
    assert len(rows) == 1
    assert rows[0]["request_id"] == "req_quota"
    assert rows[0]["status_code"] == 429
    assert rows[0]["retry_after_s"] == 172800.0
    assert rows[0]["terminal_window_s"] == TERMINAL_WINDOW_CAP_S


@pytest.mark.asyncio
async def test_quota_exhaustion_fails_fast_under_production_backoff() -> None:
    controller = ProviderAdmissionController(provider_name="TEST")
    upstream_calls = 0

    async def fail() -> None:
        nonlocal upstream_calls
        upstream_calls += 1
        raise rate_limit_error(GROQ_RPD)

    started = time.monotonic()
    with pytest.raises(openai.RateLimitError):
        await controller.run_with_retry(fail)
    first_elapsed = time.monotonic() - started

    started = time.monotonic()
    with pytest.raises(ProviderRecoveryExhausted):
        await controller.run_with_retry(fail)
    second_elapsed = time.monotonic() - started

    assert upstream_calls == 1
    assert first_elapsed < 0.5
    assert second_elapsed < 0.5


@pytest.mark.asyncio
async def test_recovery_status_reports_a_healthy_provider() -> None:
    controller = _controller(provider_name="HEALTHY")

    status = controller.recovery_status()

    assert status.provider_name == "HEALTHY"
    assert status.state == "healthy"
    assert status.generation is None
    assert status.seconds_remaining is None
    assert status.last_error_type is None
    assert status.waiters == 0
    assert status.quota_threshold_s == LONG_EXHAUSTION_THRESHOLD_S


@pytest.mark.asyncio
async def test_recovery_status_reports_an_open_probe_episode() -> None:
    controller = _controller(base_delay=2.0, max_delay=60.0)
    leader_session = controller.new_retry_session()
    waiter_session = controller.new_retry_session()
    leader = await controller.open_attempt(leader_session)
    waiter = await controller.open_attempt(waiter_session)

    assert await leader.retry(_status_error(503))
    assert await waiter.retry(_status_error(503))
    await leader.aclose()
    await waiter.aclose()

    status = controller.recovery_status()

    assert status.state == "recovering"
    assert status.generation == 1
    assert status.last_error_type == "HTTPStatusError"
    assert status.waiters == 1
    assert status.seconds_remaining is not None
    assert 0 < status.seconds_remaining <= 2.0


@pytest.mark.asyncio
async def test_recovery_status_reports_a_quota_exhausted_window() -> None:
    controller = _controller()
    attempt = await controller.open_attempt(controller.new_retry_session())

    assert not await attempt.retry(_status_error(429, retry_after="3600"))
    await attempt.aclose()

    status = controller.recovery_status()

    assert status.state == "exhausted"
    assert status.last_error_type == "HTTPStatusError"
    assert status.seconds_remaining is not None
    assert 3590 <= status.seconds_remaining <= 3600


@pytest.mark.asyncio
async def test_recovery_status_returns_to_recovering_once_the_window_expires() -> None:
    controller = _controller(quota_threshold=0.01)
    attempt = await controller.open_attempt(controller.new_retry_session())

    assert not await attempt.retry(_status_error(429, retry_after="0.05"))
    await attempt.aclose()
    await asyncio.sleep(0.06)

    assert controller.recovery_status().state == "recovering"


@pytest.mark.parametrize(
    ("quota_threshold", "retry_after", "terminal"),
    [
        (10.0, "9", False),
        (10.0, "11", True),
        (600.0, "300", False),
        (600.0, "900", True),
    ],
)
@pytest.mark.asyncio
async def test_quota_threshold_override_moves_the_fail_fast_boundary(
    quota_threshold: float, retry_after: str, terminal: bool
) -> None:
    controller = _controller(quota_threshold=quota_threshold)
    attempt = await controller.open_attempt(controller.new_retry_session())

    assert (
        await attempt.retry(_status_error(429, retry_after=retry_after)) is not terminal
    )
    await attempt.aclose()

    episode = controller._episode
    assert episode is not None
    assert (episode.terminal_until is not None) is terminal
    assert controller.recovery_status().quota_threshold_s == quota_threshold


def test_admission_rejects_a_non_positive_quota_threshold() -> None:
    with pytest.raises(ValueError, match="quota_threshold"):
        ProviderAdmissionController(provider_name="TEST", quota_threshold=0)
