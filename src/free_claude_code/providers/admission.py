"""Provider-owned admission, concurrency, and coordinated retry lifecycle."""

import asyncio
import json
import math
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar

from loguru import logger

from free_claude_code.core.rate_limit import StrictSlidingWindowLimiter
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.failure_policy import (
    ProviderFailureOverride,
    ProviderRecoveryExhausted,
    is_retryable_provider_error,
    retryable_upstream_status,
)

T = TypeVar("T")

UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS = 5
DEFAULT_UPSTREAM_BASE_DELAY = 2.0
DEFAULT_UPSTREAM_MAX_DELAY = 60.0
DEFAULT_UPSTREAM_JITTER = 1.0

# A retry hint longer than one full backoff ceiling reports quota exhaustion
# rather than momentary pressure: probing it would strand every waiter for the
# whole window, so the episode opens terminal instead and callers fail fast.
LONG_EXHAUSTION_THRESHOLD_S = DEFAULT_UPSTREAM_MAX_DELAY
TERMINAL_WINDOW_CAP_S = 24 * 3600.0

_RETRY_INFO_TYPE = "type.googleapis.com/google.rpc.RetryInfo"


class ProviderRetrySession:
    """One non-multiplying upstream-attempt budget for a logical execution."""

    def __init__(self, *, max_attempts: int, request_id: str | None = None) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be > 0")
        self._max_attempts = max_attempts
        self._request_id = request_id
        self._attempts_started = 0
        self._terminal_error: Exception | None = None

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    @property
    def request_id(self) -> str | None:
        return self._request_id

    @property
    def attempts_started(self) -> int:
        return self._attempts_started

    @property
    def can_attempt(self) -> bool:
        return self._attempts_started < self._max_attempts

    @property
    def attempts_remaining(self) -> int:
        return self._max_attempts - self._attempts_started

    def _claim_attempt(self) -> int:
        if not self.can_attempt:
            raise RuntimeError("provider retry session is exhausted")
        self._attempts_started += 1
        return self._attempts_started

    def _fail_recovery(self, error: Exception) -> None:
        self._terminal_error = error

    def _terminal_failure(self) -> Exception | None:
        return self._terminal_error


@dataclass(frozen=True, slots=True)
class ProviderRecoveryStatus:
    """A point-in-time view of one provider's shared recovery state.

    ``state`` is ``healthy`` with no open episode, ``exhausted`` while a
    quota window fails every caller fast, and ``recovering`` while one probe
    owns the half-open retry.
    """

    provider_name: str
    state: str
    generation: int | None = None
    seconds_remaining: float | None = None
    last_error_type: str | None = None
    waiters: int = 0
    quota_threshold_s: float = LONG_EXHAUSTION_THRESHOLD_S


@dataclass(frozen=True, slots=True)
class _GatePermit:
    generation: int | None
    probe: bool


@dataclass(slots=True)
class _RecoveryEpisode:
    generation: int
    leader: ProviderRetrySession | None
    ready_at: float
    last_error: Exception
    waiters: set[ProviderRetrySession] = field(default_factory=set)
    probe_active: bool = False
    terminal_until: float | None = None


class ProviderAttempt:
    """One admitted upstream attempt and its held concurrency slot."""

    def __init__(
        self,
        controller: ProviderAdmissionController,
        session: ProviderRetrySession,
        permit: _GatePermit,
    ) -> None:
        self._controller = controller
        self._session = session
        self._permit = permit
        self._resolved = False
        self._accepted = False
        self._failure_retryable: bool | None = None
        self._closed = False

    @property
    def accepted(self) -> bool:
        """Return whether upstream acceptance has resolved this attempt."""
        return self._accepted

    @property
    def failure_retryable(self) -> bool | None:
        """Return the provider classification recorded for a failed attempt."""
        return self._failure_retryable

    async def succeeded(self) -> None:
        """Record a valid upstream response and close a recovery episode if probing."""
        if self._resolved:
            return
        await self._controller._attempt_succeeded(self._session, self._permit)
        self._resolved = True
        self._accepted = True

    async def retry_immediately(self) -> None:
        """Keep probe ownership for a bounded request-shape correction."""
        if self._resolved:
            return
        await self._controller._attempt_corrected(self._session, self._permit)
        self._resolved = True

    async def retry(
        self,
        error: Exception,
        *,
        provider_failure_override: ProviderFailureOverride | None = None,
    ) -> bool:
        """Record a retryable failure and return whether this execution may retry."""
        if self._resolved:
            return False
        effective_error = (
            provider_failure_override(error)
            if provider_failure_override is not None
            else None
        )
        if effective_error is None:
            effective_error = error
        status = retryable_upstream_status(effective_error)
        retryable = is_retryable_provider_error(effective_error)
        self._failure_retryable = retryable
        if not retryable:
            await self._controller._attempt_rejected(self._session, self._permit)
            self._resolved = True
            return False
        should_retry = await self._controller._attempt_failed(
            self._session,
            self._permit,
            error=error,
            status=status,
        )
        self._resolved = True
        return should_retry

    async def aclose(self) -> None:
        """Release attempt ownership and its concurrency slot exactly once."""
        if self._closed:
            return
        self._closed = True
        try:
            if not self._resolved:
                await asyncio.shield(
                    self._controller._attempt_abandoned(self._session, self._permit)
                )
        finally:
            self._controller._release_concurrency()


class ProviderAdmissionController:
    """Coordinate one provider's rate, concurrency, and recovery state.

    Normal attempts pass through a strict sliding window and concurrency bulkhead.
    The first shared transient failure opens one recovery episode. Exactly one
    logical execution owns its half-open probes; concurrent callers wait for that
    episode instead of starting independent retry loops.
    """

    def __init__(
        self,
        *,
        provider_name: str,
        rate_limit: int = 40,
        rate_window: float = 60.0,
        max_concurrency: int = 5,
        max_attempts: int = UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS,
        base_delay: float = DEFAULT_UPSTREAM_BASE_DELAY,
        max_delay: float = DEFAULT_UPSTREAM_MAX_DELAY,
        jitter: float = DEFAULT_UPSTREAM_JITTER,
        quota_threshold: float = LONG_EXHAUSTION_THRESHOLD_S,
    ) -> None:
        if rate_limit <= 0:
            raise ValueError("rate_limit must be > 0")
        if rate_window <= 0:
            raise ValueError("rate_window must be > 0")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be > 0")
        if base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        if max_delay < base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if jitter < 0:
            raise ValueError("jitter must be >= 0")
        if quota_threshold <= 0:
            raise ValueError("quota_threshold must be > 0")

        self._provider_name = provider_name
        self._quota_threshold = quota_threshold
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._jitter = jitter
        self._proactive_limiter = StrictSlidingWindowLimiter(
            rate_limit, float(rate_window)
        )
        self._concurrency_sem = asyncio.Semaphore(max_concurrency)
        self._condition = asyncio.Condition()
        self._episode: _RecoveryEpisode | None = None
        self._next_generation = 1
        logger.info(
            "Provider admission initialized for {} ({} req / {}s, "
            "max_concurrency={}, max_attempts={})",
            provider_name,
            rate_limit,
            rate_window,
            max_concurrency,
            max_attempts,
        )

    def recovery_status(self) -> ProviderRecoveryStatus:
        """Return a snapshot of the shared recovery state for operators.

        Deliberately synchronous: without an await the episode fields cannot
        change underneath the read, so no condition lock is needed.
        """
        episode = self._episode
        if episode is None:
            return ProviderRecoveryStatus(
                provider_name=self._provider_name,
                state="healthy",
                quota_threshold_s=self._quota_threshold,
            )

        now = time.monotonic()
        terminal_until = episode.terminal_until
        exhausted = terminal_until is not None and now < terminal_until
        return ProviderRecoveryStatus(
            provider_name=self._provider_name,
            state="exhausted" if exhausted else "recovering",
            generation=episode.generation,
            seconds_remaining=(
                terminal_until - now
                if exhausted and terminal_until is not None
                else max(0.0, episode.ready_at - now)
            ),
            last_error_type=type(episode.last_error).__name__,
            waiters=len(episode.waiters),
            quota_threshold_s=self._quota_threshold,
        )

    def new_retry_session(
        self,
        *,
        request_id: str | None = None,
    ) -> ProviderRetrySession:
        """Return a fresh logical-execution retry budget."""
        return ProviderRetrySession(
            max_attempts=self._max_attempts,
            request_id=request_id,
        )

    async def open_attempt(self, session: ProviderRetrySession) -> ProviderAttempt:
        """Wait for provider admission and hold one active-operation slot."""
        if not session.can_attempt:
            raise RuntimeError("provider retry session is exhausted")

        while True:
            permit = await self._wait_for_gate(session)
            slot_acquired = False
            try:
                admitted = await self._proactive_limiter.acquire_if(
                    lambda permit=permit: self._permit_is_current(session, permit)
                )
                if not admitted:
                    await self._abandon_probe_permit(session, permit)
                    continue
                await self._concurrency_sem.acquire()
                slot_acquired = True
                if not self._permit_is_current(session, permit):
                    self._concurrency_sem.release()
                    slot_acquired = False
                    await self._abandon_probe_permit(session, permit)
                    continue
                session._claim_attempt()
                return ProviderAttempt(self, session, permit)
            except BaseException:
                if slot_acquired:
                    self._concurrency_sem.release()
                await self._abandon_probe_permit(session, permit)
                raise

    async def run_with_retry(
        self,
        fn: Callable[[], Awaitable[T]],
        *,
        provider_failure_override: ProviderFailureOverride | None = None,
        request_id: str | None = None,
    ) -> T:
        """Run one non-streaming provider operation through coordinated retries."""
        session = self.new_retry_session(request_id=request_id)
        while True:
            attempt = await self.open_attempt(session)
            try:
                result = await fn()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                should_retry = await attempt.retry(
                    error,
                    provider_failure_override=provider_failure_override,
                )
                if not should_retry:
                    raise
            else:
                await attempt.succeeded()
                return result
            finally:
                await attempt.aclose()

    async def _wait_for_gate(self, session: ProviderRetrySession) -> _GatePermit:
        while True:
            if (terminal_error := session._terminal_failure()) is not None:
                raise ProviderRecoveryExhausted(terminal_error)
            sleep_delay: float | None = None
            claimed_generation: int | None = None
            async with self._condition:
                episode = self._episode
                if episode is None:
                    return _GatePermit(generation=None, probe=False)

                now = time.monotonic()
                if episode.terminal_until is not None:
                    if now < episode.terminal_until:
                        raise ProviderRecoveryExhausted(episode.last_error)
                    episode = self._start_recovery_episode(
                        leader=session,
                        ready_at=now,
                        last_error=episode.last_error,
                    )

                if episode.leader is None:
                    episode.leader = session
                    episode.waiters.discard(session)
                if episode.leader is session:
                    if episode.probe_active:
                        await self._condition.wait()
                        continue
                    # ready_at is already bounded: coordinated backoff never
                    # exceeds max_delay, and any Retry-After longer than the
                    # quota threshold has already opened a terminal episode
                    # above. No extra clamp is needed, and clamping to the
                    # quota threshold would wrongly truncate the backoff curve
                    # when that threshold is overridden below max_delay.
                    sleep_delay = max(0.0, episode.ready_at - now)
                    claimed_generation = episode.generation
                    if sleep_delay == 0:
                        episode.probe_active = True
                        return self._probe_permit(session, episode)
                else:
                    episode.waiters.add(session)
                    try:
                        await self._condition.wait()
                    except asyncio.CancelledError:
                        current = self._episode
                        if (
                            current is not None
                            and current.generation == episode.generation
                        ):
                            current.waiters.discard(session)
                        raise
                    continue

            if sleep_delay is None or claimed_generation is None:
                continue
            try:
                if sleep_delay > 0:
                    logger.warning(
                        "Provider {} recovery active, waiting {:.1f}s for one probe",
                        self._provider_name,
                        sleep_delay,
                    )
                    await asyncio.sleep(sleep_delay)
                async with self._condition:
                    episode = self._episode
                    if (
                        episode is not None
                        and episode.generation == claimed_generation
                        and episode.terminal_until is None
                        and episode.leader is session
                        and not episode.probe_active
                    ):
                        episode.probe_active = True
                        return self._probe_permit(session, episode)
            except asyncio.CancelledError:
                await self._abandon_waiting_leader(session, claimed_generation)
                raise

    def _probe_permit(
        self,
        session: ProviderRetrySession,
        episode: _RecoveryEpisode,
    ) -> _GatePermit:
        trace_event(
            stage="provider",
            event="provider.recovery.probe",
            source="provider",
            provider=self._provider_name,
            request_id=session.request_id,
            generation=episode.generation,
            attempt=session.attempts_started + 1,
            max_attempts=session.max_attempts,
        )
        return _GatePermit(generation=episode.generation, probe=True)

    def _permit_is_current(
        self,
        session: ProviderRetrySession,
        permit: _GatePermit,
    ) -> bool:
        episode = self._episode
        if not permit.probe:
            return episode is None
        return (
            episode is not None
            and episode.generation == permit.generation
            and episode.terminal_until is None
            and episode.leader is session
            and episode.probe_active
        )

    async def _attempt_succeeded(
        self,
        session: ProviderRetrySession,
        permit: _GatePermit,
    ) -> None:
        if not permit.probe:
            return
        async with self._condition:
            episode = self._matching_probe(session, permit)
            if episode is None:
                return
            self._episode = None
            self._condition.notify_all()
        trace_event(
            stage="provider",
            event="provider.recovery.closed",
            source="provider",
            provider=self._provider_name,
            request_id=session.request_id,
            generation=permit.generation,
            attempt=session.attempts_started,
            outcome="success",
        )

    async def _attempt_corrected(
        self,
        session: ProviderRetrySession,
        permit: _GatePermit,
    ) -> None:
        if not permit.probe:
            return
        async with self._condition:
            episode = self._matching_probe(session, permit)
            if episode is None:
                return
            episode.probe_active = False
            episode.ready_at = time.monotonic()
            self._condition.notify_all()

    async def _attempt_rejected(
        self,
        session: ProviderRetrySession,
        permit: _GatePermit,
    ) -> None:
        """Close a probe episode when upstream responds with a final rejection."""
        if not permit.probe:
            return
        async with self._condition:
            episode = self._matching_probe(session, permit)
            if episode is None:
                return
            self._episode = None
            self._condition.notify_all()
        trace_event(
            stage="provider",
            event="provider.recovery.closed",
            source="provider",
            provider=self._provider_name,
            request_id=session.request_id,
            generation=permit.generation,
            attempt=session.attempts_started,
            outcome="rejected",
        )

    async def _attempt_abandoned(
        self,
        session: ProviderRetrySession,
        permit: _GatePermit,
    ) -> None:
        if not permit.probe:
            return
        async with self._condition:
            episode = self._matching_probe(session, permit)
            if episode is None:
                return
            episode.leader = None
            episode.probe_active = False
            episode.ready_at = time.monotonic()
            self._condition.notify_all()

    async def _attempt_failed(
        self,
        session: ProviderRetrySession,
        permit: _GatePermit,
        *,
        error: Exception,
        status: int | None,
    ) -> bool:
        can_retry = session.can_attempt
        retry_after = _retry_after_seconds(error)
        retry_after_s = None if retry_after is None else round(retry_after, 3)
        if retry_after is not None and retry_after > self._quota_threshold:
            await self._open_terminal_exhaustion(
                session,
                error=error,
                status=status,
                retry_after=retry_after,
            )
            return False

        delay = self._retry_delay(session.attempts_started, retry_after)
        became_leader = False
        exhausted_episode = False

        async with self._condition:
            episode = self._episode
            matching_probe = self._matching_probe(session, permit)
            if can_retry:
                if episode is None:
                    episode = self._start_recovery_episode(
                        leader=session,
                        ready_at=time.monotonic() + delay,
                        last_error=error,
                    )
                    became_leader = True
                elif matching_probe is not None:
                    episode.last_error = error
                    episode.probe_active = False
                    episode.ready_at = time.monotonic() + delay
                    became_leader = True
                elif episode.terminal_until is not None:
                    session._fail_recovery(episode.last_error)
                else:
                    episode.waiters.add(session)
                self._condition.notify_all()
            elif episode is None or matching_probe is not None:
                terminal_delay = self._retry_delay(
                    session.attempts_started,
                    retry_after,
                )
                if episode is None:
                    episode = self._start_recovery_episode(
                        leader=None,
                        ready_at=time.monotonic(),
                        last_error=error,
                        request_id=session.request_id,
                    )
                episode.last_error = error
                episode.leader = None
                episode.probe_active = False
                episode.terminal_until = time.monotonic() + terminal_delay
                for waiter in episode.waiters:
                    waiter._fail_recovery(error)
                episode.waiters.clear()
                exhausted_episode = True
                self._condition.notify_all()

        label = self._failure_label(status, error)
        if became_leader:
            logger.warning(
                "{}, attempt {}/{} failed; one provider recovery probe in {:.1f}s",
                label,
                session.attempts_started,
                session.max_attempts,
                delay,
            )
            trace_event(
                stage="provider",
                event="provider.retry.scheduled",
                source="provider",
                provider=self._provider_name,
                request_id=session.request_id,
                status_code=status,
                exc_type=type(error).__name__,
                attempt=session.attempts_started,
                max_attempts=session.max_attempts,
                delay_s=round(delay, 3),
                retry_after_s=retry_after_s,
                coordinated=True,
            )
        elif can_retry:
            trace_event(
                stage="provider",
                event="provider.retry.coalesced",
                source="provider",
                provider=self._provider_name,
                request_id=session.request_id,
                status_code=status,
                exc_type=type(error).__name__,
                attempt=session.attempts_started,
                max_attempts=session.max_attempts,
            )
        else:
            logger.warning(
                "{} retry exhausted (attempts={})",
                label,
                session.attempts_started,
            )
            trace_event(
                stage="provider",
                event="provider.retry.exhausted",
                source="provider",
                provider=self._provider_name,
                request_id=session.request_id,
                status_code=status,
                exc_type=type(error).__name__,
                attempts=session.attempts_started,
                retry_after_s=retry_after_s,
                episode_exhausted=exhausted_episode,
            )
        return can_retry

    async def _open_terminal_exhaustion(
        self,
        session: ProviderRetrySession,
        *,
        error: Exception,
        status: int | None,
        retry_after: float,
    ) -> None:
        """Fail the provider for a quota window instead of probing through it."""
        window = min(retry_after, TERMINAL_WINDOW_CAP_S)
        async with self._condition:
            episode = self._episode
            if episode is None:
                episode = self._start_recovery_episode(
                    leader=None,
                    ready_at=time.monotonic(),
                    last_error=error,
                    request_id=session.request_id,
                )
            episode.last_error = error
            episode.leader = None
            episode.probe_active = False
            episode.terminal_until = time.monotonic() + window
            for waiter in episode.waiters:
                waiter._fail_recovery(error)
            episode.waiters.clear()
            session._fail_recovery(error)
            generation = episode.generation
            self._condition.notify_all()

        logger.warning(
            "{} reports a {:.0f}s retry delay; provider {} fails fast for {:.0f}s",
            self._failure_label(status, error),
            retry_after,
            self._provider_name,
            window,
        )
        trace_event(
            stage="provider",
            event="provider.recovery.exhausted_long",
            source="provider",
            provider=self._provider_name,
            request_id=session.request_id,
            generation=generation,
            status_code=status,
            exc_type=type(error).__name__,
            attempts=session.attempts_started,
            retry_after_s=round(retry_after, 3),
            terminal_window_s=round(window, 3),
        )

    def _start_recovery_episode(
        self,
        *,
        leader: ProviderRetrySession | None,
        ready_at: float,
        last_error: Exception,
        request_id: str | None = None,
    ) -> _RecoveryEpisode:
        episode = _RecoveryEpisode(
            generation=self._next_generation,
            leader=leader,
            ready_at=ready_at,
            last_error=last_error,
        )
        self._next_generation += 1
        self._episode = episode
        trace_event(
            stage="provider",
            event="provider.recovery.opened",
            source="provider",
            provider=self._provider_name,
            request_id=(leader.request_id if leader is not None else request_id),
            generation=episode.generation,
        )
        return episode

    def _matching_probe(
        self,
        session: ProviderRetrySession,
        permit: _GatePermit,
    ) -> _RecoveryEpisode | None:
        episode = self._episode
        if (
            not permit.probe
            or episode is None
            or episode.generation != permit.generation
            or episode.leader is not session
            or not episode.probe_active
        ):
            return None
        return episode

    async def _abandon_probe_permit(
        self,
        session: ProviderRetrySession,
        permit: _GatePermit,
    ) -> None:
        if not permit.probe:
            return
        async with self._condition:
            episode = self._matching_probe(session, permit)
            if episode is None:
                return
            episode.leader = None
            episode.probe_active = False
            episode.ready_at = time.monotonic()
            self._condition.notify_all()

    async def _abandon_waiting_leader(
        self,
        session: ProviderRetrySession,
        generation: int,
    ) -> None:
        async with self._condition:
            episode = self._episode
            if (
                episode is None
                or episode.generation != generation
                or episode.leader is not session
                or episode.probe_active
            ):
                return
            episode.leader = None
            self._condition.notify_all()

    def _release_concurrency(self) -> None:
        self._concurrency_sem.release()

    def _retry_delay(self, attempt: int, retry_after: float | None) -> float:
        exponent = max(0, attempt - 1)
        backoff = min(self._base_delay * (2**exponent), self._max_delay)
        backoff += random.uniform(0, self._jitter)
        return max(backoff, retry_after or 0.0)

    @staticmethod
    def _failure_label(status: int | None, error: Exception) -> str:
        if status == 429:
            return "Rate limited (429)"
        if status is not None:
            return f"Upstream server error ({status})"
        return f"Provider transient error ({type(error).__name__})"


def _retry_after_seconds(error: Exception) -> float | None:
    """Return an upstream retry hint from the header or a documented error body."""
    header_hint = _retry_after_header_seconds(error)
    if header_hint is not None:
        return header_hint
    return _retry_info_seconds(getattr(error, "body", None))


def _retry_after_header_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if not isinstance(value, str) or not value.strip():
        return None
    stripped = value.strip()
    try:
        seconds = float(stripped)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(stripped)
        except TypeError, ValueError, OverflowError:
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        seconds = (retry_at - datetime.now(UTC)).total_seconds()
    return _finite_seconds(seconds)


def _retry_info_seconds(body: Any) -> float | None:
    """Read ``google.rpc.RetryInfo`` from a Google-style JSON error envelope.

    Google relays this envelope through the Gemini OpenAI-compatible endpoint,
    which carries the machine-readable delay in the body rather than a
    ``Retry-After`` header.
    """
    for detail in _error_details(body):
        if not isinstance(detail, Mapping) or detail.get("@type") != _RETRY_INFO_TYPE:
            continue
        delay = detail.get("retryDelay")
        if not isinstance(delay, str) or not delay.endswith("s"):
            continue
        try:
            seconds = float(delay[:-1])
        except ValueError:
            continue
        parsed = _finite_seconds(seconds)
        if parsed is not None:
            return parsed
    return None


def _error_details(body: Any) -> tuple[Any, ...]:
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            return ()
    if isinstance(body, list):
        return tuple(detail for item in body for detail in _error_details(item))
    if not isinstance(body, Mapping):
        return ()
    error = body.get("error")
    details = (
        error.get("details") if isinstance(error, Mapping) else body.get("details")
    )
    return tuple(details) if isinstance(details, list) else ()


def _finite_seconds(seconds: float) -> float | None:
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)
