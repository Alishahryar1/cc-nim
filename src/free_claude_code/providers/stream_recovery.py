"""Provider-owned stream holdback and recovery decisions."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import httpx
import openai

from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.trace import trace_event

from .failure_policy import retryable_transient_status

EARLY_TRANSPARENT_TOTAL_ATTEMPTS = 5
EARLY_TRANSPARENT_MAX_RETRIES = EARLY_TRANSPARENT_TOTAL_ATTEMPTS - 1
MIDSTREAM_RECOVERY_ATTEMPTS = 5
EARLY_HOLDBACK_SECONDS = 0.75
RECOVERY_BUFFER_MAX_BYTES = 65_536


class TruncatedProviderStreamError(RuntimeError):
    """An upstream stream ended without its required terminal marker."""


class RecoveryFailureAction(StrEnum):
    """How one provider stream should respond to an upstream failure."""

    EARLY_RETRY = "early_retry"
    MIDSTREAM_RECOVERY = "midstream_recovery"
    FINAL_ERROR = "final_error"


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """Failure decision for one provider stream attempt."""

    action: RecoveryFailureAction
    retryable: bool
    committed: bool
    has_buffered: bool
    early_retry_attempt: int | None = None
    midstream_recovery_attempt: int | None = None


class RecoveryHoldbackBuffer:
    """Briefly retain SSE so early cutoffs can be retried invisibly."""

    def __init__(
        self,
        *,
        holdback_seconds: float = EARLY_HOLDBACK_SECONDS,
        max_bytes: int = RECOVERY_BUFFER_MAX_BYTES,
        now: Callable[[], float] | None = None,
    ) -> None:
        self._holdback_seconds = holdback_seconds
        self._max_bytes = max_bytes
        self._now = now or time.monotonic
        self._events: list[str] = []
        self._bytes = 0
        self._started_at: float | None = None
        self.committed = False

    def push(self, event: str) -> list[str]:
        if self.committed:
            return [event]
        if self._started_at is None:
            self._started_at = self._now()
        self._events.append(event)
        self._bytes += len(event.encode("utf-8", errors="replace"))
        if (
            self._bytes >= self._max_bytes
            or self._now() - self._started_at >= self._holdback_seconds
        ):
            return self.flush()
        return []

    def flush(self) -> list[str]:
        if self.committed:
            return []
        self.committed = True
        events = self._events
        self._events = []
        self._bytes = 0
        self._started_at = None
        return events

    def discard(self) -> None:
        self._events = []
        self._bytes = 0
        self._started_at = None

    @property
    def has_buffered(self) -> bool:
        return bool(self._events)


class RecoveryController:
    """Own holdback and retry counters for one provider stream lifecycle."""

    def __init__(self, *, provider_name: str, request_id: str | None) -> None:
        self._provider_name = provider_name
        self._request_id = request_id
        self._holdback = RecoveryHoldbackBuffer()
        self._early_retry_count = 0
        self._midstream_recovery_count = 0

    @property
    def committed(self) -> bool:
        return self._holdback.committed

    @property
    def has_buffered(self) -> bool:
        return self._holdback.has_buffered

    @property
    def early_retries(self) -> int:
        return self._early_retry_count

    @property
    def midstream_recoveries(self) -> int:
        return self._midstream_recovery_count

    def push(self, event: str) -> list[str]:
        return self._holdback.push(event)

    def flush(self) -> list[str]:
        return self._holdback.flush()

    def discard(self) -> None:
        self._holdback.discard()

    def flush_uncommitted(self, decision: RecoveryDecision) -> list[str]:
        if not decision.committed and decision.has_buffered:
            return self.flush()
        return []

    def advance_failure(
        self,
        error: BaseException,
        *,
        stream_opened: bool,
        generated_output: bool,
        complete_tool_salvageable: bool,
    ) -> RecoveryDecision:
        retryable = is_retryable_stream_error(error)
        committed = self._holdback.committed
        has_buffered = self._holdback.has_buffered

        if (
            retryable
            and stream_opened
            and not committed
            and not complete_tool_salvageable
            and self._early_retry_count < EARLY_TRANSPARENT_MAX_RETRIES
        ):
            self._early_retry_count += 1
            self._holdback.discard()
            self._holdback = RecoveryHoldbackBuffer()
            trace_event(
                stage="provider",
                event="provider.recovery.early_retry",
                source="provider",
                provider=self._provider_name,
                request_id=self._request_id,
                retry_attempt=self._early_retry_count,
                retryable=True,
            )
            return RecoveryDecision(
                action=RecoveryFailureAction.EARLY_RETRY,
                retryable=True,
                committed=False,
                has_buffered=has_buffered,
                early_retry_attempt=self._early_retry_count,
            )

        if (
            retryable
            and generated_output
            and self._midstream_recovery_count < MIDSTREAM_RECOVERY_ATTEMPTS
        ):
            self._midstream_recovery_count += 1
            return RecoveryDecision(
                action=RecoveryFailureAction.MIDSTREAM_RECOVERY,
                retryable=True,
                committed=committed,
                has_buffered=has_buffered,
                midstream_recovery_attempt=self._midstream_recovery_count,
            )

        return RecoveryDecision(
            action=RecoveryFailureAction.FINAL_ERROR,
            retryable=retryable,
            committed=committed,
            has_buffered=has_buffered,
        )


def is_retryable_stream_error(exc: BaseException) -> bool:
    """Return whether one stream failure qualifies for retry or recovery."""
    if isinstance(exc, TruncatedProviderStreamError):
        return True
    if isinstance(exc, ExecutionFailure):
        return exc.retryable
    if isinstance(exc, openai.AuthenticationError | openai.BadRequestError):
        return False
    if retryable_transient_status(exc) is not None:
        return True
    return isinstance(
        exc,
        (
            TimeoutError,
            httpx.ReadTimeout,
            httpx.ReadError,
            httpx.RemoteProtocolError,
            httpx.ConnectError,
            httpx.NetworkError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ),
    )
