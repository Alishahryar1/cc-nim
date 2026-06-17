"""Shared stream recovery policy for provider transports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from core.anthropic.stream_recovery import (
    EARLY_TRANSPARENT_MAX_RETRIES,
    EARLY_TRANSPARENT_TOTAL_ATTEMPTS,
    RecoveryHoldbackBuffer,
    is_retryable_stream_error,
)
from core.trace import trace_event


class StreamFailureAction(StrEnum):
    """Transport action selected after a provider stream failure."""

    EARLY_RETRY = "early_retry"
    MIDSTREAM_RECOVERY = "midstream_recovery"
    FINAL_ERROR = "final_error"


@dataclass(frozen=True, slots=True)
class StreamFailureDecision:
    """Failure-state snapshot for the current provider stream transition."""

    action: StreamFailureAction
    retryable: bool
    committed: bool
    has_buffered: bool
    early_retry_attempt: int | None = None


class StreamRecoverySession:
    """Own holdback and retry policy shared by provider stream transports."""

    def __init__(self, *, provider_name: str, request_id: str | None) -> None:
        self._provider_name = provider_name
        self._request_id = request_id
        self._holdback = RecoveryHoldbackBuffer()
        self._early_retries = 0

    @property
    def committed(self) -> bool:
        return self._holdback.committed

    @property
    def has_buffered(self) -> bool:
        return self._holdback.has_buffered

    @property
    def early_retries(self) -> int:
        return self._early_retries

    def push(self, event: str) -> list[str]:
        """Buffer one downstream event through the early retry holdback."""
        return self._holdback.push(event)

    def flush(self) -> list[str]:
        """Commit and return held events."""
        return self._holdback.flush()

    def flush_uncommitted(self, decision: StreamFailureDecision) -> list[str]:
        """Commit held events when the decision snapshot is still uncommitted."""
        if decision.committed:
            return []
        return self.flush()

    def discard(self) -> None:
        """Drop held events without committing them."""
        self._holdback.discard()

    def advance_failure(
        self,
        error: BaseException,
        *,
        stream_opened: bool,
        generated_output: bool,
        complete_tool_salvageable: bool,
    ) -> StreamFailureDecision:
        """Consume a stream failure and apply shared recovery state changes."""
        committed = self.committed
        has_buffered = self.has_buffered
        retryable = is_retryable_stream_error(error)

        if (
            not committed
            and stream_opened
            and retryable
            and not complete_tool_salvageable
            and self._early_retries < EARLY_TRANSPARENT_MAX_RETRIES
        ):
            self._early_retries += 1
            attempt = self._early_retries
            self._reset_holdback()
            trace_event(
                stage="provider",
                event="provider.recovery.early_retry",
                source="provider",
                provider=self._provider_name,
                request_id=self._request_id,
                attempt=attempt,
                max_attempts=EARLY_TRANSPARENT_TOTAL_ATTEMPTS,
                exc_type=type(error).__name__,
            )
            return StreamFailureDecision(
                action=StreamFailureAction.EARLY_RETRY,
                retryable=retryable,
                committed=committed,
                has_buffered=has_buffered,
                early_retry_attempt=attempt,
            )

        if generated_output and retryable:
            return StreamFailureDecision(
                action=StreamFailureAction.MIDSTREAM_RECOVERY,
                retryable=retryable,
                committed=committed,
                has_buffered=has_buffered,
            )

        return StreamFailureDecision(
            action=StreamFailureAction.FINAL_ERROR,
            retryable=retryable,
            committed=committed,
            has_buffered=has_buffered,
        )

    def _reset_holdback(self) -> None:
        self._holdback.discard()
        self._holdback = RecoveryHoldbackBuffer()
