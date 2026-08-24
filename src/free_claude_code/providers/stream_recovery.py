"""Legacy Codex stream recovery policy pending shared-supervisor migration."""

from dataclasses import dataclass
from enum import StrEnum

from free_claude_code.core.inference import InferenceEvent
from free_claude_code.providers.streaming.publication import PublicationBuffer

from .failure_policy import RetryableProviderProtocolError


class TruncatedProviderStreamError(RetryableProviderProtocolError):
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


class RecoveryController:
    """Retain Codex's active recovery policy until its PR-4 migration."""

    def __init__(self) -> None:
        self._holdback = PublicationBuffer()

    @property
    def committed(self) -> bool:
        return self._holdback.published

    @property
    def has_buffered(self) -> bool:
        return self._holdback.has_buffered

    def push(self, event: InferenceEvent) -> list[InferenceEvent]:
        publishable = self._holdback.push(event)
        if publishable or self._holdback.seconds_until_deadline != 0:
            return publishable
        return self._holdback.flush()

    def flush(self) -> list[InferenceEvent]:
        return self._holdback.flush()

    def discard(self) -> None:
        self._holdback.discard()

    def flush_uncommitted(self, decision: RecoveryDecision) -> list[InferenceEvent]:
        if not decision.committed and decision.has_buffered:
            return self.flush()
        return []

    def advance_failure(
        self,
        *,
        retryable: bool,
        stream_opened: bool,
        generated_output: bool,
        complete_tool_salvageable: bool,
        attempts_remaining: int,
    ) -> RecoveryDecision:
        committed = self._holdback.published
        has_buffered = self._holdback.has_buffered
        retry_available = attempts_remaining > 0
        reserve_last_attempt_for_recovery = generated_output and attempts_remaining == 1

        if (
            retryable
            and retry_available
            and stream_opened
            and not committed
            and not complete_tool_salvageable
            and not reserve_last_attempt_for_recovery
        ):
            self._holdback.discard()
            return RecoveryDecision(
                action=RecoveryFailureAction.EARLY_RETRY,
                retryable=True,
                committed=False,
                has_buffered=has_buffered,
            )

        if (
            retryable
            and generated_output
            and (retry_available or complete_tool_salvageable)
        ):
            return RecoveryDecision(
                action=RecoveryFailureAction.MIDSTREAM_RECOVERY,
                retryable=True,
                committed=committed,
                has_buffered=has_buffered,
            )

        return RecoveryDecision(
            action=RecoveryFailureAction.FINAL_ERROR,
            retryable=retryable,
            committed=committed,
            has_buffered=has_buffered,
        )
