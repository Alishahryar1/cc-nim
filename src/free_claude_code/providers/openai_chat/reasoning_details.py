"""OpenRouter-format structured reasoning replay and stream conversion."""

import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal

from free_claude_code.core.inference import (
    InferenceEvent,
    InferenceStreamLedger,
    ReplayArtifact,
    ReplayArtifactKind,
    ReplayArtifactOrigin,
    ReplayAttachment,
    ReplayCompatibilityScope,
)


class StructuredReasoningStream:
    """Reconcile alternate plaintext reasoning representations for one stream."""

    def __init__(self, replay_scope: ReplayCompatibilityScope) -> None:
        self._text_source: Literal["native", "details"] | None = None
        self._replay_scope = replay_scope

    def events(
        self,
        delta: Any,
        ledger: InferenceStreamLedger,
        *,
        native_reasoning: str | None,
    ) -> Iterator[InferenceEvent]:
        """Emit plaintext once while preserving every opaque reasoning detail."""
        details = _reasoning_details(delta)
        if self._text_source is None:
            if native_reasoning:
                self._text_source = "native"
            elif any(_reasoning_detail_text(detail) for detail in details):
                self._text_source = "details"

        if self._text_source == "native" and native_reasoning:
            yield from ledger.ensure_reasoning_block()
            yield ledger.emit_reasoning_delta(native_reasoning)

        for detail in details:
            if self._text_source == "details":
                text = _reasoning_detail_text(detail)
                if text:
                    yield from ledger.ensure_reasoning_block()
                    yield ledger.emit_reasoning_delta(text)

            preserved = _preserved_reasoning_detail(detail)
            if preserved:
                yield from ledger.emit_reasoning_artifact(
                    ReplayArtifact(
                        origin=ReplayArtifactOrigin.OPENROUTER,
                        kind=ReplayArtifactKind.REASONING_DETAILS,
                        attachment=ReplayAttachment.REASONING,
                        payload=preserved,
                        scope=self._replay_scope,
                    )
                )


def _reasoning_details(delta: Any) -> Sequence[Any]:
    details = _field(delta, "reasoning_details")
    if details is None:
        extra = _field(delta, "model_extra")
        if isinstance(extra, Mapping):
            details = extra.get("reasoning_details")
    return details if _is_sequence(details) else ()


def _reasoning_detail_text(detail: Any) -> str | None:
    kind = str(_field(detail, "type") or "").lower()
    if "encrypted" in kind or "redacted" in kind:
        return None
    for key in ("text", "content", "reasoning"):
        value = _field(detail, key)
        if isinstance(value, str) and value:
            return value
    return None


def _preserved_reasoning_detail(detail: Any) -> str | None:
    if not isinstance(detail, Mapping):
        return None
    kind = str(_field(detail, "type") or "").lower()
    signature = _field(detail, "signature")
    if (
        "encrypted" in kind
        or "redacted" in kind
        or "summary" in kind
        or isinstance(signature, str)
        or _reasoning_detail_text(detail) is None
    ):
        return json.dumps(dict(detail), separators=(",", ":"))
    return None


def _field(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )
