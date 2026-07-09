"""Typed cancellation facts for messaging tree UI ownership."""

from dataclasses import dataclass
from enum import Enum

from .node import MessageNode

_CANCEL_REASON_CONTEXT_KEY = "cancel_reason"


class CancellationUiOwner(Enum):
    """Who owns the final user-visible cancellation edit for a node."""

    RUNNER = "runner"
    WORKFLOW = "workflow"


class CancellationReason(Enum):
    """Why a node was cancelled, when the runner needs UI-specific cleanup."""

    STOP = "stop"


@dataclass(frozen=True)
class CancelledNode:
    """A cancelled node plus the component responsible for its final UI edit."""

    node: MessageNode
    ui_owner: CancellationUiOwner


def set_cancel_reason(node: MessageNode, reason: CancellationReason | None) -> None:
    """Attach or remove cancellation reason without clobbering other context."""
    context = dict(node.context) if isinstance(node.context, dict) else {}
    if reason is None:
        context.pop(_CANCEL_REASON_CONTEXT_KEY, None)
    else:
        context[_CANCEL_REASON_CONTEXT_KEY] = reason.value
    node.set_context(context or None)


def get_cancel_reason(node: MessageNode) -> CancellationReason | None:
    """Return a typed cancellation reason from node context."""
    if not isinstance(node.context, dict):
        return None
    raw_reason = node.context.get(_CANCEL_REASON_CONTEXT_KEY)
    if not isinstance(raw_reason, str):
        return None
    try:
        return CancellationReason(raw_reason)
    except ValueError:
        return None
