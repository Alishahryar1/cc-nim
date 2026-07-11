"""Internal messaging tree package facade."""

from .identity import TreeIdentity
from .manager import TreeQueueManager
from .node import MessageState
from .snapshot import ConversationSnapshot, TreeSnapshot
from .transitions import (
    BranchRemovalResult,
    CancellationReason,
    CancellationResult,
    CancellationUiOwner,
    FailureResult,
    NodeClaim,
    NodeUiTarget,
    NodeView,
    QueueDecision,
    QueueEntry,
    ReplyTarget,
)

__all__ = [
    "BranchRemovalResult",
    "CancellationReason",
    "CancellationResult",
    "CancellationUiOwner",
    "ConversationSnapshot",
    "FailureResult",
    "MessageState",
    "NodeClaim",
    "NodeUiTarget",
    "NodeView",
    "QueueDecision",
    "QueueEntry",
    "ReplyTarget",
    "TreeIdentity",
    "TreeQueueManager",
    "TreeSnapshot",
]
