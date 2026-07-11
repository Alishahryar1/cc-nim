"""Message tree node model."""

from dataclasses import dataclass, field
from enum import Enum

from ..models import MessageScope


class MessageState(Enum):
    """State of a message node in the tree."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class MessageNode:
    """A single user prompt/status node in a messaging conversation tree."""

    node_id: str
    scope: MessageScope
    prompt: str
    status_message_id: str
    state: MessageState = MessageState.PENDING
    parent_id: str | None = None
    session_id: str | None = None
    children_ids: list[str] = field(default_factory=list)

    def update_state(
        self,
        state: MessageState,
        *,
        session_id: str | None = None,
    ) -> None:
        self.state = state
        if session_id:
            self.session_id = session_id

    def mark_error(self) -> None:
        self.update_state(MessageState.ERROR)
