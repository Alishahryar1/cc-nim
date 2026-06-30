"""Message tree node model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from ..models import IncomingMessage


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
    incoming: IncomingMessage
    status_message_id: str
    state: MessageState = MessageState.PENDING
    parent_id: str | None = None
    session_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    error_message: str | None = None
    context: Any = None

    def set_context(self, context: Any) -> None:
        self.context = context

    def update_state(
        self,
        state: MessageState,
        *,
        session_id: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self.state = state
        if session_id:
            self.session_id = session_id
        if error_message:
            self.error_message = error_message
        if state in (MessageState.COMPLETED, MessageState.ERROR):
            self.completed_at = datetime.now(UTC)

    def mark_error(self, error_message: str) -> None:
        self.update_state(MessageState.ERROR, error_message=error_message)
