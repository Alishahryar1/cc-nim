"""Platform-agnostic message models."""

from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime, timezone


@dataclass
class IncomingMessage:
    """
    Platform-agnostic incoming message.

    Adapters convert platform-specific events to this format.
    """

    text: str
    chat_id: str
    user_id: str
    message_id: str
    platform: str  # "telegram", "discord", "slack", etc.

    # Optional fields
    reply_to_message_id: Optional[str] = None
    username: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Platform-specific raw event for edge cases
    raw_event: Any = None

    def is_reply(self) -> bool:
        """Check if this message is a reply to another message."""
        return self.reply_to_message_id is not None
