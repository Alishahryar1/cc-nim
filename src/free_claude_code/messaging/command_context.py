"""Typed dependency surface for messaging slash commands."""

from dataclasses import dataclass
from typing import Protocol

from .managed_protocols import ManagedClaudeSessionManagerProtocol
from .models import MessageScope
from .platforms.ports import OutboundMessenger
from .transcript import RenderCtx


@dataclass(frozen=True, slots=True)
class ReplyClearResult:
    """Customer-facing result of clearing one replied-to owner."""

    message_ids: frozenset[str]
    tree_cleared: bool


class MessagingCommandContext(Protocol):
    """Operations commands need from the messaging workflow."""

    outbound: OutboundMessenger
    cli_manager: ManagedClaudeSessionManagerProtocol

    def format_status(self, emoji: str, label: str, suffix: str | None = None) -> str:
        """Format a platform-specific status line."""
        ...

    def get_render_ctx(self) -> RenderCtx:
        """Return the render context for command output."""
        ...

    async def stop_all_tasks(self) -> int:
        """Stop every pending or active messaging task."""
        ...

    async def stop_reply(
        self,
        scope: MessageScope,
        reply_id: str,
    ) -> int | None:
        """Stop the exact voice/tree owner of a replied-to message."""
        ...

    def get_tree_count(self) -> int:
        """Return the number of conversation trees."""
        ...

    async def clear_reply(
        self,
        scope: MessageScope,
        reply_id: str,
    ) -> ReplyClearResult | None:
        """Clear the exact voice/tree owner of a replied-to message."""
        ...

    async def clear_all_state(self, platform: str, chat_id: str) -> frozenset[str]:
        """Globally clear FCC state and return invoking-chat platform IDs."""
        ...

    def forget_message_ids(
        self,
        platform: str,
        chat_id: str,
        message_ids: set[str],
    ) -> None:
        """Forget deleted platform message IDs."""
        ...

    def record_outgoing_message(
        self,
        platform: str,
        chat_id: str,
        msg_id: str | None,
        kind: str,
    ) -> None:
        """Persist an outgoing platform message ID."""
        ...


__all__ = ["MessagingCommandContext", "ReplyClearResult"]
