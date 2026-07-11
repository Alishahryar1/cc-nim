"""Typed dependency surface for messaging slash commands."""

from typing import Protocol

from .managed_protocols import ManagedClaudeSessionManagerProtocol
from .models import MessageScope
from .platforms.ports import OutboundMessenger, VoiceCancellation
from .transcript import RenderCtx
from .trees import BranchRemovalResult


class MessagingCommandContext(Protocol):
    """Operations commands need from the messaging workflow."""

    outbound: OutboundMessenger
    voice_cancellation: VoiceCancellation | None
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

    async def stop_task(self, scope: MessageScope, node_id: str) -> int:
        """Stop one pending or active node."""
        ...

    async def resolve_node_id(
        self,
        scope: MessageScope,
        reference_id: str,
    ) -> str | None:
        """Resolve a node or status-message reference."""
        ...

    def get_tree_count(self) -> int:
        """Return the number of conversation trees."""
        ...

    async def clear_branch(
        self,
        scope: MessageScope,
        node_id: str,
    ) -> BranchRemovalResult:
        """Atomically cancel, remove, and persist a conversation branch."""
        ...

    async def clear_all_state(self, platform: str, chat_id: str) -> frozenset[str]:
        """Clear all FCC messaging state and return platform message IDs."""
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


__all__ = ["MessagingCommandContext"]
