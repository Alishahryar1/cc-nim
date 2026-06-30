"""Typed dependency surface for messaging slash commands."""

from __future__ import annotations

from typing import Protocol

from .managed_protocols import ManagedClaudeSessionManagerProtocol
from .platforms.ports import OutboundMessenger, VoiceCancellation
from .session import SessionStore
from .transcript import RenderCtx
from .trees import MessageNode, MessageTree, TreeQueueManager


class MessagingCommandContext(Protocol):
    """Operations commands need from the messaging workflow."""

    outbound: OutboundMessenger
    voice_cancellation: VoiceCancellation | None
    cli_manager: ManagedClaudeSessionManagerProtocol
    session_store: SessionStore

    @property
    def tree_queue(self) -> TreeQueueManager: ...

    def format_status(self, emoji: str, label: str, suffix: str | None = None) -> str:
        """Format a platform-specific status line."""
        ...

    def get_render_ctx(self) -> RenderCtx:
        """Return the render context for command output."""
        ...

    def replace_tree_queue(self, tree_queue: TreeQueueManager) -> None:
        """Replace the active tree queue after global clear/restore."""
        ...

    async def update_queue_positions(self, tree: MessageTree) -> None:
        """Refresh queued status messages."""
        ...

    async def mark_node_processing(self, tree: MessageTree, node_id: str) -> None:
        """Mark a dequeued node as processing."""
        ...

    async def stop_all_tasks(self) -> int:
        """Stop every pending or active messaging task."""
        ...

    async def stop_task(self, node_id: str) -> int:
        """Stop one pending or active node."""
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

    def update_cancelled_nodes_ui(self, nodes: list[MessageNode]) -> None:
        """Render cancellation status and persist affected trees."""
        ...


__all__ = ["MessagingCommandContext"]
