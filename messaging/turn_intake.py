"""Inbound messaging turn intake and queue admission."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from loguru import logger

from core.trace import trace_event

from .cli_event_constants import STATUS_MESSAGE_PREFIXES
from .command_context import MessagingCommandContext
from .command_dispatcher import (
    dispatch_command,
    message_kind_for_command,
    parse_command_base,
)
from .models import IncomingMessage
from .platforms.ports import OutboundMessenger
from .safe_diagnostics import format_exception_for_log
from .session import SessionStore
from .trees import MessageNode, MessageState, MessageTree, TreeQueueManager


class MessagingTurnIntake:
    """Owns inbound turn classification and queue admission."""

    def __init__(
        self,
        *,
        platform_name: str,
        outbound: OutboundMessenger,
        session_store: SessionStore,
        command_context: MessagingCommandContext,
        get_tree_queue: Callable[[], TreeQueueManager],
        process_node: Callable[[str, MessageNode], Awaitable[None]],
        format_status: Callable[[str, str, str | None], str],
        get_parse_mode: Callable[[], str | None],
        record_outgoing_message: Callable[[str, str, str | None, str], None],
        log_messaging_error_details: bool = False,
    ) -> None:
        self.platform_name = platform_name
        self.outbound = outbound
        self.session_store = session_store
        self._command_context = command_context
        self._get_tree_queue = get_tree_queue
        self._process_node = process_node
        self._format_status = format_status
        self._get_parse_mode = get_parse_mode
        self._record_outgoing_message = record_outgoing_message
        self._log_messaging_error_details = log_messaging_error_details

    async def handle_message(self, incoming: IncomingMessage) -> None:
        """
        Handle an inbound platform message and queue it if it is a user prompt.
        """
        cmd_base = parse_command_base(incoming.text)

        try:
            if incoming.message_id is not None:
                self.session_store.record_message_id(
                    incoming.platform,
                    incoming.chat_id,
                    str(incoming.message_id),
                    direction="in",
                    kind=message_kind_for_command(cmd_base),
                )
        except Exception as e:
            logger.debug(
                "Failed to record incoming message_id: {}",
                format_exception_for_log(
                    e, log_full_message=self._log_messaging_error_details
                ),
            )

        if await dispatch_command(self._command_context, incoming, cmd_base):
            return

        text = incoming.text or ""
        if any(text.startswith(p) for p in STATUS_MESSAGE_PREFIXES):
            return

        parent_node_id = None
        tree = None
        tree_queue = self._get_tree_queue()

        if incoming.is_reply() and incoming.reply_to_message_id:
            reply_id = incoming.reply_to_message_id
            tree = tree_queue.get_tree_for_node(reply_id)
            if tree:
                parent_node_id = tree_queue.resolve_parent_node_id(reply_id)
                if parent_node_id:
                    logger.info(f"Found tree for reply, parent node: {parent_node_id}")
                else:
                    logger.warning(
                        f"Reply to {incoming.reply_to_message_id} found tree but no valid parent node"
                    )
                    tree = None

        node_id = incoming.message_id
        status_text = self._get_initial_status(tree, parent_node_id)
        if incoming.status_message_id:
            status_msg_id = incoming.status_message_id
            await self.outbound.queue_edit_message(
                incoming.chat_id,
                status_msg_id,
                status_text,
                parse_mode=self._get_parse_mode(),
                fire_and_forget=False,
            )
        else:
            status_msg_id = await self.outbound.queue_send_message(
                incoming.chat_id,
                status_text,
                reply_to=incoming.message_id,
                fire_and_forget=False,
                message_thread_id=incoming.message_thread_id,
            )
        self._record_outgoing_message(
            incoming.platform, incoming.chat_id, status_msg_id, "status"
        )

        tree_queue = self._get_tree_queue()
        if parent_node_id and tree and status_msg_id:
            tree, _node = await tree_queue.add_to_tree(
                parent_node_id=parent_node_id,
                node_id=node_id,
                incoming=incoming,
                status_message_id=status_msg_id,
            )
            tree_queue.register_node(status_msg_id, tree.root_id)
            self.session_store.register_node(status_msg_id, tree.root_id)
            self.session_store.register_node(node_id, tree.root_id)
        elif status_msg_id:
            tree = await tree_queue.create_tree(
                node_id=node_id,
                incoming=incoming,
                status_message_id=status_msg_id,
            )
            tree_queue.register_node(status_msg_id, tree.root_id)
            self.session_store.register_node(node_id, tree.root_id)
            self.session_store.register_node(status_msg_id, tree.root_id)

        if tree:
            self.session_store.save_tree(tree.root_id, tree.to_dict())

        was_queued = await tree_queue.enqueue(
            node_id=node_id,
            processor=self._process_node,
        )

        if was_queued and status_msg_id:
            queue_size = tree_queue.get_queue_size(node_id)
            trace_event(
                stage="routing",
                event="turn.queued",
                source=self.platform_name,
                chat_id=incoming.chat_id,
                platform_message_id=node_id,
                status_message_id=status_msg_id,
                queue_size=queue_size,
            )
            await self.outbound.queue_edit_message(
                incoming.chat_id,
                status_msg_id,
                self._format_status(
                    "📋", "Queued", f"(position {queue_size}) - waiting..."
                ),
                parse_mode=self._get_parse_mode(),
            )

    async def update_queue_positions(self, tree: MessageTree) -> None:
        """Refresh queued status messages after a dequeue."""
        try:
            queued_ids = await tree.get_queue_snapshot()
        except Exception as e:
            logger.warning(
                "Failed to read queue snapshot: {}",
                format_exception_for_log(
                    e, log_full_message=self._log_messaging_error_details
                ),
            )
            return

        if not queued_ids:
            return

        position = 0
        for node_id in queued_ids:
            node = tree.get_node(node_id)
            if not node or node.state != MessageState.PENDING:
                continue
            position += 1
            self.outbound.fire_and_forget(
                self.outbound.queue_edit_message(
                    node.incoming.chat_id,
                    node.status_message_id,
                    self._format_status(
                        "📋", "Queued", f"(position {position}) - waiting..."
                    ),
                    parse_mode=self._get_parse_mode(),
                )
            )

    async def mark_node_processing(self, tree: MessageTree, node_id: str) -> None:
        """Update the dequeued node's status to processing immediately."""
        node = tree.get_node(node_id)
        if not node or node.state == MessageState.ERROR:
            return
        self.outbound.fire_and_forget(
            self.outbound.queue_edit_message(
                node.incoming.chat_id,
                node.status_message_id,
                self._format_status("🔄", "Processing...", None),
                parse_mode=self._get_parse_mode(),
            )
        )

    def _get_initial_status(
        self,
        tree: object | None,
        parent_node_id: str | None,
    ) -> str:
        """Get initial status message text."""
        tree_queue = self._get_tree_queue()
        if tree and parent_node_id:
            if tree_queue.is_node_tree_busy(parent_node_id):
                queue_size = tree_queue.get_queue_size(parent_node_id) + 1
                return self._format_status(
                    "📋", "Queued", f"(position {queue_size}) - waiting..."
                )
            return self._format_status("🔄", "Continuing conversation...", None)

        return self._format_status("⏳", "Launching new Claude CLI instance...", None)


__all__ = ["MessagingTurnIntake"]
