"""Messaging workflow coordinator for Discord and Telegram prompts."""

from __future__ import annotations

from loguru import logger

from core.trace import trace_event

from .managed_protocols import ManagedClaudeSessionManagerProtocol
from .models import IncomingMessage
from .node_runner import MessagingNodeRunner
from .platforms.ports import OutboundMessenger, VoiceCancellation
from .rendering.profiles import build_rendering_profile
from .safe_diagnostics import format_exception_for_log
from .session import SessionStore
from .transcript import RenderCtx
from .trees import MessageNode, MessageState, MessageTree, TreeQueueManager
from .turn_intake import MessagingTurnIntake


class MessagingWorkflow:
    """
    Platform-agnostic messaging workflow.

    It coordinates dependencies and owns stop/clear side effects. Inbound turn
    intake and queued node execution live in dedicated collaborators.
    """

    def __init__(
        self,
        outbound: OutboundMessenger,
        cli_manager: ManagedClaudeSessionManagerProtocol,
        session_store: SessionStore,
        *,
        platform_name: str | None = None,
        voice_cancellation: VoiceCancellation | None = None,
        debug_platform_edits: bool = False,
        debug_subagent_stack: bool = False,
        log_raw_messaging_content: bool = False,
        log_raw_cli_diagnostics: bool = False,
        log_messaging_error_details: bool = False,
    ):
        self.platform_name = platform_name or "messaging"
        self.outbound = outbound
        self.voice_cancellation = voice_cancellation
        self.cli_manager = cli_manager
        self.session_store = session_store
        self._log_messaging_error_details = log_messaging_error_details
        self._tree_queue = TreeQueueManager()
        self._rendering_profile = build_rendering_profile(self.platform_name)

        self.node_runner = MessagingNodeRunner(
            platform_name=self.platform_name,
            outbound=outbound,
            cli_manager=cli_manager,
            session_store=session_store,
            get_tree_queue=lambda: self._tree_queue,
            format_status=self.format_status,
            get_parse_mode=self._parse_mode,
            get_render_ctx=self.get_render_ctx,
            get_limit_chars=self._get_limit_chars,
            debug_platform_edits=debug_platform_edits,
            debug_subagent_stack=debug_subagent_stack,
            log_raw_cli_diagnostics=log_raw_cli_diagnostics,
            log_messaging_error_details=log_messaging_error_details,
        )
        self.turn_intake = MessagingTurnIntake(
            platform_name=self.platform_name,
            outbound=outbound,
            session_store=session_store,
            command_context=self,
            get_tree_queue=lambda: self._tree_queue,
            process_node=self.node_runner.process_node,
            format_status=self.format_status,
            get_parse_mode=self._parse_mode,
            record_outgoing_message=self.record_outgoing_message,
            log_messaging_error_details=log_messaging_error_details,
        )
        self._wire_tree_callbacks()

    def _wire_tree_callbacks(self) -> None:
        self._tree_queue.set_queue_update_callback(
            self.turn_intake.update_queue_positions
        )
        self._tree_queue.set_node_started_callback(
            self.turn_intake.mark_node_processing
        )

    def format_status(self, emoji: str, label: str, suffix: str | None = None) -> str:
        return self._rendering_profile.format_status(emoji, label, suffix)

    def _parse_mode(self) -> str | None:
        return self._rendering_profile.parse_mode

    def get_render_ctx(self) -> RenderCtx:
        return self._rendering_profile.render_ctx

    def _get_limit_chars(self) -> int:
        return self._rendering_profile.limit_chars

    @property
    def tree_queue(self) -> TreeQueueManager:
        """Accessor for the current tree queue manager."""
        return self._tree_queue

    def replace_tree_queue(self, tree_queue: TreeQueueManager) -> None:
        """Replace tree queue manager via explicit API."""
        self._tree_queue = tree_queue
        self._wire_tree_callbacks()

    async def handle_message(self, incoming: IncomingMessage) -> None:
        """
        Main entry point for handling an incoming platform message.
        """
        trace_event(
            stage="ingress",
            event="turn.received",
            source=self.platform_name,
            chat_id=incoming.chat_id,
            platform_message_id=incoming.message_id,
            reply_to_message_id=incoming.reply_to_message_id,
            thread_id=getattr(incoming, "message_thread_id", None),
            message_text=incoming.text or "",
        )

        with logger.contextualize(
            chat_id=incoming.chat_id, node_id=incoming.message_id
        ):
            await self.turn_intake.handle_message(incoming)

    async def update_queue_positions(self, tree: MessageTree) -> None:
        """Refresh queued status messages after a dequeue."""
        await self.turn_intake.update_queue_positions(tree)

    async def mark_node_processing(self, tree: MessageTree, node_id: str) -> None:
        """Update the dequeued node's status to processing immediately."""
        await self.turn_intake.mark_node_processing(tree, node_id)

    async def stop_all_tasks(self) -> int:
        """
        Stop all pending and in-progress messaging tasks.
        """
        logger.info("Cancelling tree queue tasks...")
        cancelled_nodes = await self.tree_queue.cancel_all()
        logger.info(f"Cancelled {len(cancelled_nodes)} nodes")

        logger.info("Stopping all CLI sessions...")
        await self.cli_manager.stop_all()

        self.update_cancelled_nodes_ui(cancelled_nodes)
        return len(cancelled_nodes)

    async def stop_task(self, node_id: str) -> int:
        """Stop a single queued or in-progress task node."""
        tree = self.tree_queue.get_tree_for_node(node_id)
        if tree:
            node = tree.get_node(node_id)
            if node and node.state not in (MessageState.COMPLETED, MessageState.ERROR):
                node.set_context({"cancel_reason": "stop"})

        cancelled_nodes = await self.tree_queue.cancel_node(node_id)
        self.update_cancelled_nodes_ui(cancelled_nodes)
        return len(cancelled_nodes)

    def record_outgoing_message(
        self,
        platform: str,
        chat_id: str,
        msg_id: str | None,
        kind: str,
    ) -> None:
        """Record outgoing message ID for /clear. Best-effort, never raises."""
        if not msg_id:
            return
        try:
            self.session_store.record_message_id(
                platform, chat_id, str(msg_id), direction="out", kind=kind
            )
        except Exception as e:
            logger.debug(
                "Failed to record message_id: {}",
                format_exception_for_log(
                    e, log_full_message=self._log_messaging_error_details
                ),
            )

    def update_cancelled_nodes_ui(self, nodes: list[MessageNode]) -> None:
        """Update status messages and persist tree state for cancelled nodes."""
        trees_to_save: dict[str, MessageTree] = {}
        for node in nodes:
            self.outbound.fire_and_forget(
                self.outbound.queue_edit_message(
                    node.incoming.chat_id,
                    node.status_message_id,
                    self.format_status("⏹", "Stopped."),
                    parse_mode=self._parse_mode(),
                )
            )
            tree = self.tree_queue.get_tree_for_node(node.node_id)
            if tree:
                trees_to_save[tree.root_id] = tree
        for root_id, tree in trees_to_save.items():
            self.session_store.save_tree(root_id, tree.to_dict())


__all__ = ["MessagingWorkflow"]
