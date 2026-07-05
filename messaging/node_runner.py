"""Run queued messaging nodes through a managed CLI session."""

import asyncio
from collections.abc import Callable

from loguru import logger

from core.anthropic import format_user_error_preview, get_user_facing_error_message
from core.trace import trace_event

from .event_parser import parse_cli_event
from .managed_protocols import ManagedClaudeSessionManagerProtocol
from .node_event_pipeline import handle_session_info_event, process_parsed_cli_event
from .platforms.ports import OutboundMessenger
from .safe_diagnostics import format_exception_for_log
from .session import SessionStore
from .transcript import RenderCtx, TranscriptBuffer
from .trees import MessageNode, MessageState, MessageTree, TreeQueueManager
from .ui_updates import ThrottledTranscriptEditor


class MessagingNodeRunner:
    """Owns the lifecycle of one queued messaging node."""

    def __init__(
        self,
        *,
        platform_name: str,
        outbound: OutboundMessenger,
        cli_manager: ManagedClaudeSessionManagerProtocol,
        session_store: SessionStore,
        get_tree_queue: Callable[[], TreeQueueManager],
        format_status: Callable[[str, str, str | None], str],
        get_parse_mode: Callable[[], str | None],
        get_render_ctx: Callable[[], RenderCtx],
        get_limit_chars: Callable[[], int],
        debug_platform_edits: bool = False,
        debug_subagent_stack: bool = False,
        log_raw_cli_diagnostics: bool = False,
        log_messaging_error_details: bool = False,
    ) -> None:
        self.platform_name = platform_name
        self.outbound = outbound
        self.cli_manager = cli_manager
        self.session_store = session_store
        self._get_tree_queue = get_tree_queue
        self._format_status = format_status
        self._get_parse_mode = get_parse_mode
        self._get_render_ctx = get_render_ctx
        self._get_limit_chars = get_limit_chars
        self._debug_platform_edits = debug_platform_edits
        self._debug_subagent_stack = debug_subagent_stack
        self._log_raw_cli_diagnostics = log_raw_cli_diagnostics
        self._log_messaging_error_details = log_messaging_error_details

    def _create_transcript_and_render_ctx(
        self,
    ) -> tuple[TranscriptBuffer, RenderCtx]:
        """Create transcript buffer and render context for node processing."""
        transcript = TranscriptBuffer(
            show_tool_results=False,
            debug_subagent_stack=self._debug_subagent_stack,
        )
        return transcript, self._get_render_ctx()

    def _save_tree(self, tree: MessageTree | None) -> None:
        """Persist tree state after runner-owned mutations."""
        if tree:
            self.session_store.save_tree_snapshot(tree.snapshot())

    async def process_node(
        self,
        node_id: str,
        node: MessageNode,
    ) -> None:
        """Core task processor for a single CLI interaction."""
        incoming = node.incoming
        status_msg_id = node.status_message_id
        chat_id = incoming.chat_id

        with logger.contextualize(node_id=node_id, chat_id=chat_id):
            await self._process_node_impl(node_id, node, chat_id, status_msg_id)

    async def _process_node_impl(
        self,
        node_id: str,
        node: MessageNode,
        chat_id: str,
        status_msg_id: str,
    ) -> None:
        """Internal implementation of process_node with context bound."""
        incoming = node.incoming

        tree_queue = self._get_tree_queue()
        tree = tree_queue.get_tree_for_node(node_id)
        if tree:
            await tree.update_state(node_id, MessageState.IN_PROGRESS)

        transcript, render_ctx = self._create_transcript_and_render_ctx()

        had_transcript_events = False
        captured_session_id = None
        temp_session_id = None
        last_status: str | None = None

        parent_session_id = None
        platform_nm = self.platform_name
        if tree and node.parent_id:
            parent_session_id = tree.get_parent_session_id(node_id)
            if parent_session_id:
                trace_event(
                    stage="claude_cli",
                    event="claude_cli.fork.from_parent_session",
                    source=platform_nm,
                    chat_id=chat_id,
                    node_id=node_id,
                    parent_session_id=parent_session_id,
                )

        editor = ThrottledTranscriptEditor(
            outbound=self.outbound,
            parse_mode=self._get_parse_mode(),
            get_limit_chars=self._get_limit_chars,
            transcript=transcript,
            render_ctx=render_ctx,
            node_id=node_id,
            chat_id=chat_id,
            status_msg_id=status_msg_id,
            debug_platform_edits=self._debug_platform_edits,
            log_messaging_error_details=self._log_messaging_error_details,
        )

        async def update_ui(status: str | None = None, force: bool = False) -> None:
            await editor.update(status, force=force)

        try:
            try:
                (
                    cli_session,
                    session_or_temp_id,
                    is_new,
                ) = await self.cli_manager.get_or_create_session(
                    session_id=parent_session_id
                )
                if is_new:
                    temp_session_id = session_or_temp_id
                else:
                    captured_session_id = session_or_temp_id

                sess_evt = (
                    "claude_cli.session.pending_created"
                    if is_new
                    else "claude_cli.session.reused"
                )
                trace_event(
                    stage="claude_cli",
                    event=sess_evt,
                    source=platform_nm,
                    chat_id=chat_id,
                    node_id=node_id,
                    status_message_id=status_msg_id,
                    session_handle=str(session_or_temp_id),
                    parent_resume_session_id=parent_session_id,
                    fork_requested=bool(parent_session_id),
                )
                trace_event(
                    stage="claude_cli",
                    event="claude_cli.request.sent",
                    source=platform_nm,
                    chat_id=chat_id,
                    node_id=node_id,
                    prompt=incoming.text,
                    fork_session_arg=bool(parent_session_id),
                    resume_session_arg=parent_session_id,
                )
            except RuntimeError as e:
                error_message = get_user_facing_error_message(e)
                transcript.apply({"type": "error", "message": error_message})
                await update_ui(
                    self._format_status("⏳", "Session limit reached", None),
                    force=True,
                )
                if tree:
                    await tree.update_state(
                        node_id,
                        MessageState.ERROR,
                        error_message=error_message,
                    )
                    self._save_tree(tree)
                trace_event(
                    stage="claude_cli",
                    event="claude_cli.session.limit_reached",
                    source=platform_nm,
                    chat_id=chat_id,
                    node_id=node_id,
                )
                return

            async for event_data in cli_session.start_task(
                incoming.text,
                session_id=parent_session_id,
                fork_session=bool(parent_session_id),
            ):
                if not isinstance(event_data, dict):
                    logger.warning(
                        f"HANDLER: Non-dict event received: {type(event_data)}"
                    )
                    continue

                (
                    captured_session_id,
                    temp_session_id,
                ) = await handle_session_info_event(
                    event_data,
                    tree,
                    node_id,
                    captured_session_id,
                    temp_session_id,
                    cli_manager=self.cli_manager,
                    session_store=self.session_store,
                )
                if event_data.get("type") == "session_info":
                    continue

                parsed_list = parse_cli_event(
                    event_data, log_raw_cli=self._log_raw_cli_diagnostics
                )

                for parsed in parsed_list:
                    (
                        last_status,
                        had_transcript_events,
                    ) = await process_parsed_cli_event(
                        parsed,
                        transcript,
                        update_ui,
                        last_status,
                        had_transcript_events,
                        tree,
                        node_id,
                        captured_session_id,
                        session_store=self.session_store,
                        format_status=self._format_status,
                        propagate_error_to_children=self.propagate_error_to_children,
                        log_messaging_error_details=self._log_messaging_error_details,
                    )

        except asyncio.CancelledError:
            trace_event(
                stage="claude_cli",
                event="turn.processor.cancelled",
                source=platform_nm,
                chat_id=chat_id,
                node_id=node_id,
            )
            logger.warning(f"HANDLER: Task cancelled for node {node_id}")
            cancel_reason = None
            if isinstance(node.context, dict):
                cancel_reason = node.context.get("cancel_reason")

            if cancel_reason == "stop":
                await update_ui(self._format_status("⏹", "Stopped.", None), force=True)
            else:
                transcript.apply({"type": "error", "message": "Task was cancelled"})
                await update_ui(
                    self._format_status("❌", "Cancelled", None), force=True
                )

            if tree:
                await tree.update_state(
                    node_id, MessageState.ERROR, error_message="Cancelled by user"
                )
                self._save_tree(tree)
        except Exception as e:
            trace_event(
                stage="claude_cli",
                event="turn.processor.exception",
                source=platform_nm,
                chat_id=chat_id,
                node_id=node_id,
                exc_type=type(e).__name__,
            )
            logger.error(
                "HANDLER: Task failed with exception: {}",
                format_exception_for_log(
                    e, log_full_message=self._log_messaging_error_details
                ),
            )
            error_msg = format_user_error_preview(e)
            transcript.apply({"type": "error", "message": error_msg})
            await update_ui(self._format_status("💥", "Task Failed", None), force=True)
            if tree:
                await self.propagate_error_to_children(
                    node_id, error_msg, "Parent task failed"
                )
        finally:
            trace_event(
                stage="routing",
                event="turn.processor.finished",
                source=platform_nm,
                chat_id=chat_id,
                node_id=node_id,
                claude_session_id=captured_session_id or temp_session_id,
            )
            try:
                if captured_session_id:
                    await self.cli_manager.remove_session(captured_session_id)
                elif temp_session_id:
                    await self.cli_manager.remove_session(temp_session_id)
            except Exception as e:
                logger.debug(
                    "Failed to remove session for node {}: {}",
                    node_id,
                    format_exception_for_log(
                        e, log_full_message=self._log_messaging_error_details
                    ),
                )

    async def propagate_error_to_children(
        self,
        node_id: str,
        error_msg: str,
        child_status_text: str,
    ) -> None:
        """Mark node as error and propagate to pending children with UI updates."""
        tree_queue = self._get_tree_queue()
        affected = await tree_queue.mark_node_error(
            node_id, error_msg, propagate_to_children=True
        )
        if affected:
            self._save_tree(tree_queue.get_tree_for_node(node_id))
        for child in affected[1:]:
            self.outbound.fire_and_forget(
                self.outbound.queue_edit_message(
                    child.incoming.chat_id,
                    child.status_message_id,
                    self._format_status("❌", "Cancelled:", child_status_text),
                    parse_mode=self._get_parse_mode(),
                )
            )


__all__ = ["MessagingNodeRunner"]
