"""Command handlers for messaging platform commands (/stop, /stats, /clear).

Commands depend on MessagingCommandContext instead of the concrete workflow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from .command_context import MessagingCommandContext

if TYPE_CHECKING:
    from messaging.models import IncomingMessage


async def handle_stop_command(
    handler: MessagingCommandContext, incoming: IncomingMessage
) -> None:
    """Handle /stop command from messaging platform."""
    # Reply-scoped stop: reply "/stop" to stop only that task.
    if incoming.is_reply() and incoming.reply_to_message_id:
        reply_id = incoming.reply_to_message_id
        tree = handler.tree_queue.get_tree_for_node(reply_id)
        node_id = handler.tree_queue.resolve_parent_node_id(reply_id) if tree else None

        if not node_id:
            msg_id = await handler.outbound.queue_send_message(
                incoming.chat_id,
                handler.format_status(
                    "⏹", "Stopped.", "Nothing to stop for that message."
                ),
                fire_and_forget=False,
                message_thread_id=incoming.message_thread_id,
            )
            handler.record_outgoing_message(
                incoming.platform, incoming.chat_id, msg_id, "command"
            )
            return

        count = await handler.stop_task(node_id)
        noun = "request" if count == 1 else "requests"
        msg_id = await handler.outbound.queue_send_message(
            incoming.chat_id,
            handler.format_status("⏹", "Stopped.", f"Cancelled {count} {noun}."),
            fire_and_forget=False,
            message_thread_id=incoming.message_thread_id,
        )
        handler.record_outgoing_message(
            incoming.platform, incoming.chat_id, msg_id, "command"
        )
        return

    # Global stop: legacy behavior (stop everything)
    count = await handler.stop_all_tasks()
    msg_id = await handler.outbound.queue_send_message(
        incoming.chat_id,
        handler.format_status(
            "⏹", "Stopped.", f"Cancelled {count} pending or active requests."
        ),
        fire_and_forget=False,
        message_thread_id=incoming.message_thread_id,
    )
    handler.record_outgoing_message(
        incoming.platform, incoming.chat_id, msg_id, "command"
    )


async def handle_stats_command(
    handler: MessagingCommandContext, incoming: IncomingMessage
) -> None:
    """Handle /stats command."""
    stats = handler.cli_manager.get_stats()
    tree_count = handler.tree_queue.get_tree_count()
    ctx = handler.get_render_ctx()
    msg_id = await handler.outbound.queue_send_message(
        incoming.chat_id,
        "📊 "
        + ctx.bold("Stats")
        + "\n"
        + ctx.escape_text(f"• Active CLI: {stats['active_sessions']}")
        + "\n"
        + ctx.escape_text(f"• Message Trees: {tree_count}"),
        fire_and_forget=False,
        message_thread_id=incoming.message_thread_id,
    )
    handler.record_outgoing_message(
        incoming.platform, incoming.chat_id, msg_id, "command"
    )


async def _delete_message_ids(
    handler: MessagingCommandContext, chat_id: str, msg_ids: set[str]
) -> None:
    """Best-effort delete messages by ID. Sorts numeric IDs descending."""
    if not msg_ids:
        return

    def _as_int(s: str) -> int | None:
        try:
            return int(str(s))
        except Exception:
            return None

    numeric: list[tuple[int, str]] = []
    non_numeric: list[str] = []
    for mid in msg_ids:
        n = _as_int(mid)
        if n is None:
            non_numeric.append(mid)
        else:
            numeric.append((n, mid))
    numeric.sort(reverse=True)
    ordered = [mid for _, mid in numeric] + non_numeric

    try:
        CHUNK = 100
        for i in range(0, len(ordered), CHUNK):
            chunk = ordered[i : i + CHUNK]
            await handler.outbound.queue_delete_messages(
                chat_id, chunk, fire_and_forget=False
            )
    except Exception as e:
        logger.debug(f"Batch delete failed: {type(e).__name__}: {e}")


async def _handle_clear_branch(
    handler: MessagingCommandContext,
    incoming: IncomingMessage,
    branch_root_id: str,
) -> None:
    """
    Clear a branch (replied-to node + all descendants).

    Order: cancel tasks, delete messages, remove branch, update session store.
    """
    tree = handler.tree_queue.get_tree_for_node(branch_root_id)
    if not tree:
        return

    # 1) Cancel branch tasks (no stop_all)
    cancelled = await handler.tree_queue.cancel_branch(branch_root_id)
    handler.update_cancelled_nodes_ui(cancelled)

    # 2) Collect message IDs from branch nodes only
    msg_ids: set[str] = set()
    branch_ids = tree.get_descendants(branch_root_id)
    for nid in branch_ids:
        node = tree.get_node(nid)
        if node:
            if node.incoming.message_id:
                msg_ids.add(str(node.incoming.message_id))
            if node.status_message_id:
                msg_ids.add(str(node.status_message_id))
    if incoming.message_id:
        msg_ids.add(str(incoming.message_id))

    # 3) Delete messages (best-effort)
    await _delete_message_ids(handler, incoming.chat_id, msg_ids)

    # 4) Remove branch from tree
    _removed, root_id, removed_entire_tree = await handler.tree_queue.remove_branch(
        branch_root_id
    )

    # 5) Update session store
    try:
        if removed_entire_tree:
            handler.session_store.remove_tree_snapshot(root_id)
        else:
            updated_tree = handler.tree_queue.get_tree(root_id)
            if updated_tree:
                handler.session_store.save_tree_snapshot(updated_tree.snapshot())
    except Exception as e:
        logger.warning(f"Failed to update session store after branch clear: {e}")


async def handle_clear_command(
    handler: MessagingCommandContext, incoming: IncomingMessage
) -> None:
    """
    Handle /clear command.

    Reply-scoped: reply to a message to clear that branch (node + descendants).
    Standalone: global clear (stop all, delete all chat messages, reset store).
    """
    from messaging.trees import TreeQueueManager

    if incoming.is_reply() and incoming.reply_to_message_id:
        reply_id = incoming.reply_to_message_id
        tree = handler.tree_queue.get_tree_for_node(reply_id)
        branch_root_id = (
            handler.tree_queue.resolve_parent_node_id(reply_id) if tree else None
        )
        if not branch_root_id:
            if handler.voice_cancellation is not None:
                cancelled = await handler.voice_cancellation.cancel_pending_voice(
                    incoming.chat_id, reply_id
                )
                if cancelled is not None:
                    voice_msg_id, status_msg_id = cancelled
                    msg_ids_to_del: set[str] = {voice_msg_id, status_msg_id}
                    if incoming.message_id is not None:
                        msg_ids_to_del.add(str(incoming.message_id))
                    await _delete_message_ids(handler, incoming.chat_id, msg_ids_to_del)
                    msg_id = await handler.outbound.queue_send_message(
                        incoming.chat_id,
                        handler.format_status("🗑", "Cleared.", "Voice note cancelled."),
                        fire_and_forget=False,
                        message_thread_id=incoming.message_thread_id,
                    )
                    handler.record_outgoing_message(
                        incoming.platform, incoming.chat_id, msg_id, "command"
                    )
                    return
            msg_id = await handler.outbound.queue_send_message(
                incoming.chat_id,
                handler.format_status(
                    "🗑", "Cleared.", "Nothing to clear for that message."
                ),
                fire_and_forget=False,
                message_thread_id=incoming.message_thread_id,
            )
            handler.record_outgoing_message(
                incoming.platform, incoming.chat_id, msg_id, "command"
            )
            return
        await _handle_clear_branch(handler, incoming, branch_root_id)
        return

    # Global clear
    # 1) Stop tasks first (ensures no more work is running).
    await handler.stop_all_tasks()

    # 2) Clear chat: best-effort delete messages we can identify.
    msg_ids: set[str] = set()

    # Add any recorded message IDs for this chat (commands, command replies, etc).
    try:
        for mid in handler.session_store.get_message_ids_for_chat(
            incoming.platform, incoming.chat_id
        ):
            if mid is not None:
                msg_ids.add(str(mid))
    except Exception as e:
        logger.debug(f"Failed to read message log for /clear: {e}")

    try:
        msg_ids.update(
            handler.tree_queue.get_message_ids_for_chat(
                incoming.platform, incoming.chat_id
            )
        )
    except Exception as e:
        logger.warning(f"Failed to gather messages for /clear: {e}")

    # Also delete the command message itself.
    if incoming.message_id is not None:
        msg_ids.add(str(incoming.message_id))

    await _delete_message_ids(handler, incoming.chat_id, msg_ids)

    # 3) Clear persistent state and reset in-memory queue/tree state.
    try:
        handler.session_store.clear_all()
    except Exception as e:
        logger.warning(f"Failed to clear session store: {e}")

    handler.replace_tree_queue(
        TreeQueueManager(
            queue_update_callback=handler.update_queue_positions,
            node_started_callback=handler.mark_node_processing,
        )
    )
