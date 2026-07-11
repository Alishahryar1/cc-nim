"""Command handlers for messaging platform commands (/stop, /stats, /clear).

Commands depend on MessagingCommandContext instead of the concrete workflow.
"""

from loguru import logger

from .command_context import MessagingCommandContext
from .models import IncomingMessage


async def handle_stop_command(
    handler: MessagingCommandContext, incoming: IncomingMessage
) -> None:
    """Handle /stop command from messaging platform."""
    # Reply-scoped stop: reply "/stop" to stop only that task.
    if incoming.is_reply() and incoming.reply_to_message_id:
        reply_id = incoming.reply_to_message_id
        node_id = await handler.resolve_node_id(incoming.scope, reply_id)

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

        count = await handler.stop_task(incoming.scope, node_id)
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
    tree_count = handler.get_tree_count()
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

    failed = 0
    try:
        await handler.outbound.queue_delete_messages(
            chat_id,
            ordered,
            fire_and_forget=False,
        )
    except Exception as e:
        failed = len(ordered)
        logger.debug("Message delete failed for chat {}: {}", chat_id, type(e).__name__)

    if ordered:
        logger.info(
            "Clear delete attempted={} failed={}",
            len(ordered),
            failed,
        )


async def _handle_clear_branch(
    handler: MessagingCommandContext,
    incoming: IncomingMessage,
    branch_root_id: str,
) -> None:
    """
    Clear a branch (replied-to node + all descendants).

    FCC state is removed and persisted before platform deletion begins.
    """
    result = await handler.clear_branch(incoming.scope, branch_root_id)
    msg_ids = set(result.message_ids)
    if incoming.message_id:
        msg_ids.add(str(incoming.message_id))

    await _delete_message_ids(handler, incoming.chat_id, msg_ids)
    handler.forget_message_ids(incoming.platform, incoming.chat_id, msg_ids)


async def handle_clear_command(
    handler: MessagingCommandContext, incoming: IncomingMessage
) -> None:
    """
    Handle /clear command.

    Reply-scoped: reply to a message to clear that branch (node + descendants).
    Standalone: global clear (stop all, delete all chat messages, reset store).
    """
    if incoming.is_reply() and incoming.reply_to_message_id:
        reply_id = incoming.reply_to_message_id
        branch_root_id = await handler.resolve_node_id(incoming.scope, reply_id)
        if not branch_root_id:
            if handler.voice_cancellation is not None:
                cancelled = await handler.voice_cancellation.cancel_pending_voice(
                    incoming.scope, reply_id
                )
                if cancelled is not None:
                    msg_ids_to_del = {cancelled.voice_message_id}
                    if cancelled.status_message_id is not None:
                        msg_ids_to_del.add(cancelled.status_message_id)
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

    msg_ids = set(await handler.clear_all_state(incoming.platform, incoming.chat_id))

    # Also delete the command message itself.
    if incoming.message_id is not None:
        msg_ids.add(str(incoming.message_id))

    await _delete_message_ids(handler, incoming.chat_id, msg_ids)
