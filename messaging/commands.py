"""
Command utilities for chat operations like /stop, /stats, /clear.
Refactored for resilience, retries, and cleaner async handling.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Iterable

from loguru import logger

if TYPE_CHECKING:
    from messaging.handler import ClaudeMessageHandler
    from messaging.models import IncomingMessage


# ------------------ Helpers ------------------

async def _safe_send(handler, chat, text, thread_id):
    """Send message with timeout protection."""
    try:
        return await asyncio.wait_for(
            handler.platform.queue_send_message(
                chat,
                text,
                fire_and_forget=False,
                message_thread_id=thread_id,
            ),
            timeout=10,
        )
    except Exception as err:
        logger.error(f"Send failed: {err}")
        return None


async def _retry_delete(handler, chat, ids: Iterable[str], retries: int = 2):
    """Retry deletion for robustness."""
    for attempt in range(retries + 1):
        try:
            await handler.platform.queue_delete_messages(
                chat, list(ids), fire_and_forget=False
            )
            return
        except Exception as err:
            logger.debug(f"Delete attempt {attempt+1} failed: {err}")
            await asyncio.sleep(0.5)


def _format_ids(values: set[str]) -> list[str]:
    """Sort numeric IDs descending, others appended."""
    def parse(x):
        try:
            return int(x)
        except Exception:
            return None

    numeric = [(parse(v), v) for v in values if parse(v) is not None]
    text = [v for v in values if parse(v) is None]

    numeric.sort(reverse=True)
    return [v for _, v in numeric] + text


# ------------------ STOP ------------------

async def handle_stop_command(handler: ClaudeMessageHandler, incoming: IncomingMessage):
    """Stop running tasks (scoped or global)."""

    def make_msg(text):
        return handler.format_status("⏹", "Stopped.", text)

    if incoming.is_reply() and incoming.reply_to_message_id:
        ref = incoming.reply_to_message_id
        tree_obj = handler.tree_queue.get_tree_for_node(ref)

        node = (
            handler.tree_queue.resolve_parent_node_id(ref)
            if tree_obj else None
        )

        if node:
            total = await handler.stop_task(node)
            label = "request" if total == 1 else "requests"
            msg = make_msg(f"Cancelled {total} {label}.")
        else:
            msg = make_msg("Nothing to stop for that message.")

        sent = await _safe_send(handler, incoming.chat_id, msg, incoming.message_thread_id)
        if sent:
            handler.record_outgoing_message(incoming.platform, incoming.chat_id, sent, "command")
        return

    total = await handler.stop_all_tasks()
    msg = make_msg(f"Cancelled {total} pending or active requests.")

    sent = await _safe_send(handler, incoming.chat_id, msg, incoming.message_thread_id)
    if sent:
        handler.record_outgoing_message(incoming.platform, incoming.chat_id, sent, "command")


# ------------------ STATS ------------------

async def handle_stats_command(handler: ClaudeMessageHandler, incoming: IncomingMessage):
    """Display system stats."""
    stats = handler.cli_manager.get_stats()
    trees = handler.tree_queue.get_tree_count()
    ctx = handler.get_render_ctx()

    content = (
        "📊 "
        + ctx.bold("Stats")
        + "\n"
        + ctx.escape_text(f"• Active CLI: {stats.get('active_sessions', 0)}")
        + "\n"
        + ctx.escape_text(f"• Message Trees: {trees}")
    )

    sent = await _safe_send(handler, incoming.chat_id, content, incoming.message_thread_id)
    if sent:
        handler.record_outgoing_message(incoming.platform, incoming.chat_id, sent, "command")


# ------------------ CLEAR (CORE) ------------------

async def _collect_branch_ids(handler, root_id, incoming):
    """Gather all related message IDs."""
    ids = set()
    tree = handler.tree_queue.get_tree_for_node(root_id)

    if not tree:
        return ids

    for nid in tree.get_descendants(root_id):
        node = tree.get_node(nid)
        if not node:
            continue

        if node.incoming.message_id:
            ids.add(str(node.incoming.message_id))

        if node.status_message_id:
            ids.add(str(node.status_message_id))

    if incoming.message_id:
        ids.add(str(incoming.message_id))

    return ids


async def _clear_branch(handler, incoming, root_id):
    """Clear a subtree safely."""
    cancelled_nodes = await handler.tree_queue.cancel_branch(root_id)
    handler.update_cancelled_nodes_ui(cancelled_nodes)

    ids = await _collect_branch_ids(handler, root_id, incoming)
    ordered = _format_ids(ids)

    if ordered:
        await _retry_delete(handler, incoming.chat_id, ordered)

    removed, tree_root, wiped = await handler.tree_queue.remove_branch(root_id)

    try:
        handler.session_store.remove_node_mappings([n.node_id for n in removed])

        if wiped:
            handler.session_store.remove_tree(tree_root)
        else:
            updated = handler.tree_queue.get_tree(tree_root)
            if updated:
                handler.session_store.save_tree(tree_root, updated.to_dict())
    except Exception as err:
        logger.warning(f"State update failed: {err}")


# ------------------ CLEAR (ENTRY) ------------------

async def handle_clear_command(handler: ClaudeMessageHandler, incoming: IncomingMessage):
    """Handle clearing logic (branch or full reset)."""
    from messaging.trees import TreeQueueManager

    def msg(text):
        return handler.format_status("🗑", "Cleared.", text)

    if incoming.is_reply() and incoming.reply_to_message_id:
        ref = incoming.reply_to_message_id
        tree = handler.tree_queue.get_tree_for_node(ref)

        root = (
            handler.tree_queue.resolve_parent_node_id(ref)
            if tree else None
        )

        if root:
            await _clear_branch(handler, incoming, root)
            return

        cancel_voice = getattr(handler.platform, "cancel_pending_voice", None)
        if cancel_voice:
            result = await cancel_voice(incoming.chat_id, ref)
            if result:
                voice_id, status_id = result
                ids = {voice_id, status_id}
                if incoming.message_id:
                    ids.add(str(incoming.message_id))

                await _retry_delete(handler, incoming.chat_id, ids)

                sent = await _safe_send(handler, incoming.chat_id, msg("Voice note cancelled."), incoming.message_thread_id)
                if sent:
                    handler.record_outgoing_message(incoming.platform, incoming.chat_id, sent, "command")
                return

        sent = await _safe_send(handler, incoming.chat_id, msg("Nothing to clear for that message."), incoming.message_thread_id)
        if sent:
            handler.record_outgoing_message(incoming.platform, incoming.chat_id, sent, "command")
        return

    # -------- GLOBAL CLEAR --------

    await handler.stop_all_tasks()

    ids = set()

    try:
        ids.update(
            str(x)
            for x in handler.session_store.get_message_ids_for_chat(
                incoming.platform, incoming.chat_id
            )
            if x is not None
        )
    except Exception as err:
        logger.debug(f"Session read error: {err}")

    try:
        ids.update(
            handler.tree_queue.get_message_ids_for_chat(
                incoming.platform, incoming.chat_id
            )
        )
    except Exception as err:
        logger.warning(f"Tree read error: {err}")

    if incoming.message_id:
        ids.add(str(incoming.message_id))

    ordered = _format_ids(ids)

    if ordered:
        await _retry_delete(handler, incoming.chat_id, ordered)

    try:
        handler.session_store.clear_all()
    except Exception as err:
        logger.warning(f"Storage clear failed: {err}")

    handler.replace_tree_queue(
        TreeQueueManager(
            queue_update_callback=handler.update_queue_positions,
            node_started_callback=handler.mark_node_processing,
        )
    )            handler.record_outgoing_message(
                incoming.platform, incoming.chat_id, msg_id, "command"
            )
            return

        count = await handler.stop_task(node_id)
        noun = "request" if count == 1 else "requests"
        msg_id = await handler.platform.queue_send_message(
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
    msg_id = await handler.platform.queue_send_message(
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
    handler: ClaudeMessageHandler, incoming: IncomingMessage
) -> None:
    """Handle /stats command."""
    stats = handler.cli_manager.get_stats()
    tree_count = handler.tree_queue.get_tree_count()
    ctx = handler.get_render_ctx()
    msg_id = await handler.platform.queue_send_message(
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
    handler: ClaudeMessageHandler, chat_id: str, msg_ids: set[str]
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
            await handler.platform.queue_delete_messages(
                chat_id, chunk, fire_and_forget=False
            )
    except Exception as e:
        logger.debug(f"Batch delete failed: {type(e).__name__}: {e}")


async def _handle_clear_branch(
    handler: ClaudeMessageHandler,
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
    removed, root_id, removed_entire_tree = await handler.tree_queue.remove_branch(
        branch_root_id
    )

    # 5) Update session store
    try:
        handler.session_store.remove_node_mappings([n.node_id for n in removed])
        if removed_entire_tree:
            handler.session_store.remove_tree(root_id)
        else:
            updated_tree = handler.tree_queue.get_tree(root_id)
            if updated_tree:
                handler.session_store.save_tree(root_id, updated_tree.to_dict())
    except Exception as e:
        logger.warning(f"Failed to update session store after branch clear: {e}")


async def handle_clear_command(
    handler: ClaudeMessageHandler, incoming: IncomingMessage
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
            cancel_fn = getattr(handler.platform, "cancel_pending_voice", None)
            if cancel_fn is not None:
                cancelled = await cancel_fn(incoming.chat_id, reply_id)
                if cancelled is not None:
                    voice_msg_id, status_msg_id = cancelled
                    msg_ids_to_del: set[str] = {voice_msg_id, status_msg_id}
                    if incoming.message_id is not None:
                        msg_ids_to_del.add(str(incoming.message_id))
                    await _delete_message_ids(handler, incoming.chat_id, msg_ids_to_del)
                    msg_id = await handler.platform.queue_send_message(
                        incoming.chat_id,
                        handler.format_status("🗑", "Cleared.", "Voice note cancelled."),
                        fire_and_forget=False,
                        message_thread_id=incoming.message_thread_id,
                    )
                    handler.record_outgoing_message(
                        incoming.platform, incoming.chat_id, msg_id, "command"
                    )
                    return
            msg_id = await handler.platform.queue_send_message(
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
