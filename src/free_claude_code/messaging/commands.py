"""Command handlers for messaging platform commands (/stop, /stats, /clear).

Commands depend on MessagingCommandContext instead of the concrete workflow.
"""

from loguru import logger

from .command_context import MessagingCommandContext
from .models import IncomingMessage


async def _send_stop_feedback(
    handler: MessagingCommandContext,
    incoming: IncomingMessage,
    suffix: str,
) -> None:
    """Send stop feedback only when no existing status can represent the result."""
    msg_id = await handler.outbound.queue_send_message(
        incoming.chat_id,
        handler.format_status("⏹", "Stopped.", suffix),
        fire_and_forget=False,
        message_thread_id=incoming.message_thread_id,
    )
    handler.record_outgoing_message(
        incoming.platform, incoming.chat_id, msg_id, "command"
    )


async def handle_stop_command(
    handler: MessagingCommandContext, incoming: IncomingMessage
) -> None:
    """Handle /stop command from messaging platform."""
    # Reply-scoped stop: reply "/stop" to stop only that task.
    if incoming.is_reply() and incoming.reply_to_message_id:
        outcome = await handler.stop_reply(
            incoming.scope,
            incoming.reply_to_message_id,
        )

        if outcome.cancelled_count == 0:
            await _send_stop_feedback(
                handler,
                incoming,
                "Nothing to stop for that message.",
            )
            return

        if outcome.requires_confirmation(incoming.scope):
            noun = "request" if outcome.cancelled_count == 1 else "requests"
            await _send_stop_feedback(
                handler,
                incoming,
                f"Cancelled {outcome.cancelled_count} {noun}.",
            )
        return

    # Global stop: legacy behavior (stop everything)
    outcome = await handler.stop_all_tasks()
    if outcome.cancelled_count == 0:
        await _send_stop_feedback(handler, incoming, "Nothing to stop.")
    elif outcome.requires_confirmation(incoming.scope):
        noun = "request" if outcome.cancelled_count == 1 else "requests"
        await _send_stop_feedback(
            handler,
            incoming,
            f"Cancelled {outcome.cancelled_count} pending or active {noun}.",
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
    non_numeric.sort(reverse=True)
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


async def handle_clear_command(
    handler: MessagingCommandContext, incoming: IncomingMessage
) -> None:
    """
    Handle /clear command.

    Reply-scoped: delete the selected message and its literal reply subtree.
    Standalone: reset and delete the invoking chat's managed conversation.
    """
    if (
        incoming.platform == "telegram"
        and getattr(handler, "apply_config", None) is not None
        and not (incoming.message_id and incoming.message_id.startswith("cb_clear_"))
    ):
        from free_claude_code.messaging.keyboards import make_clear_confirm_keyboard
        msg_id = await handler.outbound.queue_send_message(
            incoming.chat_id,
            "⚠️ Are you sure you want to reset this session?",
            reply_to=incoming.message_id,
            fire_and_forget=False,
            message_thread_id=incoming.message_thread_id,
            reply_markup=make_clear_confirm_keyboard(),
        )
        handler.record_outgoing_message(
            incoming.platform, incoming.chat_id, msg_id, "command"
        )
        return

    from free_claude_code.messaging.trees import TreeQueueManager
    if incoming.is_reply() and incoming.reply_to_message_id:
        result = await handler.clear_reply(
            incoming.scope,
            incoming.reply_to_message_id,
        )
        if result is None:
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

        delete_message_ids = set(result.delete_message_ids)
        if incoming.message_id is not None:
            delete_message_ids.add(str(incoming.message_id))
        await _delete_message_ids(handler, incoming.chat_id, delete_message_ids)
        handler.forget_tracked_message_ids(
            incoming.platform,
            incoming.chat_id,
            delete_message_ids,
        )
        return

    msg_ids = set(await handler.clear_chat(incoming.platform, incoming.chat_id))

    # Also delete the command message itself.
    if incoming.message_id is not None:
        msg_ids.add(str(incoming.message_id))

    await _delete_message_ids(handler, incoming.chat_id, msg_ids)


async def handle_model_command(
    handler: MessagingCommandContext, incoming: IncomingMessage
) -> None:
    """Handle /model command, showing an interactive selection menu."""
    if incoming.platform != "telegram":
        msg_id = await handler.outbound.queue_send_message(
            incoming.chat_id,
            "❌ This command is only supported on Telegram.",
            reply_to=incoming.message_id,
            fire_and_forget=False,
            message_thread_id=incoming.message_thread_id,
        )
        handler.record_outgoing_message(
            incoming.platform, incoming.chat_id, msg_id, "command"
        )
        return

    # Fetch current model
    current_model = ""
    if hasattr(handler, "get_settings") and handler.get_settings:
        settings = handler.get_settings()
        current_model = getattr(settings, "model", "")

    from free_claude_code.messaging.keyboards import make_model_keyboard
    text, kb = make_model_keyboard(current_model)
    msg_id = await handler.outbound.queue_send_message(
        incoming.chat_id,
        text,
        reply_to=incoming.message_id,
        fire_and_forget=False,
        message_thread_id=incoming.message_thread_id,
        reply_markup=kb,
        parse_mode="HTML",
    )
    handler.record_outgoing_message(
        incoming.platform, incoming.chat_id, msg_id, "command"
    )


async def handle_settings_command(
    handler: MessagingCommandContext, incoming: IncomingMessage
) -> None:
    """Handle /settings command, showing an interactive toggle dashboard."""
    if incoming.platform != "telegram":
        msg_id = await handler.outbound.queue_send_message(
            incoming.chat_id,
            "❌ This command is only supported on Telegram.",
            reply_to=incoming.message_id,
            fire_and_forget=False,
            message_thread_id=incoming.message_thread_id,
        )
        handler.record_outgoing_message(
            incoming.platform, incoming.chat_id, msg_id, "command"
        )
        return

    web_tools_enabled = False
    debug_platform_edits = False
    if hasattr(handler, "get_settings") and handler.get_settings:
        settings = handler.get_settings()
        web_tools_enabled = getattr(settings, "enable_web_server_tools", False)
        debug_platform_edits = getattr(settings, "debug_platform_edits", False)

    from free_claude_code.messaging.keyboards import make_settings_keyboard
    msg_id = await handler.outbound.queue_send_message(
        incoming.chat_id,
        "⚙️ *Telegram Bot settings Panel*:",
        reply_to=incoming.message_id,
        fire_and_forget=False,
        message_thread_id=incoming.message_thread_id,
        reply_markup=make_settings_keyboard(
            web_tools_enabled=web_tools_enabled,
            debug_platform_edits=debug_platform_edits,
        ),
    )
    handler.record_outgoing_message(
        incoming.platform, incoming.chat_id, msg_id, "command"
    )


async def handle_workspace_command(
    handler: MessagingCommandContext, incoming: IncomingMessage
) -> None:
    """Handle /workspace command, showing the directory explorer."""
    if incoming.platform != "telegram":
        msg_id = await handler.outbound.queue_send_message(
            incoming.chat_id,
            "❌ This command is only supported on Telegram.",
            reply_to=incoming.message_id,
            fire_and_forget=False,
            message_thread_id=incoming.message_thread_id,
        )
        handler.record_outgoing_message(
            incoming.platform, incoming.chat_id, msg_id, "command"
        )
        return

    workspace_dir = None
    if hasattr(handler, "cli_manager") and handler.cli_manager:
        workspace_dir = handler.cli_manager.workspace

    if not workspace_dir:
        msg_id = await handler.outbound.queue_send_message(
            incoming.chat_id,
            "❌ Workspace directory not found or not initialized.",
            reply_to=incoming.message_id,
            fire_and_forget=False,
            message_thread_id=incoming.message_thread_id,
        )
        handler.record_outgoing_message(
            incoming.platform, incoming.chat_id, msg_id, "command"
        )
        return

    # Extract optional path argument from the command text
    # e.g., "/workspace src" -> "src"
    parts = (incoming.text or "").strip().split(None, 1)
    rel_path = parts[1] if len(parts) > 1 else ""

    from free_claude_code.messaging.keyboards import make_workspace_keyboard
    text, kb = make_workspace_keyboard(workspace_dir, rel_path)

    msg_id = await handler.outbound.queue_send_message(
        incoming.chat_id,
        text,
        reply_to=incoming.message_id,
        fire_and_forget=False,
        message_thread_id=incoming.message_thread_id,
        reply_markup=kb,
        parse_mode="HTML",
    )
    handler.record_outgoing_message(
        incoming.platform, incoming.chat_id, msg_id, "command"
    )


async def handle_start_command(
    handler: MessagingCommandContext, incoming: IncomingMessage
) -> None:
    """Handle /start command, showing greeting and instructions."""
    help_text = (
        "👋 <b>Hello! I am the Claude Code Proxy Bot.</b>\n\n"
        "I help you control and manage Claude Code and Codex directly from Telegram.\n\n"
        "<b>Available Commands & Actions:</b>\n"
        "🤖 Ask me anything by sending a message\n"
        "📁 Browse workspace files and directories\n"
        "⚙️ Manage web search and debug settings\n"
        "🤖 Switch upstream AI model overrides\n"
        "🧹 Clear conversation logs and history\n"
        "🎙 Send voice notes to transcribe and run\n"
        "📥 Upload files to your workspace"
    )

    from free_claude_code.messaging.keyboards import make_start_keyboard
    is_tg = incoming.platform == "telegram"
    reply_markup = make_start_keyboard() if is_tg else None
    parse_mode = "HTML" if is_tg else None

    msg_id = await handler.outbound.queue_send_message(
        incoming.chat_id,
        help_text,
        reply_to=incoming.message_id,
        fire_and_forget=False,
        message_thread_id=incoming.message_thread_id,
        reply_markup=reply_markup,
        parse_mode=parse_mode,
    )
    handler.record_outgoing_message(
        incoming.platform, incoming.chat_id, msg_id, "command"
    )
