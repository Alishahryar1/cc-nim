"""CLI event handling for a single queued node (transcript + session + errors)."""

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from core.trace import trace_event

from .cli_event_constants import TRANSCRIPT_EVENT_TYPES, get_status_for_event
from .managed_protocols import ManagedClaudeSessionManagerProtocol
from .safe_diagnostics import text_len_hint
from .session import SessionStore
from .transcript import TranscriptBuffer
from .trees import MessageState, MessageTree


async def handle_session_info_event(
    event_data: dict[str, Any],
    tree: MessageTree | None,
    node_id: str,
    captured_session_id: str | None,
    temp_session_id: str | None,
    *,
    cli_manager: ManagedClaudeSessionManagerProtocol,
    session_store: SessionStore,
) -> tuple[str | None, str | None]:
    """Handle session_info event; return updated (captured_session_id, temp_session_id)."""
    if event_data.get("type") != "session_info":
        return captured_session_id, temp_session_id

    real_session_id = event_data.get("session_id")
    if not real_session_id or not temp_session_id:
        return captured_session_id, temp_session_id

    await cli_manager.register_real_session_id(temp_session_id, real_session_id)
    trace_event(
        stage="claude_cli",
        event="claude_cli.session.registered",
        source="claude_cli",
        node_id=node_id,
        temp_session_id=temp_session_id,
        real_session_id=real_session_id,
        tree_root_id=tree.root_id if tree else None,
    )
    if tree and real_session_id:
        await tree.update_state(
            node_id,
            MessageState.IN_PROGRESS,
            session_id=real_session_id,
        )
        session_store.save_tree_snapshot(tree.snapshot())

    return real_session_id, None


async def process_parsed_cli_event(
    parsed: dict[str, Any],
    transcript: TranscriptBuffer,
    update_ui: Callable[..., Awaitable[None]],
    last_status: str | None,
    had_transcript_events: bool,
    tree: MessageTree | None,
    node_id: str,
    captured_session_id: str | None,
    *,
    session_store: SessionStore,
    format_status: Callable[..., str],
    propagate_error_to_children: Callable[[str, str, str], Awaitable[None]],
    log_messaging_error_details: bool = False,
) -> tuple[str | None, bool]:
    """Process a single parsed CLI event. Returns (last_status, had_transcript_events)."""
    ptype = parsed.get("type") or ""

    if ptype in TRANSCRIPT_EVENT_TYPES:
        transcript.apply(parsed)
        had_transcript_events = True

    status = get_status_for_event(ptype, parsed, format_status)
    if status is not None:
        await update_ui(status)
        last_status = status
    elif ptype == "block_stop":
        await update_ui(last_status, force=True)
    elif ptype == "complete":
        if not had_transcript_events:
            transcript.apply({"type": "text_chunk", "text": "Done."})
        trace_event(
            stage="claude_cli",
            event="turn.completed",
            source="cli_event",
            node_id=node_id,
            claude_session_id=captured_session_id,
        )
        await update_ui(format_status("✅", "Complete"), force=True)
        if tree and captured_session_id:
            await tree.update_state(
                node_id,
                MessageState.COMPLETED,
                session_id=captured_session_id,
            )
            session_store.save_tree_snapshot(tree.snapshot())
    elif ptype == "error":
        error_msg = parsed.get("message", "Unknown error")
        em = error_msg if isinstance(error_msg, str) else str(error_msg)
        trace_event(
            stage="claude_cli",
            event="turn.failed",
            source="cli_event",
            node_id=node_id,
            claude_session_id=captured_session_id,
            cli_error_message=em,
        )
        if log_messaging_error_details:
            logger.error("HANDLER: Error event received: {}", error_msg)
        else:
            logger.error(
                "HANDLER: Error event received: message_chars={}",
                text_len_hint(em),
            )
        await update_ui(format_status("❌", "Error"), force=True)
        if tree:
            await propagate_error_to_children(node_id, error_msg, "Parent task failed")

    return last_status, had_transcript_events
