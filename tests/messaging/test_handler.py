from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from messaging.models import IncomingMessage
from messaging.trees import MessageNode, MessageState, MessageTree
from messaging.workflow import MessagingWorkflow


@pytest.fixture
def handler(mock_platform, mock_cli_manager, mock_session_store):
    return MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        mock_session_store,
        voice_cancellation=mock_platform,
    )


@pytest.mark.asyncio
async def test_handle_message_turn_trace_includes_full_message_text(
    mock_platform, mock_cli_manager, mock_session_store, incoming_message_factory
):
    """turn.received always records the verbatim user message (local debugging)."""
    secret = "user-message-content-visible-in-trace"
    handler = MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        mock_session_store,
        log_raw_messaging_content=False,
    )
    incoming = incoming_message_factory(text=secret)
    with (
        patch.object(handler.turn_intake, "handle_message", new_callable=AsyncMock),
        patch("messaging.workflow.trace_event") as trace_mock,
    ):
        await handler.handle_message(incoming)
    kwargs = trace_mock.call_args.kwargs
    assert kwargs["event"] == "turn.received"
    assert kwargs["message_text"] == secret


@pytest.mark.asyncio
async def test_handle_message_log_raw_messaging_does_not_change_turn_received_shape(
    mock_platform, mock_cli_manager, mock_session_store, incoming_message_factory
):
    """LOG_RAW_MESSAGING_CONTENT is adapter-only; ingress TRACE always includes text."""
    text = "visible-either-way"
    handler = MessagingWorkflow(
        mock_platform,
        mock_cli_manager,
        mock_session_store,
        log_raw_messaging_content=True,
    )
    incoming = incoming_message_factory(text=text)
    with (
        patch.object(handler.turn_intake, "handle_message", new_callable=AsyncMock),
        patch("messaging.workflow.trace_event") as trace_mock,
    ):
        await handler.handle_message(incoming)
    assert trace_mock.call_args.kwargs["message_text"] == text


def test_get_initial_status_new_conversation(handler):
    """New conversation always returns launching message."""
    result = handler.turn_intake._get_initial_status(None, None)
    assert "Launching" in result


def test_get_initial_status_reply_tree_busy_queued(handler):
    """Reply to tree when busy returns queued message."""
    mock_queue = MagicMock()
    mock_queue.is_node_tree_busy.return_value = True
    mock_queue.get_queue_size.return_value = 2
    handler.replace_tree_queue(mock_queue)
    result = handler.turn_intake._get_initial_status(MagicMock(), "parent_1")
    assert "Queued" in result
    assert "position 3" in result


def test_get_initial_status_reply_tree_not_busy_continuing(handler):
    """Reply to tree when not busy returns continuing message."""
    mock_queue = MagicMock()
    mock_queue.is_node_tree_busy.return_value = False
    handler.replace_tree_queue(mock_queue)
    result = handler.turn_intake._get_initial_status(MagicMock(), "parent_1")
    assert "Continuing" in result


@pytest.mark.asyncio
async def test_handle_message_stop_command(
    handler, mock_platform, incoming_message_factory
):
    incoming = incoming_message_factory(text="/stop")

    # Mock stop_all_tasks
    handler.stop_all_tasks = AsyncMock(return_value=5)

    await handler.handle_message(incoming)

    handler.stop_all_tasks.assert_called_once()
    mock_platform.queue_send_message.assert_called_once_with(
        incoming.chat_id,
        "⏹ *Stopped\\.* Cancelled 5 pending or active requests\\.",
        fire_and_forget=False,
        message_thread_id=None,
    )


@pytest.mark.asyncio
async def test_handle_message_stop_command_reply_stops_only_target_node(
    handler, mock_platform, mock_cli_manager, incoming_message_factory
):
    # Create a tree with a root node and register its status message ID mapping.
    root_incoming = incoming_message_factory(
        text="do something", message_id="root_msg", reply_to_message_id=None
    )
    tree = await handler.tree_queue.create_tree(
        node_id="root_msg",
        incoming=root_incoming,
        status_message_id="status_root",
    )
    handler.tree_queue.register_node("status_root", tree.root_id)

    # Reply "/stop" to the status message; should stop only that node.
    incoming = incoming_message_factory(
        text="/stop",
        message_id="stop_msg",
        reply_to_message_id="status_root",
    )

    handler.stop_all_tasks = AsyncMock(return_value=999)

    await handler.handle_message(incoming)

    handler.stop_all_tasks.assert_not_called()
    mock_cli_manager.stop_all.assert_not_called()
    assert tree.get_node("root_msg").state == MessageState.ERROR
    mock_platform.queue_send_message.assert_called_once_with(
        incoming.chat_id,
        "⏹ *Stopped\\.* Cancelled 1 request\\.",
        fire_and_forget=False,
        message_thread_id=None,
    )


@pytest.mark.asyncio
async def test_handle_message_stop_command_reply_unknown_does_not_stop_all(
    handler, mock_platform, mock_cli_manager, incoming_message_factory
):
    incoming = incoming_message_factory(
        text="/stop",
        message_id="stop_msg",
        reply_to_message_id="unknown_msg",
    )

    handler.stop_all_tasks = AsyncMock(return_value=5)

    await handler.handle_message(incoming)

    handler.stop_all_tasks.assert_not_called()
    mock_cli_manager.stop_all.assert_not_called()
    mock_platform.queue_send_message.assert_called_once_with(
        incoming.chat_id,
        "⏹ *Stopped\\.* Nothing to stop for that message\\.",
        fire_and_forget=False,
        message_thread_id=None,
    )


@pytest.mark.asyncio
async def test_handle_message_stats_command(
    handler, mock_platform, mock_cli_manager, incoming_message_factory
):
    incoming = incoming_message_factory(text="/stats")
    mock_cli_manager.get_stats.return_value = {"active_sessions": 2}

    await handler.handle_message(incoming)

    mock_platform.queue_send_message.assert_called_once()
    args, kwargs = mock_platform.queue_send_message.call_args
    assert "Active CLI: 2" in args[1]
    assert kwargs["fire_and_forget"] is False
    assert kwargs.get("message_thread_id") is None


@pytest.mark.asyncio
async def test_handle_message_filters_status_messages(
    handler, mock_platform, incoming_message_factory
):
    incoming = incoming_message_factory(text="⏳ Thinking...")

    await handler.handle_message(incoming)

    mock_platform.queue_send_message.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_new_conversation(
    handler, mock_platform, mock_session_store, incoming_message_factory
):
    incoming = incoming_message_factory(text="hello")
    mock_platform.queue_send_message.return_value = "status_123"

    # We need to mock tree_queue methods
    with (
        patch.object(handler.tree_queue, "create_tree", AsyncMock()) as mock_create,
        patch.object(
            handler.tree_queue, "enqueue", AsyncMock(return_value=False)
        ) as mock_enqueue,
    ):
        mock_tree = MagicMock()
        mock_tree.root_id = "root_1"
        mock_tree.snapshot.return_value = {"data": "tree"}
        mock_create.return_value = mock_tree

        await handler.handle_message(incoming)

        mock_create.assert_called_once()
        mock_enqueue.assert_called_once()
        mock_session_store.save_tree_snapshot.assert_called_once_with({"data": "tree"})


@pytest.mark.asyncio
async def test_handle_message_queued(handler, mock_platform, incoming_message_factory):
    incoming = incoming_message_factory(text="hello", message_id="msg_1")
    mock_platform.queue_send_message.return_value = "status_123"

    with (
        patch.object(handler.tree_queue, "create_tree", AsyncMock()) as mock_create,
        patch.object(handler.tree_queue, "enqueue", AsyncMock(return_value=True)),
        patch.object(handler.tree_queue, "get_queue_size", MagicMock(return_value=3)),
    ):
        mock_tree = MagicMock()
        mock_tree.root_id = "root_1"
        mock_tree.snapshot.return_value = {}
        mock_create.return_value = mock_tree

        await handler.handle_message(incoming)

    mock_platform.queue_edit_message.assert_called_once_with(
        incoming.chat_id,
        "status_123",
        "📋 *Queued* \\(position 3\\) \\- waiting\\.\\.\\.",
        parse_mode="MarkdownV2",
    )


@pytest.mark.asyncio
async def test_update_queue_positions(handler, mock_platform):
    root_incoming = IncomingMessage(
        text="Root",
        chat_id="chat_1",
        user_id="user_1",
        message_id="root",
        platform="telegram",
    )
    root = MessageNode(
        node_id="root",
        incoming=root_incoming,
        status_message_id="status_root",
    )
    tree = MessageTree(root)

    child_incoming_1 = IncomingMessage(
        text="Child 1",
        chat_id="chat_1",
        user_id="user_1",
        message_id="child_1",
        platform="telegram",
        reply_to_message_id="root",
    )
    child_incoming_2 = IncomingMessage(
        text="Child 2",
        chat_id="chat_1",
        user_id="user_1",
        message_id="child_2",
        platform="telegram",
        reply_to_message_id="root",
    )

    await tree.add_node(
        node_id="child_1",
        incoming=child_incoming_1,
        status_message_id="status_1",
        parent_id="root",
    )
    await tree.add_node(
        node_id="child_2",
        incoming=child_incoming_2,
        status_message_id="status_2",
        parent_id="root",
    )

    await tree.enqueue("child_1")
    await tree.enqueue("child_2")

    await handler.update_queue_positions(tree)

    calls = mock_platform.queue_edit_message.call_args_list
    assert len(calls) == 2
    assert calls[0][0][0] == "chat_1"
    assert calls[0][0][1] == "status_1"
    assert "position 1" in calls[0][0][2]
    assert calls[1][0][0] == "chat_1"
    assert calls[1][0][1] == "status_2"
    assert "position 2" in calls[1][0][2]


@pytest.mark.asyncio
async def test_mark_node_processing(handler, mock_platform):
    root_incoming = IncomingMessage(
        text="Root",
        chat_id="chat_1",
        user_id="user_1",
        message_id="root",
        platform="telegram",
    )
    root = MessageNode(
        node_id="root",
        incoming=root_incoming,
        status_message_id="status_root",
    )
    tree = MessageTree(root)

    child_incoming = IncomingMessage(
        text="Child",
        chat_id="chat_1",
        user_id="user_1",
        message_id="child",
        platform="telegram",
        reply_to_message_id="root",
    )

    await tree.add_node(
        node_id="child",
        incoming=child_incoming,
        status_message_id="status_child",
        parent_id="root",
    )

    await handler.mark_node_processing(tree, "child")

    mock_platform.queue_edit_message.assert_called_once()
    args, kwargs = mock_platform.queue_edit_message.call_args
    assert args[0] == "chat_1"
    assert args[1] == "status_child"
    assert "Processing" in args[2]
    assert kwargs["parse_mode"] == "MarkdownV2"


@pytest.mark.asyncio
async def test_stop_all_tasks(handler, mock_cli_manager, mock_platform):
    mock_node = MagicMock()
    mock_node.incoming.chat_id = "chat_1"
    mock_node.status_message_id = "status_1"

    with patch.object(
        handler.tree_queue, "cancel_all", AsyncMock(return_value=[mock_node])
    ):
        count = await handler.stop_all_tasks()

        assert count == 1
        mock_cli_manager.stop_all.assert_called_once()
        mock_platform.fire_and_forget.assert_called_once()


async def mock_async_gen(events):
    for e in events:
        yield e


@pytest.mark.asyncio
async def test_node_runner_process_node_success_flow(
    handler, mock_cli_manager, mock_platform
):
    # Setup
    node_id = "node_1"
    mock_node = MagicMock()
    mock_node.incoming.chat_id = "chat_1"
    mock_node.incoming.text = "hello"
    mock_node.status_message_id = "status_1"
    mock_node.parent_id = None

    mock_session = MagicMock()
    # Mock start_task to return our async generator
    events = [
        {
            "type": "assistant",
            "message": {"content": [{"type": "thinking", "thinking": "Let me think"}]},
        },
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hello world"}]},
        },
        {"type": "exit", "code": 0},
    ]
    mock_session.start_task.return_value = mock_async_gen(events)

    mock_cli_manager.get_or_create_session.return_value = (
        mock_session,
        "session_1",
        False,
    )

    mock_tree = MagicMock()
    mock_tree.update_state = AsyncMock()
    mock_tree.root_id = "root_1"
    mock_tree.snapshot.return_value = {}

    with patch.object(
        handler.tree_queue, "get_tree_for_node", MagicMock(return_value=mock_tree)
    ):
        await handler.node_runner.process_node(node_id, mock_node)

        # Verify state updates
        mock_tree.update_state.assert_any_call(node_id, MessageState.IN_PROGRESS)
        mock_tree.update_state.assert_any_call(
            node_id, MessageState.COMPLETED, session_id="session_1"
        )

        # Verify UI updates (at least the final one)
        # Note: update_ui is debounced, but COMPLETED/ERROR/CANCELLED are forced
        mock_platform.queue_edit_message.assert_called()
        last_call = mock_platform.queue_edit_message.call_args_list[-1]
        assert "✅ *Complete*" in last_call[0][2]
        assert "Hello world" in last_call[0][2]

    mock_cli_manager.get_or_create_session.assert_awaited_once_with(session_id=None)
    mock_session.start_task.assert_called_once()
    st_kw = mock_session.start_task.call_args
    assert st_kw.kwargs.get("session_id") is None
    assert st_kw.kwargs.get("fork_session") is False


@pytest.mark.asyncio
async def test_node_runner_process_node_reply_uses_parent_session_for_manager_and_fork(
    handler, mock_cli_manager, mock_platform
):
    """Telegram follow-ups must reuse parent Claude session (issue #233)."""
    node_id = "child_1"
    mock_node = MagicMock()
    mock_node.incoming.chat_id = "chat_1"
    mock_node.incoming.text = "follow up"
    mock_node.status_message_id = "status_child"
    mock_node.parent_id = "root_msg"

    parent_claude_session = "claude_sess_parent"
    mock_session = MagicMock()
    mock_session.start_task.return_value = mock_async_gen([{"type": "exit", "code": 0}])
    mock_cli_manager.get_or_create_session.return_value = (
        mock_session,
        parent_claude_session,
        False,
    )

    mock_tree = MagicMock()
    mock_tree.update_state = AsyncMock()
    mock_tree.root_id = "root_msg"
    mock_tree.snapshot.return_value = {}
    mock_tree.get_parent_session_id = MagicMock(return_value=parent_claude_session)

    with patch.object(
        handler.tree_queue, "get_tree_for_node", MagicMock(return_value=mock_tree)
    ):
        await handler.node_runner.process_node(node_id, mock_node)

    mock_tree.get_parent_session_id.assert_called_once_with(node_id)
    mock_cli_manager.get_or_create_session.assert_awaited_once_with(
        session_id=parent_claude_session
    )
    mock_session.start_task.assert_called_once()
    st_kw = mock_session.start_task.call_args
    assert st_kw.kwargs.get("session_id") == parent_claude_session
    assert st_kw.kwargs.get("fork_session") is True


@pytest.mark.asyncio
async def test_node_runner_process_node_error_flow(
    handler, mock_cli_manager, mock_platform
):
    node_id = "node_1"
    mock_node = MagicMock()
    mock_node.incoming.chat_id = "chat_1"
    mock_node.incoming.text = "hello"
    mock_node.status_message_id = "status_1"

    mock_session = MagicMock()
    events = [{"type": "error", "error": {"message": "CLI crashed"}}]
    mock_session.start_task.return_value = mock_async_gen(events)
    mock_cli_manager.get_or_create_session.return_value = (
        mock_session,
        "session_1",
        False,
    )

    mock_tree = MagicMock()
    mock_tree.root_id = "root_1"
    mock_tree.snapshot.return_value = {"data": "tree"}
    mock_tree.update_state = AsyncMock()

    with (
        patch.object(
            handler.tree_queue, "get_tree_for_node", MagicMock(return_value=mock_tree)
        ),
        patch.object(
            handler.tree_queue, "mark_node_error", AsyncMock(return_value=[mock_node])
        ),
    ):
        await handler.node_runner.process_node(node_id, mock_node)

        handler.tree_queue.mark_node_error.assert_called_once_with(
            node_id, "CLI crashed", propagate_to_children=True
        )
        handler.session_store.save_tree_snapshot.assert_called_once_with(
            {"data": "tree"}
        )

        last_call = mock_platform.queue_edit_message.call_args_list[-1]
        assert "❌ *Error*" in last_call[0][2]
        assert "CLI crashed" in last_call[0][2]


@pytest.mark.asyncio
async def test_handle_message_clear_command_stops_deletes_and_wipes_state(
    handler, mock_platform, mock_session_store, incoming_message_factory
):
    # Create some tracked messages across two chats. /clear should only delete
    # messages for the current chat.
    root_1 = incoming_message_factory(
        text="do something",
        chat_id="chat_1",
        message_id="100",
        reply_to_message_id=None,
    )
    await handler.tree_queue.create_tree(
        node_id="100",
        incoming=root_1,
        status_message_id="101",
    )

    root_2 = incoming_message_factory(
        text="other chat",
        chat_id="chat_2",
        message_id="200",
        reply_to_message_id=None,
    )
    await handler.tree_queue.create_tree(
        node_id="200",
        incoming=root_2,
        status_message_id="201",
    )

    events = []

    async def _stop():
        events.append("stop")
        return 0

    async def _del(chat_id, message_id, fire_and_forget=True):
        events.append(f"del:{chat_id}:{message_id}:{fire_and_forget}")

    handler.stop_all_tasks = AsyncMock(side_effect=_stop)
    mock_platform.queue_delete_message = AsyncMock(side_effect=_del)

    incoming = incoming_message_factory(
        text="/clear", chat_id="chat_1", message_id="150"
    )
    await handler.handle_message(incoming)

    assert events and events[0] == "stop"
    deleted_ids = {e.split(":")[2] for e in events[1:]}
    assert deleted_ids == {"100", "101", "150"}
    assert all(e.endswith(":False") for e in events[1:])

    mock_session_store.clear_all.assert_called_once()
    assert handler.tree_queue.get_tree_count() == 0
    mock_platform.queue_send_message.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_clear_command_with_mention(
    handler, mock_platform, mock_session_store, incoming_message_factory
):
    handler.stop_all_tasks = AsyncMock(return_value=0)

    incoming = incoming_message_factory(
        text="/clear@MyBot", chat_id="chat_1", message_id="10"
    )
    await handler.handle_message(incoming)

    handler.stop_all_tasks.assert_called_once()
    mock_platform.queue_delete_message.assert_called_once_with(
        "chat_1",
        "10",
        fire_and_forget=False,
    )
    mock_session_store.clear_all.assert_called_once()


@pytest.mark.asyncio
async def test_handle_message_clear_command_deletes_message_log_ids(
    handler, mock_platform, mock_session_store, incoming_message_factory
):
    handler.stop_all_tasks = AsyncMock(return_value=0)
    mock_session_store.get_message_ids_for_chat.return_value = ["42", "43"]

    incoming = incoming_message_factory(
        text="/clear", chat_id="chat_1", message_id="150"
    )
    await handler.handle_message(incoming)

    deleted = {c.args[1] for c in mock_platform.queue_delete_message.call_args_list}
    assert deleted == {"42", "43", "150"}


@pytest.mark.asyncio
async def test_handle_message_clear_command_reply_clears_branch(
    handler, mock_platform, mock_session_store, incoming_message_factory
):
    """Reply /clear to a message clears only that branch."""
    root_incoming = incoming_message_factory(
        text="root", chat_id="chat_1", message_id="100", reply_to_message_id=None
    )
    tree = await handler.tree_queue.create_tree(
        node_id="100", incoming=root_incoming, status_message_id="101"
    )
    handler.tree_queue.register_node("101", tree.root_id)

    child_incoming = incoming_message_factory(
        text="child",
        chat_id="chat_1",
        message_id="102",
        reply_to_message_id="100",
    )
    await handler.tree_queue.add_to_tree(
        parent_node_id="100",
        node_id="102",
        incoming=child_incoming,
        status_message_id="103",
    )

    deleted_ids = []

    async def _capture_delete(chat_id, message_id, fire_and_forget=True):
        deleted_ids.append(message_id)

    mock_platform.queue_delete_message = AsyncMock(side_effect=_capture_delete)

    incoming = incoming_message_factory(
        text="/clear",
        chat_id="chat_1",
        message_id="150",
        reply_to_message_id="102",
    )
    await handler.handle_message(incoming)

    assert set(deleted_ids) == {"102", "103", "150"}
    assert "100" not in deleted_ids
    assert "101" not in deleted_ids
    mock_session_store.save_tree_snapshot.assert_called()
    assert handler.tree_queue.get_tree_for_node("102") is None
    assert handler.tree_queue.get_tree_for_node("100") is not None


@pytest.mark.asyncio
async def test_handle_message_clear_command_reply_unknown_sends_nothing(
    handler, mock_platform, mock_session_store, incoming_message_factory
):
    """Reply /clear to unknown message sends 'Nothing to clear'."""
    incoming = incoming_message_factory(
        text="/clear",
        chat_id="chat_1",
        message_id="150",
        reply_to_message_id="999",
    )
    await handler.handle_message(incoming)

    mock_platform.queue_send_message.assert_called_once()
    call_args = mock_platform.queue_send_message.call_args[0]
    assert "Nothing to clear" in call_args[1]
    mock_session_store.clear_all.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_clear_command_reply_to_root_clears_tree(
    handler, mock_platform, mock_session_store, incoming_message_factory
):
    """Reply /clear to root message clears entire tree."""
    root_incoming = incoming_message_factory(
        text="root", chat_id="chat_1", message_id="100", reply_to_message_id=None
    )
    await handler.tree_queue.create_tree(
        node_id="100", incoming=root_incoming, status_message_id="101"
    )

    deleted_ids = []

    async def _capture_delete(chat_id, message_id, fire_and_forget=True):
        deleted_ids.append(message_id)

    mock_platform.queue_delete_message = AsyncMock(side_effect=_capture_delete)

    incoming = incoming_message_factory(
        text="/clear",
        chat_id="chat_1",
        message_id="150",
        reply_to_message_id="100",
    )
    await handler.handle_message(incoming)

    assert set(deleted_ids) == {"100", "101", "150"}
    mock_session_store.remove_tree_snapshot.assert_called_once_with("100")
    assert handler.tree_queue.get_tree_count() == 0


@pytest.mark.asyncio
async def test_handle_message_clear_command_reply_pending_voice_cancels(
    handler, mock_platform, mock_session_store, incoming_message_factory
):
    """Reply /clear to a voice note during transcription cancels it."""

    async def cancel_pending(chat_id, reply_id):
        if reply_id == "100":
            return ("100", "101")
        return None

    mock_platform.cancel_pending_voice = AsyncMock(side_effect=cancel_pending)
    mock_platform.queue_delete_message = AsyncMock()
    deleted_ids = []

    async def _capture_delete(chat_id, message_id, fire_and_forget=True):
        deleted_ids.append(message_id)

    mock_platform.queue_delete_message = AsyncMock(side_effect=_capture_delete)

    incoming = incoming_message_factory(
        text="/clear",
        chat_id="chat_1",
        message_id="150",
        reply_to_message_id="100",
    )
    await handler.handle_message(incoming)

    mock_platform.cancel_pending_voice.assert_called_once_with("chat_1", "100")
    assert set(deleted_ids) == {"100", "101", "150"}
    call_args = mock_platform.queue_send_message.call_args[0]
    assert "Voice note cancelled" in call_args[1]
