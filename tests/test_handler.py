import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from messaging.tree_data import MessageTree, MessageNode
from messaging.models import IncomingMessage
from messaging.handler import ClaudeMessageHandler
from messaging.tree_queue import MessageState


@pytest.fixture
def handler(mock_platform, mock_cli_manager, mock_session_store):
    return ClaudeMessageHandler(mock_platform, mock_cli_manager, mock_session_store)


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
    )


@pytest.mark.asyncio
async def test_handle_message_stats_command(
    handler, mock_platform, mock_cli_manager, incoming_message_factory
):
    incoming = incoming_message_factory(text="/stats")
    mock_cli_manager.get_stats.return_value = {"active_sessions": 2, "max_sessions": 5}

    await handler.handle_message(incoming)

    mock_platform.queue_send_message.assert_called_once()
    args, _ = mock_platform.queue_send_message.call_args
    assert "Active CLI: 2" in args[1]
    assert "Max CLI: 5" in args[1]


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
        mock_tree.to_dict.return_value = {"data": "tree"}
        mock_create.return_value = mock_tree

        await handler.handle_message(incoming)

        mock_create.assert_called_once()
        mock_enqueue.assert_called_once()
        mock_session_store.save_tree.assert_called_once_with("root_1", {"data": "tree"})


@pytest.mark.asyncio
async def test_handle_message_queued(handler, mock_platform, incoming_message_factory):
    incoming = incoming_message_factory(text="hello", message_id="msg_1")
    mock_platform.queue_send_message.return_value = "status_123"

    with (
        patch.object(handler.tree_queue, "create_tree", AsyncMock()) as mock_create,
        patch.object(
            handler.tree_queue, "enqueue", AsyncMock(return_value=True)
        ) as mock_enqueue,
        patch.object(handler.tree_queue, "get_queue_size", MagicMock(return_value=3)),
    ):
        mock_tree = MagicMock()
        mock_tree.root_id = "root_1"
        mock_tree.to_dict.return_value = {}
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

    await handler._update_queue_positions(tree)

    calls = mock_platform.queue_edit_message.call_args_list
    assert len(calls) == 2
    assert calls[0][0][0] == "chat_1"
    assert calls[0][0][1] == "status_1"
    assert "position 1" in calls[0][0][2]
    assert calls[1][0][0] == "chat_1"
    assert calls[1][0][1] == "status_2"
    assert "position 2" in calls[1][0][2]


@pytest.mark.asyncio
async def test_stop_all_tasks(handler, mock_cli_manager, mock_platform):
    mock_node = MagicMock()
    mock_node.incoming.chat_id = "chat_1"
    mock_node.status_message_id = "status_1"

    with patch.object(
        handler.tree_queue, "cancel_all_sync", MagicMock(return_value=[mock_node])
    ):
        count = await handler.stop_all_tasks()

        assert count == 1
        mock_cli_manager.stop_all.assert_called_once()
        mock_platform.fire_and_forget.assert_called_once()


async def mock_async_gen(events):
    for e in events:
        yield e


@pytest.mark.asyncio
async def test_process_node_success_flow(handler, mock_cli_manager, mock_platform):
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
    mock_tree.to_dict.return_value = {}

    with patch.object(
        handler.tree_queue, "get_tree_for_node", MagicMock(return_value=mock_tree)
    ):
        await handler._process_node(node_id, mock_node)

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


@pytest.mark.asyncio
async def test_process_node_error_flow(handler, mock_cli_manager, mock_platform):
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
    mock_tree.update_state = AsyncMock()

    with (
        patch.object(
            handler.tree_queue, "get_tree_for_node", MagicMock(return_value=mock_tree)
        ),
        patch.object(
            handler.tree_queue, "mark_node_error", AsyncMock(return_value=[mock_node])
        ),
    ):
        await handler._process_node(node_id, mock_node)

        handler.tree_queue.mark_node_error.assert_called_once_with(
            node_id, "CLI crashed", propagate_to_children=True
        )

        last_call = mock_platform.queue_edit_message.call_args_list[-1]
        assert "❌ *Error*" in last_call[0][2]
        assert "CLI crashed" in last_call[0][2]
