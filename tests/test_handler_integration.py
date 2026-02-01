import pytest
import asyncio
from unittest.mock import MagicMock
from messaging.handler import ClaudeMessageHandler
from messaging.tree_core import MessageState


@pytest.fixture
def handler_integration(mock_platform, mock_cli_manager, mock_session_store):
    # Use real TreeQueueManager
    handler = ClaudeMessageHandler(mock_platform, mock_cli_manager, mock_session_store)
    return handler


async def mock_async_gen(events):
    for e in events:
        yield e


@pytest.mark.asyncio
async def test_full_conversation_flow_single_user(
    handler_integration, mock_platform, mock_cli_manager, incoming_message_factory
):
    # 1. First message
    msg1 = incoming_message_factory(text="message 1", message_id="m1")
    mock_platform.queue_send_message.return_value = "s1"

    # Mock CLI session for m1
    mock_session1 = MagicMock()
    mock_session1.start_task.return_value = mock_async_gen(
        [
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Reply 1"}]},
            },
            {"type": "exit", "code": 0},
        ]
    )
    mock_cli_manager.get_or_create_session.return_value = (
        mock_session1,
        "sess1",
        False,
    )

    await handler_integration.handle_message(msg1)

    # Wait for processing
    tree = handler_integration.tree_queue.get_tree_for_node("m1")
    for _ in range(10):
        if tree.get_node("m1").state.value == MessageState.COMPLETED.value:
            break
        await asyncio.sleep(0.01)

    assert tree.get_node("m1").state.value == MessageState.COMPLETED.value
    assert tree.get_node("m1").session_id == "sess1"

    # 2. Reply to m1
    msg2 = incoming_message_factory(
        text="message 2", message_id="m2", reply_to_message_id="m1"
    )
    mock_platform.queue_send_message.return_value = "s2"

    # Mock CLI session for m2
    mock_session2 = MagicMock()
    mock_session2.start_task.return_value = mock_async_gen(
        [
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Reply 2"}]},
            },
            {"type": "exit", "code": 0},
        ]
    )
    mock_cli_manager.get_or_create_session.reset_mock()
    mock_cli_manager.get_or_create_session.return_value = (
        mock_session2,
        "sess2",
        False,
    )

    await handler_integration.handle_message(msg2)

    # Wait for processing
    for _ in range(10):
        if tree.get_node("m2").state.value == MessageState.COMPLETED.value:
            break
        await asyncio.sleep(0.01)

    assert tree.get_node("m2").state.value == MessageState.COMPLETED.value
    assert tree.get_node("m2").parent_id == "m1"
    mock_cli_manager.get_or_create_session.assert_called_with(session_id="sess1")


@pytest.mark.asyncio
async def test_error_propagation_chain(
    handler_integration, mock_platform, mock_cli_manager, incoming_message_factory
):
    msg1 = incoming_message_factory(text="m1", message_id="m1")
    mock_platform.queue_send_message.return_value = "s1"

    mock_session1 = MagicMock()
    mock_session1.start_task.return_value = mock_async_gen(
        [{"type": "error", "error": {"message": "failed"}}]
    )
    mock_cli_manager.get_or_create_session.return_value = (
        mock_session1,
        "sess1",
        False,
    )

    await handler_integration.handle_message(msg1)
    tree = handler_integration.tree_queue.get_tree_for_node("m1")

    msg2 = incoming_message_factory(
        text="m2", message_id="m2", reply_to_message_id="m1"
    )
    await handler_integration.handle_message(msg2)

    # Wait for m1 to fail
    for _ in range(20):
        if tree.get_node("m1").state.value == MessageState.ERROR.value:
            break
        await asyncio.sleep(0.01)

    # Give a tiny bit of time for propagation and skipping in processor
    await asyncio.sleep(0.05)

    assert tree.get_node("m1").state.value == MessageState.ERROR.value
    assert tree.get_node("m2").state.value == MessageState.ERROR.value
    assert "Parent failed" in tree.get_node("m2").error_message


@pytest.mark.asyncio
async def test_concurrent_replies_to_different_trees(
    handler_integration, mock_platform, mock_cli_manager, incoming_message_factory
):
    msg1 = incoming_message_factory(text="t1", message_id="t1")
    msg2 = incoming_message_factory(text="t2", message_id="t2")

    mock_session1 = MagicMock()
    mock_session1.start_task.return_value = mock_async_gen(
        [{"type": "exit", "code": 0}]
    )
    mock_session2 = MagicMock()
    mock_session2.start_task.return_value = mock_async_gen(
        [{"type": "exit", "code": 0}]
    )

    mock_cli_manager.get_or_create_session.side_effect = [
        (mock_session1, "s1", False),
        (mock_session2, "s2", False),
    ]

    await handler_integration.handle_message(msg1)
    await handler_integration.handle_message(msg2)

    # Wait for both
    for _ in range(20):
        node1 = handler_integration.tree_queue.get_node("t1")
        node2 = handler_integration.tree_queue.get_node("t2")
        if (
            node1
            and node2
            and node1.state.value == MessageState.COMPLETED.value
            and node2.state.value == MessageState.COMPLETED.value
        ):
            break
        await asyncio.sleep(0.01)

    assert (
        handler_integration.tree_queue.get_node("t1").state.value
        == MessageState.COMPLETED.value
    )
    assert (
        handler_integration.tree_queue.get_node("t2").state.value
        == MessageState.COMPLETED.value
    )
