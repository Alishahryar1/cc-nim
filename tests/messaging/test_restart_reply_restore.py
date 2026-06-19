from unittest.mock import AsyncMock, patch

import pytest

from messaging.models import IncomingMessage
from messaging.session import SessionStore
from messaging.trees import TreeQueueManager
from messaging.workflow import MessagingWorkflow


@pytest.mark.asyncio
async def test_reply_to_old_status_message_after_restore_routes_to_parent(
    tmp_path, mock_platform, mock_cli_manager
):
    # Build a persisted tree with a root node A and a bot status message id.
    store_path = tmp_path / "sessions.json"
    store = SessionStore(storage_path=str(store_path))

    handler1 = MessagingWorkflow(mock_platform, mock_cli_manager, store)
    a_incoming = IncomingMessage(
        text="A",
        chat_id="chat_1",
        user_id="user_1",
        message_id="A",
        platform="telegram",
    )
    tree = await handler1.tree_queue.create_tree(
        "A", a_incoming, status_message_id="status_A"
    )
    handler1.tree_queue.register_node("status_A", tree.root_id)
    store.register_node("status_A", tree.root_id)
    store.save_tree(tree.root_id, tree.to_dict())
    store.flush_pending_save()

    # "Restart": new store instance loads from disk, and we restore TreeQueueManager.
    store2 = SessionStore(storage_path=str(store_path))
    handler2 = MessagingWorkflow(mock_platform, mock_cli_manager, store2)
    handler2.replace_tree_queue(
        TreeQueueManager.from_dict(
            {
                "trees": store2.get_all_trees(),
                "node_to_tree": store2.get_node_mapping(),
            },
            queue_update_callback=handler2.update_queue_positions,
            node_started_callback=handler2.mark_node_processing,
        )
    )

    # Prevent background task scheduling; we only want to validate routing/tree mutation.
    mock_platform.queue_send_message = AsyncMock(return_value="status_reply")

    reply = IncomingMessage(
        text="R1",
        chat_id="chat_1",
        user_id="user_1",
        message_id="R1",
        platform="telegram",
        reply_to_message_id="status_A",
    )

    with patch.object(handler2.tree_queue, "enqueue", AsyncMock(return_value=False)):
        await handler2.handle_message(reply)

    restored_tree = handler2.tree_queue.get_tree_for_node("A")
    assert restored_tree is not None
    node_r1 = restored_tree.get_node("R1")
    assert node_r1 is not None
    assert node_r1.parent_id == "A"


@pytest.mark.asyncio
async def test_save_tree_persists_status_message_mapping_without_manual_register(
    tmp_path, mock_platform, mock_cli_manager
):
    store_path = tmp_path / "sessions.json"
    store = SessionStore(storage_path=str(store_path))

    handler1 = MessagingWorkflow(mock_platform, mock_cli_manager, store)
    a_incoming = IncomingMessage(
        text="A",
        chat_id="chat_1",
        user_id="user_1",
        message_id="A",
        platform="telegram",
    )
    tree = await handler1.tree_queue.create_tree(
        "A", a_incoming, status_message_id="status_A"
    )
    store.save_tree(tree.root_id, tree.to_dict())
    store.flush_pending_save()

    store2 = SessionStore(storage_path=str(store_path))
    handler2 = MessagingWorkflow(mock_platform, mock_cli_manager, store2)
    handler2.replace_tree_queue(
        TreeQueueManager.from_dict(
            {
                "trees": store2.get_all_trees(),
                "node_to_tree": store2.get_node_mapping(),
            },
            queue_update_callback=handler2.update_queue_positions,
            node_started_callback=handler2.mark_node_processing,
        )
    )
    mock_platform.queue_send_message = AsyncMock(return_value="status_reply")

    reply = IncomingMessage(
        text="R1",
        chat_id="chat_1",
        user_id="user_1",
        message_id="R1",
        platform="telegram",
        reply_to_message_id="status_A",
    )

    with patch.object(handler2.tree_queue, "enqueue", AsyncMock(return_value=False)):
        await handler2.handle_message(reply)

    restored_tree = handler2.tree_queue.get_tree_for_node("A")
    assert restored_tree is not None
    node_r1 = restored_tree.get_node("R1")
    assert node_r1 is not None
    assert node_r1.parent_id == "A"


@pytest.mark.asyncio
async def test_reply_clear_purges_removed_status_mapping_from_persisted_store(
    tmp_path, mock_platform, mock_cli_manager
):
    store_path = tmp_path / "sessions.json"
    store = SessionStore(storage_path=str(store_path))
    handler = MessagingWorkflow(mock_platform, mock_cli_manager, store)

    root_incoming = IncomingMessage(
        text="root",
        chat_id="chat_1",
        user_id="user_1",
        message_id="root",
        platform="telegram",
    )
    tree = await handler.tree_queue.create_tree(
        "root", root_incoming, status_message_id="root_status"
    )
    handler.tree_queue.register_node("root_status", tree.root_id)

    child_incoming = IncomingMessage(
        text="child",
        chat_id="chat_1",
        user_id="user_1",
        message_id="child",
        platform="telegram",
        reply_to_message_id="root",
    )
    await handler.tree_queue.add_to_tree(
        "root", "child", child_incoming, status_message_id="child_status"
    )
    handler.tree_queue.register_node("child_status", tree.root_id)
    store.save_tree(tree.root_id, tree.to_dict())

    clear_reply = IncomingMessage(
        text="/clear",
        chat_id="chat_1",
        user_id="user_1",
        message_id="clear_command",
        platform="telegram",
        reply_to_message_id="child",
    )

    await handler.handle_message(clear_reply)
    store.flush_pending_save()

    restored_store = SessionStore(storage_path=str(store_path))
    restored_tree_queue = TreeQueueManager.from_dict(
        {
            "trees": restored_store.get_all_trees(),
            "node_to_tree": restored_store.get_node_mapping(),
        }
    )

    assert restored_tree_queue.get_tree_for_node("root") is not None
    assert restored_tree_queue.get_tree_for_node("root_status") is not None
    assert restored_tree_queue.get_tree_for_node("child") is None
    assert restored_tree_queue.get_tree_for_node("child_status") is None
