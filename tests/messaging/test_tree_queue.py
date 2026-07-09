"""Tests for tree-based message queue system."""

import asyncio
import contextlib
from unittest.mock import AsyncMock

import pytest

from free_claude_code.messaging.models import IncomingMessage
from free_claude_code.messaging.trees import (
    CancellationReason,
    CancellationUiOwner,
    MessageNode,
    MessageState,
    MessageTree,
    TreeQueueManager,
    TreeSnapshot,
    get_cancel_reason,
    set_cancel_reason,
)
from free_claude_code.messaging.trees import manager as tree_manager_module
from free_claude_code.messaging.trees.graph import MessageTreeGraph
from free_claude_code.messaging.trees.snapshot import (
    node_from_snapshot,
    node_to_snapshot,
)


class TestMessageState:
    """Test MessageState enum."""

    def test_state_values(self):
        """Test state enum values."""
        assert MessageState.PENDING.value == "pending"
        assert MessageState.IN_PROGRESS.value == "in_progress"
        assert MessageState.COMPLETED.value == "completed"
        assert MessageState.ERROR.value == "error"


class TestMessageNode:
    """Test MessageNode dataclass."""

    def test_node_creation(self):
        """Test creating a message node."""
        incoming = IncomingMessage(
            text="Hello",
            chat_id="123",
            user_id="456",
            message_id="789",
            platform="telegram",
        )
        node = MessageNode(
            node_id="789",
            incoming=incoming,
            status_message_id="status_1",
        )

        assert node.node_id == "789"
        assert node.state == MessageState.PENDING
        assert node.parent_id is None
        assert node.children_ids == []
        assert node.session_id is None

    def test_node_to_snapshot(self):
        """Test serializing a node."""
        incoming = IncomingMessage(
            text="Test",
            chat_id="1",
            user_id="2",
            message_id="3",
            platform="test",
        )
        node = MessageNode(
            node_id="3",
            incoming=incoming,
            status_message_id="s1",
            state=MessageState.COMPLETED,
            session_id="sess_123",
        )

        data = node_to_snapshot(node)
        assert data["node_id"] == "3"
        assert data["state"] == "completed"
        assert data["session_id"] == "sess_123"

    def test_node_from_snapshot(self):
        """Test deserializing a node."""
        data = {
            "node_id": "n1",
            "incoming": {
                "text": "Hello",
                "chat_id": "c1",
                "user_id": "u1",
                "message_id": "m1",
                "platform": "test",
            },
            "status_message_id": "s1",
            "state": "in_progress",
            "parent_id": "parent_1",
            "session_id": None,
            "children_ids": ["child_1"],
            "created_at": "2025-01-01T00:00:00",
        }

        node = node_from_snapshot(data)
        assert node.node_id == "n1"
        assert node.state == MessageState.IN_PROGRESS
        assert node.parent_id == "parent_1"
        assert "child_1" in node.children_ids

    def test_completed_state_clears_stale_error_message(self):
        """Successful completion is terminal and should clear prior node errors."""
        incoming = IncomingMessage(
            text="Test",
            chat_id="1",
            user_id="2",
            message_id="3",
            platform="test",
        )
        node = MessageNode(node_id="3", incoming=incoming, status_message_id="s1")
        node.mark_error("temporary failure")

        node.update_state(MessageState.COMPLETED, session_id="session_1")

        assert node.state is MessageState.COMPLETED
        assert node.session_id == "session_1"
        assert node.error_message is None

    def test_cancel_reason_helpers_preserve_existing_context(self):
        """Cancellation reason is typed and does not clobber other context."""
        incoming = IncomingMessage(
            text="Test",
            chat_id="1",
            user_id="2",
            message_id="3",
            platform="test",
        )
        node = MessageNode(node_id="3", incoming=incoming, status_message_id="s1")
        node.set_context({"other": "value"})

        set_cancel_reason(node, CancellationReason.STOP)

        assert get_cancel_reason(node) is CancellationReason.STOP
        assert node.context == {"other": "value", "cancel_reason": "stop"}

        set_cancel_reason(node, None)

        assert get_cancel_reason(node) is None
        assert node.context == {"other": "value"}


class TestMessageTree:
    """Test MessageTree class."""

    def test_tree_creation(self):
        """Test creating a tree with root node."""
        incoming = IncomingMessage(
            text="Root",
            chat_id="1",
            user_id="1",
            message_id="root_msg",
            platform="test",
        )
        root = MessageNode(
            node_id="root_msg",
            incoming=incoming,
            status_message_id="status_1",
        )

        tree = MessageTree(root)
        assert tree.root_id == "root_msg"
        assert tree.get_node("root_msg") is not None
        assert tree.is_processing is False

    @pytest.mark.asyncio
    async def test_add_child_node(self):
        """Test adding a child node to the tree."""
        # Create root
        root_incoming = IncomingMessage(
            text="Root",
            chat_id="1",
            user_id="1",
            message_id="root",
            platform="test",
        )
        root = MessageNode(
            node_id="root",
            incoming=root_incoming,
            status_message_id="s1",
        )
        tree = MessageTree(root)

        # Add child
        child_incoming = IncomingMessage(
            text="Child",
            chat_id="1",
            user_id="1",
            message_id="child",
            platform="test",
            reply_to_message_id="root",
        )
        child = await tree.add_node(
            node_id="child",
            incoming=child_incoming,
            status_message_id="s2",
            parent_id="root",
        )

        assert child.node_id == "child"
        assert child.parent_id == "root"
        assert "child" in tree.get_root().children_ids
        parent = tree.get_parent("child")
        assert parent is not None
        assert parent.node_id == "root"

    @pytest.mark.asyncio
    async def test_update_state(self):
        """Test updating node state."""
        incoming = IncomingMessage(
            text="Test",
            chat_id="1",
            user_id="1",
            message_id="m1",
            platform="test",
        )
        root = MessageNode(node_id="m1", incoming=incoming, status_message_id="s1")
        tree = MessageTree(root)

        await tree.update_state("m1", MessageState.IN_PROGRESS)
        node = tree.get_node("m1")
        assert node is not None
        assert node.state == MessageState.IN_PROGRESS

        await tree.update_state("m1", MessageState.COMPLETED, session_id="sess_abc")
        node = tree.get_node("m1")
        assert node is not None
        assert node.state == MessageState.COMPLETED
        assert node.session_id == "sess_abc"
        assert node.completed_at is not None

    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self):
        """Test queue operations."""
        incoming = IncomingMessage(
            text="Test",
            chat_id="1",
            user_id="1",
            message_id="m1",
            platform="test",
        )
        root = MessageNode(node_id="m1", incoming=incoming, status_message_id="s1")
        tree = MessageTree(root)

        # Enqueue
        pos = await tree.enqueue("m1")
        assert pos == 1
        assert tree.get_queue_size() == 1

        # Dequeue
        node_id = await tree.dequeue()
        assert node_id == "m1"
        assert tree.get_queue_size() == 0

    @pytest.mark.asyncio
    async def test_queue_snapshot(self):
        """Test queue snapshot order."""
        incoming = IncomingMessage(
            text="Root",
            chat_id="1",
            user_id="1",
            message_id="root",
            platform="test",
        )
        root = MessageNode(node_id="root", incoming=incoming, status_message_id="s1")
        tree = MessageTree(root)

        child_incoming_1 = IncomingMessage(
            text="Child 1",
            chat_id="1",
            user_id="1",
            message_id="child_1",
            platform="test",
            reply_to_message_id="root",
        )
        child_incoming_2 = IncomingMessage(
            text="Child 2",
            chat_id="1",
            user_id="1",
            message_id="child_2",
            platform="test",
            reply_to_message_id="root",
        )

        await tree.add_node(
            node_id="child_1",
            incoming=child_incoming_1,
            status_message_id="s2",
            parent_id="root",
        )
        await tree.add_node(
            node_id="child_2",
            incoming=child_incoming_2,
            status_message_id="s3",
            parent_id="root",
        )

        await tree.enqueue("child_1")
        await tree.enqueue("child_2")

        snapshot = await tree.get_queue_snapshot()
        assert snapshot == ["child_1", "child_2"]

    def test_tree_snapshot_round_trip(self):
        """Test tree snapshot round-trip."""
        incoming = IncomingMessage(
            text="Test",
            chat_id="1",
            user_id="1",
            message_id="m1",
            platform="test",
        )
        root = MessageNode(
            node_id="m1",
            incoming=incoming,
            status_message_id="s1",
            state=MessageState.COMPLETED,
            session_id="sess_1",
        )
        tree = MessageTree(root)

        snapshot = tree.snapshot()
        restored = MessageTree.from_snapshot(snapshot)

        assert restored.root_id == "m1"
        node = restored.get_node("m1")
        assert node is not None
        assert node.session_id == "sess_1"

    def test_tree_from_snapshot_uses_one_graph_construction(self, monkeypatch):
        """Restore should not build a temporary graph and replace it."""
        incoming = IncomingMessage(
            text="Test",
            chat_id="1",
            user_id="1",
            message_id="m1",
            platform="test",
        )
        root = MessageNode(
            node_id="m1",
            incoming=incoming,
            status_message_id="s1",
        )
        snapshot = MessageTree(root).snapshot()
        original_init = MessageTreeGraph.__init__
        init_calls = 0

        def counting_init(self: MessageTreeGraph, root_node: MessageNode) -> None:
            nonlocal init_calls
            init_calls += 1
            original_init(self, root_node)

        monkeypatch.setattr(MessageTreeGraph, "__init__", counting_init)

        restored = MessageTree.from_snapshot(snapshot)

        assert restored.root_id == "m1"
        assert init_calls == 1

    @pytest.mark.asyncio
    async def test_get_descendants(self):
        """Test get_descendants returns node and all descendants."""
        root_incoming = IncomingMessage(
            text="Root", chat_id="1", user_id="1", message_id="root", platform="test"
        )
        root = MessageNode(
            node_id="root", incoming=root_incoming, status_message_id="s1"
        )
        tree = MessageTree(root)

        child_incoming = IncomingMessage(
            text="Child",
            chat_id="1",
            user_id="1",
            message_id="child",
            platform="test",
            reply_to_message_id="root",
        )
        await tree.add_node("child", child_incoming, "s2", "root")

        grandchild_incoming = IncomingMessage(
            text="Grand",
            chat_id="1",
            user_id="1",
            message_id="grand",
            platform="test",
            reply_to_message_id="child",
        )
        await tree.add_node("grand", grandchild_incoming, "s3", "child")

        assert tree.get_descendants("root") == ["root", "child", "grand"]
        assert tree.get_descendants("child") == ["child", "grand"]
        assert tree.get_descendants("grand") == ["grand"]
        assert tree.get_descendants("nonexistent") == []

    @pytest.mark.asyncio
    async def test_remove_branch(self):
        """Test remove_branch removes subtree and updates parent."""
        root_incoming = IncomingMessage(
            text="Root", chat_id="1", user_id="1", message_id="root", platform="test"
        )
        root = MessageNode(
            node_id="root", incoming=root_incoming, status_message_id="s1"
        )
        tree = MessageTree(root)

        child_incoming = IncomingMessage(
            text="Child",
            chat_id="1",
            user_id="1",
            message_id="child",
            platform="test",
            reply_to_message_id="root",
        )
        await tree.add_node("child", child_incoming, "s2", "root")

        grandchild_incoming = IncomingMessage(
            text="Grand",
            chat_id="1",
            user_id="1",
            message_id="grand",
            platform="test",
            reply_to_message_id="child",
        )
        await tree.add_node("grand", grandchild_incoming, "s3", "child")

        async with tree.with_lock():
            removed = tree.remove_branch("child")

        assert len(removed) == 2
        assert {n.node_id for n in removed} == {"child", "grand"}
        assert tree.get_node("child") is None
        assert tree.get_node("grand") is None
        assert tree.get_node("root") is not None
        assert "child" not in tree.get_root().children_ids


class TestTreeQueueManager:
    """Test TreeQueueManager class."""

    @pytest.mark.asyncio
    async def test_create_tree(self):
        """Test creating a new tree."""
        manager = TreeQueueManager()

        incoming = IncomingMessage(
            text="New message",
            chat_id="1",
            user_id="1",
            message_id="msg_1",
            platform="test",
        )

        tree = await manager.create_tree(
            node_id="msg_1",
            incoming=incoming,
            status_message_id="status_1",
        )

        assert tree is not None
        assert tree.root_id == "msg_1"
        assert manager.get_tree("msg_1") is tree

    @pytest.mark.asyncio
    async def test_add_reply_to_tree(self):
        """Test adding a reply to existing tree."""
        manager = TreeQueueManager()

        # Create root
        root_incoming = IncomingMessage(
            text="Root",
            chat_id="1",
            user_id="1",
            message_id="root",
            platform="test",
        )
        await manager.create_tree("root", root_incoming, "s1")

        # Add reply
        reply_incoming = IncomingMessage(
            text="Reply",
            chat_id="1",
            user_id="1",
            message_id="reply",
            platform="test",
            reply_to_message_id="root",
        )
        tree, node = await manager.add_to_tree(
            parent_node_id="root",
            node_id="reply",
            incoming=reply_incoming,
            status_message_id="s2",
        )

        assert node.parent_id == "root"
        assert manager.get_tree_for_node("reply") is tree

    @pytest.mark.asyncio
    async def test_enqueue_and_process(self):
        """Test enqueueing and processing."""
        manager = TreeQueueManager()
        processed = []

        async def processor(node_id, node):
            processed.append(node_id)
            await asyncio.sleep(0.01)  # Simulate work

        incoming = IncomingMessage(
            text="Test",
            chat_id="1",
            user_id="1",
            message_id="m1",
            platform="test",
        )
        await manager.create_tree("m1", incoming, "s1")

        was_queued = await manager.enqueue("m1", processor)
        assert was_queued is False  # First message processes immediately

        # Wait for processing
        await asyncio.sleep(0.1)
        assert "m1" in processed

    @pytest.mark.asyncio
    async def test_queue_when_busy(self):
        """Test that messages queue when tree is busy."""
        manager = TreeQueueManager()
        processing_started = asyncio.Event()
        processing_complete = asyncio.Event()

        async def slow_processor(node_id, node):
            processing_started.set()
            await processing_complete.wait()

        # Create tree with root
        root_incoming = IncomingMessage(
            text="Root",
            chat_id="1",
            user_id="1",
            message_id="root",
            platform="test",
        )
        await manager.create_tree("root", root_incoming, "s1")

        # Start processing root
        was_queued = await manager.enqueue("root", slow_processor)
        assert was_queued is False

        # Wait for processing to start
        await processing_started.wait()

        # Add a child
        child_incoming = IncomingMessage(
            text="Child",
            chat_id="1",
            user_id="1",
            message_id="child",
            platform="test",
            reply_to_message_id="root",
        )
        await manager.add_to_tree("root", "child", child_incoming, "s2")

        # Try to enqueue child - should be queued since tree is busy
        was_queued = await manager.enqueue("child", slow_processor)
        assert was_queued is True
        assert manager.get_queue_size("child") == 1

        # Cleanup
        processing_complete.set()

    @pytest.mark.asyncio
    async def test_cancel_tree(self):
        """Test cancelling a tree."""
        manager = TreeQueueManager()
        processing_complete = asyncio.Event()

        async def slow_processor(node_id, node):
            await processing_complete.wait()

        incoming = IncomingMessage(
            text="Test",
            chat_id="1",
            user_id="1",
            message_id="m1",
            platform="test",
        )
        await manager.create_tree("m1", incoming, "s1")
        await manager.enqueue("m1", slow_processor)

        # Cancel
        cancelled = await manager.cancel_tree("m1")
        assert len(cancelled) == 1

        processing_complete.set()

    @pytest.mark.asyncio
    async def test_cancel_tree_waits_for_current_task_cleanup(self):
        """cancel_tree returns only after the cancelled task cleanup runs."""
        manager = TreeQueueManager()
        started = asyncio.Event()
        cleanup_done = asyncio.Event()

        async def slow_processor(node_id, node):
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cleanup_done.set()
                raise

        incoming = IncomingMessage(
            text="Test",
            chat_id="1",
            user_id="1",
            message_id="m1",
            platform="test",
        )
        await manager.create_tree("m1", incoming, "s1")
        await manager.enqueue("m1", slow_processor)
        await started.wait()

        cancelled = await manager.cancel_tree("m1", reason=CancellationReason.STOP)

        assert [entry.node.node_id for entry in cancelled] == ["m1"]
        assert [entry.ui_owner for entry in cancelled] == [CancellationUiOwner.RUNNER]
        assert get_cancel_reason(cancelled[0].node) is CancellationReason.STOP
        assert cleanup_done.is_set()

    @pytest.mark.asyncio
    async def test_cancel_node_waits_for_current_task_cleanup(self):
        """cancel_node waits when the target node is actively running."""
        manager = TreeQueueManager()
        started = asyncio.Event()
        cleanup_done = asyncio.Event()

        async def slow_processor(node_id, node):
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cleanup_done.set()
                raise

        incoming = IncomingMessage(
            text="Test",
            chat_id="1",
            user_id="1",
            message_id="m1",
            platform="test",
        )
        await manager.create_tree("m1", incoming, "s1")
        await manager.enqueue("m1", slow_processor)
        await started.wait()

        cancelled = await manager.cancel_node("m1", reason=CancellationReason.STOP)

        assert [entry.node.node_id for entry in cancelled] == ["m1"]
        assert [entry.ui_owner for entry in cancelled] == [CancellationUiOwner.RUNNER]
        assert get_cancel_reason(cancelled[0].node) is CancellationReason.STOP
        assert cleanup_done.is_set()

    @pytest.mark.asyncio
    async def test_cancel_branch(self):
        """Test cancel_branch cancels only nodes in subtree."""
        manager = TreeQueueManager()

        root_incoming = IncomingMessage(
            text="Root", chat_id="1", user_id="1", message_id="root", platform="test"
        )
        await manager.create_tree("root", root_incoming, "s1")

        child_incoming = IncomingMessage(
            text="Child",
            chat_id="1",
            user_id="1",
            message_id="child",
            platform="test",
            reply_to_message_id="root",
        )
        tree, _ = await manager.add_to_tree("root", "child", child_incoming, "s2")

        sibling_incoming = IncomingMessage(
            text="Sibling",
            chat_id="1",
            user_id="1",
            message_id="sibling",
            platform="test",
            reply_to_message_id="root",
        )
        await manager.add_to_tree("root", "sibling", sibling_incoming, "s3")

        cancelled = await manager.cancel_branch("child")
        assert len(cancelled) == 1
        assert cancelled[0].node.node_id == "child"
        assert cancelled[0].ui_owner is CancellationUiOwner.WORKFLOW

        child_node = tree.get_node("child")
        assert child_node is not None
        assert child_node.state == MessageState.ERROR

        sibling_node = tree.get_node("sibling")
        assert sibling_node is not None
        assert sibling_node.state == MessageState.PENDING

    @pytest.mark.asyncio
    async def test_cancel_branch_waits_for_current_task_cleanup(self):
        """cancel_branch waits when the active node is inside the branch."""
        manager = TreeQueueManager()
        started = asyncio.Event()
        cleanup_done = asyncio.Event()

        async def slow_processor(node_id, node):
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cleanup_done.set()
                raise

        root_incoming = IncomingMessage(
            text="Root", chat_id="1", user_id="1", message_id="root", platform="test"
        )
        await manager.create_tree("root", root_incoming, "s1")
        child_incoming = IncomingMessage(
            text="Child",
            chat_id="1",
            user_id="1",
            message_id="child",
            platform="test",
            reply_to_message_id="root",
        )
        await manager.add_to_tree("root", "child", child_incoming, "s2")
        await manager.enqueue("child", slow_processor)
        await started.wait()

        cancelled = await manager.cancel_branch("child")

        assert [entry.node.node_id for entry in cancelled] == ["child"]
        assert [entry.ui_owner for entry in cancelled] == [CancellationUiOwner.RUNNER]
        assert cleanup_done.is_set()

    @pytest.mark.asyncio
    async def test_cancel_all_waits_for_current_task_cleanup_across_trees(self):
        """cancel_all waits for active task cleanup in every tree."""
        manager = TreeQueueManager()
        started: set[str] = set()
        cleanup_done: set[str] = set()
        all_started = asyncio.Event()

        async def slow_processor(node_id, node):
            started.add(node_id)
            if started == {"a", "b"}:
                all_started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cleanup_done.add(node_id)
                raise

        for node_id in ("a", "b"):
            incoming = IncomingMessage(
                text="Test",
                chat_id="1",
                user_id="1",
                message_id=node_id,
                platform="test",
            )
            await manager.create_tree(node_id, incoming, f"s_{node_id}")
            await manager.enqueue(node_id, slow_processor)
        await all_started.wait()

        cancelled = await manager.cancel_all(reason=CancellationReason.STOP)

        assert {entry.node.node_id for entry in cancelled} == {"a", "b"}
        assert {entry.ui_owner for entry in cancelled} == {CancellationUiOwner.RUNNER}
        assert {get_cancel_reason(entry.node) for entry in cancelled} == {
            CancellationReason.STOP
        }
        assert cleanup_done == {"a", "b"}

    @pytest.mark.asyncio
    async def test_cancel_task_drain_timeout_is_bounded(self, monkeypatch):
        """Cancellation drain returns when a task refuses to finish cleanup."""
        monkeypatch.setattr(tree_manager_module, "CANCEL_TASK_DRAIN_TIMEOUT_S", 0.01)
        started = asyncio.Event()
        cleanup_started = asyncio.Event()
        cleanup_release = asyncio.Event()

        async def stubborn_task():
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cleanup_started.set()
                await cleanup_release.wait()
                raise

        task = asyncio.create_task(stubborn_task())
        await started.wait()
        task.cancel()
        await cleanup_started.wait()

        start = asyncio.get_running_loop().time()
        try:
            await tree_manager_module._drain_cancelled_tasks([task])
            elapsed = asyncio.get_running_loop().time() - start
            assert not task.done()
        finally:
            cleanup_release.set()
            if not task.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        assert elapsed < 0.5

    @pytest.mark.asyncio
    async def test_cancel_node_refreshes_queue_positions_for_remaining_nodes(self):
        """Reply-scoped stop refreshes queued siblings after removing one queued node."""
        queue_updated = AsyncMock()
        manager = TreeQueueManager(queue_update_callback=queue_updated)

        root_incoming = IncomingMessage(
            text="Root", chat_id="1", user_id="1", message_id="root", platform="test"
        )
        tree = await manager.create_tree("root", root_incoming, "s1")

        first_incoming = IncomingMessage(
            text="First",
            chat_id="1",
            user_id="1",
            message_id="queued_first",
            platform="test",
            reply_to_message_id="root",
        )
        await manager.add_to_tree("root", "queued_first", first_incoming, "s2")

        second_incoming = IncomingMessage(
            text="Second",
            chat_id="1",
            user_id="1",
            message_id="queued_second",
            platform="test",
            reply_to_message_id="root",
        )
        await manager.add_to_tree("root", "queued_second", second_incoming, "s3")

        async with tree.with_lock():
            tree.set_processing_state("root", True)
            tree.put_queue_unlocked("queued_first")
            tree.put_queue_unlocked("queued_second")

        cancelled = await manager.cancel_node("queued_first")

        assert [entry.node.node_id for entry in cancelled] == ["queued_first"]
        assert [entry.ui_owner for entry in cancelled] == [CancellationUiOwner.WORKFLOW]
        assert await tree.get_queue_snapshot() == ["queued_second"]
        queue_updated.assert_awaited_once_with(tree)

    @pytest.mark.asyncio
    async def test_cancel_branch_refreshes_queue_positions_for_remaining_nodes(self):
        """Reply-scoped clear refreshes queued siblings after removing a queued branch."""
        queue_updated = AsyncMock()
        manager = TreeQueueManager(queue_update_callback=queue_updated)

        root_incoming = IncomingMessage(
            text="Root", chat_id="1", user_id="1", message_id="root", platform="test"
        )
        tree = await manager.create_tree("root", root_incoming, "s1")

        child_incoming = IncomingMessage(
            text="Child",
            chat_id="1",
            user_id="1",
            message_id="queued_first",
            platform="test",
            reply_to_message_id="root",
        )
        await manager.add_to_tree("root", "queued_first", child_incoming, "s2")

        sibling_incoming = IncomingMessage(
            text="Sibling",
            chat_id="1",
            user_id="1",
            message_id="queued_second",
            platform="test",
            reply_to_message_id="root",
        )
        await manager.add_to_tree("root", "queued_second", sibling_incoming, "s3")

        async with tree.with_lock():
            tree.set_processing_state("root", True)
            tree.put_queue_unlocked("queued_first")
            tree.put_queue_unlocked("queued_second")

        cancelled = await manager.cancel_branch("queued_first")

        assert [entry.node.node_id for entry in cancelled] == ["queued_first"]
        assert [entry.ui_owner for entry in cancelled] == [CancellationUiOwner.WORKFLOW]
        assert await tree.get_queue_snapshot() == ["queued_second"]
        queue_updated.assert_awaited_once_with(tree)

    @pytest.mark.asyncio
    async def test_remove_branch_non_root(self):
        """Test remove_branch removes only the subtree when branch is not root."""
        manager = TreeQueueManager()

        root_incoming = IncomingMessage(
            text="Root", chat_id="1", user_id="1", message_id="root", platform="test"
        )
        await manager.create_tree("root", root_incoming, "s1")

        child_incoming = IncomingMessage(
            text="Child",
            chat_id="1",
            user_id="1",
            message_id="child",
            platform="test",
            reply_to_message_id="root",
        )
        tree, _ = await manager.add_to_tree("root", "child", child_incoming, "s2")

        removed, root_id, removed_entire = await manager.remove_branch("child")

        assert len(removed) == 1
        assert removed[0].node_id == "child"
        assert root_id == "root"
        assert removed_entire is False
        assert manager.get_tree_for_node("child") is None
        assert manager.get_tree("root") is not None
        assert tree.get_node("child") is None
        assert "child" not in tree.get_root().children_ids

    @pytest.mark.asyncio
    async def test_remove_branch_unregisters_status_message_mappings(self):
        """Removing a branch removes node and status lookup keys."""
        manager = TreeQueueManager()

        root_incoming = IncomingMessage(
            text="Root", chat_id="1", user_id="1", message_id="root", platform="test"
        )
        await manager.create_tree("root", root_incoming, "s1")

        child_incoming = IncomingMessage(
            text="Child",
            chat_id="1",
            user_id="1",
            message_id="child",
            platform="test",
            reply_to_message_id="root",
        )
        await manager.add_to_tree("root", "child", child_incoming, "s2")
        manager.register_node("s2", "root")

        grandchild_incoming = IncomingMessage(
            text="Grandchild",
            chat_id="1",
            user_id="1",
            message_id="grandchild",
            platform="test",
            reply_to_message_id="child",
        )
        await manager.add_to_tree("child", "grandchild", grandchild_incoming, "s3")
        manager.register_node("s3", "root")

        removed, root_id, removed_entire = await manager.remove_branch("child")

        assert {node.node_id for node in removed} == {"child", "grandchild"}
        assert root_id == "root"
        assert removed_entire is False
        assert manager.get_tree_for_node("child") is None
        assert manager.get_tree_for_node("grandchild") is None
        assert manager.get_tree_for_node("s2") is None
        assert manager.get_tree_for_node("s3") is None

    @pytest.mark.asyncio
    async def test_remove_branch_root_removes_tree(self):
        """Test remove_branch when branch is root removes entire tree."""
        manager = TreeQueueManager()

        root_incoming = IncomingMessage(
            text="Root", chat_id="1", user_id="1", message_id="root", platform="test"
        )
        await manager.create_tree("root", root_incoming, "s1")

        removed, root_id, removed_entire = await manager.remove_branch("root")

        assert len(removed) == 1
        assert root_id == "root"
        assert removed_entire is True
        assert manager.get_tree("root") is None
        assert manager.get_tree_for_node("root") is None

    @pytest.mark.asyncio
    async def test_remove_branch_root_unregisters_status_message_mapping(self):
        """Removing an entire tree removes root status lookup keys too."""
        manager = TreeQueueManager()

        root_incoming = IncomingMessage(
            text="Root", chat_id="1", user_id="1", message_id="root", platform="test"
        )
        await manager.create_tree("root", root_incoming, "s1")
        manager.register_node("s1", "root")

        removed, root_id, removed_entire = await manager.remove_branch("root")

        assert [node.node_id for node in removed] == ["root"]
        assert root_id == "root"
        assert removed_entire is True
        assert manager.get_tree_for_node("root") is None
        assert manager.get_tree_for_node("s1") is None


class TestSessionStoreTrees:
    """Test SessionStore tree methods."""

    def test_save_and_get_tree(self, tmp_path):
        """Test saving and retrieving a tree."""
        from free_claude_code.messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))

        tree_data = {
            "root_id": "root_1",
            "nodes": {
                "root_1": {
                    "node_id": "root_1",
                    "state": "completed",
                    "session_id": "sess_abc",
                }
            },
        }

        snapshot = TreeSnapshot.from_json(tree_data)
        assert snapshot is not None
        store.save_tree_snapshot(snapshot)

        retrieved = store.get_tree_snapshot("root_1")
        assert retrieved is not None
        assert retrieved.root_id == "root_1"

    def test_get_tree_by_root_id(self, tmp_path):
        """Test getting tree by root ID and node mapping."""
        from free_claude_code.messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))

        tree_data = {
            "root_id": "root",
            "nodes": {
                "root": {"node_id": "root"},
                "child": {"node_id": "child"},
            },
        }

        snapshot = TreeSnapshot.from_json(tree_data)
        assert snapshot is not None
        store.save_tree_snapshot(snapshot)

        retrieved = store.get_tree_snapshot("root")
        assert retrieved is not None
        assert retrieved.root_id == "root"
        assert (
            store.load_conversation_snapshot().derive_node_to_tree()["child"] == "root"
        )
