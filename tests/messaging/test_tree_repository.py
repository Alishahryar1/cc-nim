from unittest.mock import MagicMock

import pytest

from messaging.models import IncomingMessage
from messaging.trees import MessageNode, MessageTree
from messaging.trees.repository import TreeRepository


@pytest.fixture
def repository():
    return TreeRepository()


@pytest.fixture
def sample_tree():
    incoming = IncomingMessage(
        text="root",
        chat_id="c1",
        user_id="u1",
        message_id="root_id",
        platform="telegram",
    )
    node = MessageNode(node_id="root_id", incoming=incoming, status_message_id="s1")
    return MessageTree(node)


def test_add_and_get_tree(repository, sample_tree):
    repository.add_tree("root_id", sample_tree)

    assert repository.get_tree("root_id") == sample_tree
    assert repository.get_tree_for_node("root_id") == sample_tree
    assert repository.has_node("root_id")


def test_get_tree_nonexistent(repository):
    assert repository.get_tree("none") is None
    assert repository.get_tree_for_node("none") is None


def test_register_node(repository, sample_tree):
    repository.add_tree("root_id", sample_tree)
    repository.register_node("child_id", "root_id")

    assert repository.get_tree_for_node("child_id") == sample_tree
    assert repository.has_node("child_id")


def test_get_node(repository, sample_tree):
    repository.add_tree("root_id", sample_tree)
    node = repository.get_node("root_id")

    assert node is not None
    assert node.node_id == "root_id"
    assert repository.get_node("none") is None


def test_is_tree_busy(repository, sample_tree):
    repository.add_tree("root_id", sample_tree)
    assert repository.is_tree_busy("root_id") is False

    sample_tree._is_processing = True
    assert repository.is_tree_busy("root_id") is True
    assert repository.is_node_tree_busy("root_id") is True


def test_get_queue_size(repository, sample_tree):
    repository.add_tree("root_id", sample_tree)
    assert repository.get_queue_size("root_id") == 0

    # We can't easily put items in asyncio.Queue without async,
    # but we can mock it for this unit test if needed, or just skip if it's too complex.
    # Actually, we can use a mock queue since this is a unit test of the repository wrapper.
    sample_tree._queue = MagicMock()
    sample_tree._queue.qsize.return_value = 5

    assert repository.get_queue_size("root_id") == 5


def test_resolve_parent_node_id(repository, sample_tree):
    repository.add_tree("root_id", sample_tree)
    repository.register_node("s1", "root_id")

    # 1. Direct node match
    assert repository.resolve_parent_node_id("root_id") == "root_id"

    # 2. Status message match
    # find_node_by_status_message is used inside resolve_parent_node_id
    # sample_tree has root_id node with status_message_id "s1"
    assert repository.resolve_parent_node_id("s1") == "root_id"

    # 3. No match
    assert repository.resolve_parent_node_id("unknown") is None


@pytest.mark.asyncio
async def test_get_pending_children(repository, sample_tree):
    repository.add_tree("root_id", sample_tree)

    child_incoming = IncomingMessage(
        text="child",
        chat_id="c1",
        user_id="u1",
        message_id="child_id",
        platform="telegram",
    )
    await sample_tree.add_node("child_id", child_incoming, "s2", "root_id")
    repository.register_node("child_id", "root_id")

    pending = repository.get_pending_children("root_id")
    assert len(pending) == 1
    assert pending[0].node_id == "child_id"


def test_snapshot_round_trip(repository, sample_tree):
    repository.add_tree("root_id", sample_tree)
    snapshot = repository.snapshot()

    assert "root_id" in snapshot.trees
    assert snapshot.derive_node_to_tree()["root_id"] == "root_id"

    new_repo = TreeRepository.from_snapshot(snapshot)
    tree = new_repo.get_tree("root_id")
    assert tree is not None
    assert tree.root_id == "root_id"
    assert new_repo.get_tree_for_node("root_id") == tree


def test_all_trees(repository, sample_tree):
    repository.add_tree("root_id", sample_tree)
    assert len(repository.all_trees()) == 1
    assert repository.tree_ids() == ["root_id"]
