"""In-memory repository for messaging trees and node indexes."""

from loguru import logger

from .node import MessageNode, MessageState
from .runtime import MessageTree
from .snapshot import ConversationSnapshot


class TreeRepository:
    """
    In-memory index of trees and node-to-root mappings.

    Used only by TreeQueueManager; kept as a named type for focused tests.
    """

    def __init__(self) -> None:
        self._trees: dict[str, MessageTree] = {}
        self._node_to_tree: dict[str, str] = {}

    def get_tree(self, root_id: str) -> MessageTree | None:
        """Get a tree by its root ID."""
        return self._trees.get(root_id)

    def get_tree_for_node(self, node_id: str) -> MessageTree | None:
        """Get the tree containing a given node."""
        root_id = self._node_to_tree.get(node_id)
        if not root_id:
            return None
        return self._trees.get(root_id)

    def get_node(self, node_id: str) -> MessageNode | None:
        """Get a node from any tree."""
        tree = self.get_tree_for_node(node_id)
        return tree.get_node(node_id) if tree else None

    def add_tree(self, root_id: str, tree: MessageTree) -> None:
        """Add a new tree to the repository."""
        self._trees[root_id] = tree
        self._node_to_tree[root_id] = root_id
        logger.debug("TREE_REPO: add_tree root_id={}", root_id)

    def register_node(self, node_id: str, root_id: str) -> None:
        """Register a node ID to a tree."""
        self._node_to_tree[node_id] = root_id
        logger.debug("TREE_REPO: register_node node_id={} root_id={}", node_id, root_id)

    def has_node(self, node_id: str) -> bool:
        """Check if a node is registered in any tree."""
        return node_id in self._node_to_tree

    def tree_count(self) -> int:
        """Get the number of trees in the repository."""
        return len(self._trees)

    def is_tree_busy(self, root_id: str) -> bool:
        """Check if a tree is currently processing."""
        tree = self._trees.get(root_id)
        return tree.is_processing if tree else False

    def is_node_tree_busy(self, node_id: str) -> bool:
        """Check if the tree containing a node is busy."""
        tree = self.get_tree_for_node(node_id)
        return tree.is_processing if tree else False

    def get_queue_size(self, node_id: str) -> int:
        """Get queue size for the tree containing a node."""
        tree = self.get_tree_for_node(node_id)
        return tree.get_queue_size() if tree else 0

    def resolve_parent_node_id(self, msg_id: str) -> str | None:
        """
        Resolve a message ID to the actual parent node ID.

        Handles the case where msg_id is a status message ID
        (which maps to the tree but isn't an actual node).
        """
        tree = self.get_tree_for_node(msg_id)
        if not tree:
            return None

        if tree.has_node(msg_id):
            return msg_id

        node = tree.find_node_by_status_message(msg_id)
        if node:
            return node.node_id

        return None

    def get_pending_children(self, node_id: str) -> list[MessageNode]:
        """Get all pending child nodes recursively for error propagation."""
        tree = self.get_tree_for_node(node_id)
        if not tree:
            return []

        pending: list[MessageNode] = []
        stack = [node_id]

        while stack:
            current_id = stack.pop()
            node = tree.get_node(current_id)
            if not node:
                continue
            for child_id in node.children_ids:
                child = tree.get_node(child_id)
                if child and child.state == MessageState.PENDING:
                    pending.append(child)
                    stack.append(child_id)

        return pending

    def all_trees(self) -> list[MessageTree]:
        """Get all trees in the repository."""
        return list(self._trees.values())

    def tree_ids(self) -> list[str]:
        """Get all tree root IDs."""
        return list(self._trees.keys())

    def unregister_nodes(self, node_ids: list[str]) -> None:
        """Remove lookup IDs from the node-to-tree mapping."""
        for nid in node_ids:
            self._node_to_tree.pop(nid, None)

    def unregister_node_lookups(self, nodes: list[MessageNode]) -> None:
        """Remove node and status-message lookup IDs for removed nodes."""
        lookup_ids: list[str] = []
        for node in nodes:
            lookup_ids.append(node.node_id)
            if node.status_message_id:
                lookup_ids.append(node.status_message_id)
        self.unregister_nodes(lookup_ids)

    def remove_tree(self, root_id: str) -> MessageTree | None:
        """
        Remove a tree and all its node mappings from the repository.

        Returns the removed tree, or None if not found.
        """
        tree = self._trees.pop(root_id, None)
        if not tree:
            return None
        self.unregister_node_lookups(tree.all_nodes())
        logger.debug("TREE_REPO: remove_tree root_id={}", root_id)
        return tree

    def get_message_ids_for_chat(self, platform: str, chat_id: str) -> set[str]:
        """Get all message IDs (incoming + status) for a platform/chat."""
        msg_ids: set[str] = set()
        for tree in self._trees.values():
            for node in tree.all_nodes():
                if str(node.incoming.platform) == str(platform) and str(
                    node.incoming.chat_id
                ) == str(chat_id):
                    if node.incoming.message_id is not None:
                        msg_ids.add(str(node.incoming.message_id))
                    if node.status_message_id:
                        msg_ids.add(str(node.status_message_id))
        return msg_ids

    def snapshot(self) -> ConversationSnapshot:
        """Serialize all trees into a typed conversation snapshot."""
        return ConversationSnapshot(
            trees={root_id: tree.snapshot() for root_id, tree in self._trees.items()}
        )

    @classmethod
    def from_snapshot(cls, snapshot: ConversationSnapshot) -> TreeRepository:
        """Restore repository state from a typed conversation snapshot."""
        repo = cls()
        for root_id, tree_snapshot in snapshot.trees.items():
            repo._trees[root_id] = MessageTree.from_snapshot(tree_snapshot)
        repo._node_to_tree = snapshot.derive_node_to_tree()
        return repo


__all__ = ["TreeRepository"]
