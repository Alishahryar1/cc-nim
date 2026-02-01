"""Tree data structures and repository for message queue.

This module consolidates:
- MessageState: Enum for node processing states
- MessageNode: Individual message node in the tree
- MessageTree: Tree structure with queue functionality
- TreeRepository: Data access layer for managing trees
"""

import asyncio
import logging
from enum import Enum
from datetime import datetime, timezone
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field

from .models import IncomingMessage

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


class MessageState(Enum):
    """State of a message node in the tree."""

    PENDING = "pending"  # Queued, waiting to be processed
    IN_PROGRESS = "in_progress"  # Currently being processed by Claude
    COMPLETED = "completed"  # Processing finished successfully
    ERROR = "error"  # Processing failed


@dataclass
class MessageNode:
    """
    A node in the message tree.

    Each node represents a single message and tracks:
    - Its relationship to parent/children
    - Its processing state
    - Claude session information
    """

    node_id: str  # Unique ID (typically message_id)
    incoming: IncomingMessage  # The original message
    status_message_id: str  # Bot's status message ID
    state: MessageState = MessageState.PENDING
    parent_id: Optional[str] = None  # Parent node ID (None for root)
    session_id: Optional[str] = None  # Claude session ID (forked from parent)
    children_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    context: Any = None  # Additional context if needed

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "node_id": self.node_id,
            "incoming": {
                "text": self.incoming.text,
                "chat_id": self.incoming.chat_id,
                "user_id": self.incoming.user_id,
                "message_id": self.incoming.message_id,
                "platform": self.incoming.platform,
                "reply_to_message_id": self.incoming.reply_to_message_id,
                "username": self.incoming.username,
            },
            "status_message_id": self.status_message_id,
            "state": self.state.value,
            "parent_id": self.parent_id,
            "session_id": self.session_id,
            "children_ids": self.children_ids,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MessageNode":
        """Create from dictionary (JSON deserialization)."""
        incoming_data = data["incoming"]
        incoming = IncomingMessage(
            text=incoming_data["text"],
            chat_id=incoming_data["chat_id"],
            user_id=incoming_data["user_id"],
            message_id=incoming_data["message_id"],
            platform=incoming_data["platform"],
            reply_to_message_id=incoming_data.get("reply_to_message_id"),
            username=incoming_data.get("username"),
        )
        return cls(
            node_id=data["node_id"],
            incoming=incoming,
            status_message_id=data["status_message_id"],
            state=MessageState(data["state"]),
            parent_id=data.get("parent_id"),
            session_id=data.get("session_id"),
            children_ids=data.get("children_ids", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            error_message=data.get("error_message"),
        )


class MessageTree:
    """
    A tree of message nodes with queue functionality.

    Provides:
    - O(1) node lookup via hashmap
    - Per-tree message queue
    - Thread-safe operations via asyncio.Lock
    """

    def __init__(self, root_node: MessageNode):
        """
        Initialize tree with a root node.

        Args:
            root_node: The root message node
        """
        self.root_id = root_node.node_id
        self._nodes: Dict[str, MessageNode] = {root_node.node_id: root_node}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._is_processing = False
        self._current_node_id: Optional[str] = None
        self._current_task: Optional[asyncio.Task] = None

        logger.debug(f"Created MessageTree with root {self.root_id}")

    @property
    def is_processing(self) -> bool:
        """Check if tree is currently processing a message."""
        return self._is_processing

    async def add_node(
        self,
        node_id: str,
        incoming: IncomingMessage,
        status_message_id: str,
        parent_id: str,
    ) -> MessageNode:
        """
        Add a child node to the tree.

        Args:
            node_id: Unique ID for the new node
            incoming: The incoming message
            status_message_id: Bot's status message ID
            parent_id: Parent node ID

        Returns:
            The created MessageNode
        """
        async with self._lock:
            if parent_id not in self._nodes:
                raise ValueError(f"Parent node {parent_id} not found in tree")

            node = MessageNode(
                node_id=node_id,
                incoming=incoming,
                status_message_id=status_message_id,
                parent_id=parent_id,
                state=MessageState.PENDING,
            )

            self._nodes[node_id] = node
            self._nodes[parent_id].children_ids.append(node_id)

            logger.debug(f"Added node {node_id} as child of {parent_id}")
            return node

    def get_node(self, node_id: str) -> Optional[MessageNode]:
        """Get a node by ID (O(1) lookup)."""
        return self._nodes.get(node_id)

    def get_root(self) -> MessageNode:
        """Get the root node."""
        return self._nodes[self.root_id]

    def get_children(self, node_id: str) -> List[MessageNode]:
        """Get all child nodes of a given node."""
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[cid] for cid in node.children_ids if cid in self._nodes]

    def get_parent(self, node_id: str) -> Optional[MessageNode]:
        """Get the parent node."""
        node = self._nodes.get(node_id)
        if not node or not node.parent_id:
            return None
        return self._nodes.get(node.parent_id)

    def get_parent_session_id(self, node_id: str) -> Optional[str]:
        """
        Get the parent's session ID for forking.

        Returns None for root nodes.
        """
        parent = self.get_parent(node_id)
        return parent.session_id if parent else None

    async def update_state(
        self,
        node_id: str,
        state: MessageState,
        session_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update a node's state."""
        async with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                logger.warning(f"Node {node_id} not found for state update")
                return

            node.state = state
            if session_id:
                node.session_id = session_id
            if error_message:
                node.error_message = error_message
            if state in (MessageState.COMPLETED, MessageState.ERROR):
                node.completed_at = datetime.now(timezone.utc)

            logger.debug(f"Node {node_id} state -> {state.value}")

    async def enqueue(self, node_id: str) -> int:
        """
        Add a node to the processing queue.

        Returns:
            Queue position (1-indexed)
        """
        async with self._lock:
            await self._queue.put(node_id)
            position = self._queue.qsize()
            logger.debug(f"Enqueued node {node_id}, position {position}")
            return position

    async def dequeue(self) -> Optional[str]:
        """
        Get the next node ID from the queue.

        Returns None if queue is empty.
        """
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def get_queue_size(self) -> int:
        """Get number of messages waiting in queue."""
        return self._queue.qsize()

    def to_dict(self) -> dict:
        """Serialize tree to dictionary."""
        return {
            "root_id": self.root_id,
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MessageTree":
        """Deserialize tree from dictionary."""
        root_id = data["root_id"]
        nodes_data = data["nodes"]

        # Create root node first
        root_node = MessageNode.from_dict(nodes_data[root_id])
        tree = cls(root_node)

        # Add remaining nodes
        for node_id, node_data in nodes_data.items():
            if node_id != root_id:
                tree._nodes[node_id] = MessageNode.from_dict(node_data)

        return tree

    def all_nodes(self) -> List[MessageNode]:
        """Get all nodes in the tree."""
        return list(self._nodes.values())

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in this tree."""
        return node_id in self._nodes

    def find_node_by_status_message(self, status_msg_id: str) -> Optional[MessageNode]:
        """Find the node that has this status message ID."""
        for node in self._nodes.values():
            if node.status_message_id == status_msg_id:
                return node
        return None


# =============================================================================
# Repository
# =============================================================================


class TreeRepository:
    """
    Repository for message tree data access.

    Manages the storage and lookup of trees and node-to-tree mappings.
    """

    def __init__(self):
        self._trees: Dict[str, MessageTree] = {}  # root_id -> tree
        self._node_to_tree: Dict[str, str] = {}  # node_id -> root_id

    def get_tree(self, root_id: str) -> Optional[MessageTree]:
        """Get a tree by its root ID."""
        return self._trees.get(root_id)

    def get_tree_for_node(self, node_id: str) -> Optional[MessageTree]:
        """Get the tree containing a given node."""
        root_id = self._node_to_tree.get(node_id)
        if not root_id:
            return None
        return self._trees.get(root_id)

    def get_node(self, node_id: str) -> Optional[MessageNode]:
        """Get a node from any tree."""
        tree = self.get_tree_for_node(node_id)
        return tree.get_node(node_id) if tree else None

    def add_tree(self, root_id: str, tree: MessageTree) -> None:
        """Add a new tree to the repository."""
        self._trees[root_id] = tree
        self._node_to_tree[root_id] = root_id

    def register_node(self, node_id: str, root_id: str) -> None:
        """Register a node ID to a tree."""
        self._node_to_tree[node_id] = root_id

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

    def resolve_parent_node_id(self, msg_id: str) -> Optional[str]:
        """
        Resolve a message ID to the actual parent node ID.

        Handles the case where msg_id is a status message ID
        (which maps to the tree but isn't an actual node).

        Returns:
            The node_id to use as parent, or None if not found
        """
        tree = self.get_tree_for_node(msg_id)
        if not tree:
            return None

        # Check if msg_id is an actual node
        if tree.has_node(msg_id):
            return msg_id

        # Otherwise, it might be a status message - find the owning node
        node = tree.find_node_by_status_message(msg_id)
        if node:
            return node.node_id

        return None

    def get_pending_children(self, node_id: str) -> List[MessageNode]:
        """
        Get all pending child nodes (recursively) of a given node.

        Used for error propagation - when a node fails, its pending
        children should also be marked as failed.
        """
        tree = self.get_tree_for_node(node_id)
        if not tree:
            return []

        pending = []
        node = tree.get_node(node_id)
        if not node:
            return []

        for child_id in node.children_ids:
            child = tree.get_node(child_id)
            if child and child.state == MessageState.PENDING:
                pending.append(child)
                # Recursively get children of pending children
                pending.extend(self.get_pending_children(child_id))

        return pending

    def all_trees(self) -> List[MessageTree]:
        """Get all trees in the repository."""
        return list(self._trees.values())

    def tree_ids(self) -> List[str]:
        """Get all tree root IDs."""
        return list(self._trees.keys())

    def to_dict(self) -> dict:
        """Serialize all trees."""
        return {
            "trees": {rid: tree.to_dict() for rid, tree in self._trees.items()},
            "node_to_tree": self._node_to_tree.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TreeRepository":
        """Deserialize from dictionary."""
        repo = cls()
        for root_id, tree_data in data.get("trees", {}).items():
            repo._trees[root_id] = MessageTree.from_dict(tree_data)
        repo._node_to_tree = data.get("node_to_tree", {})
        return repo
