"""Tree-Based Message Queue Manager.

Coordinates data access, async processing, and error handling.
Uses TreeRepository for data, TreeQueueProcessor for async logic.

This module consolidates the queue manager and processor into a single file.
"""

import asyncio
import logging
from typing import Callable, Awaitable, List, Optional

from .models import IncomingMessage
from .tree_core import (
    MessageState,
    MessageNode,
    MessageTree,
    TreeRepository,
)

# Backward compatibility: re-export data structures
__all__ = [
    "TreeQueueManager",
    "TreeQueueProcessor",
    "MessageState",
    "MessageNode",
    "MessageTree",
    "TreeRepository",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Queue Processor
# =============================================================================


class TreeQueueProcessor:
    """
    Handles async queue processing for a single tree.

    Separates the async processing logic from the data management.
    """

    async def process_node(
        self,
        tree: MessageTree,
        node: MessageNode,
        processor: Callable[[str, MessageNode], Awaitable[None]],
    ) -> None:
        """Process a single node and then check the queue."""
        # Skip if already in terminal state (e.g. from error propagation)
        if node.state.value == MessageState.ERROR.value:
            logger.info(
                f"Skipping node {node.node_id} as it is already in state {node.state}"
            )
            # Still need to check for next messages
            await self._process_next(tree, processor)
            return

        try:
            await processor(node.node_id, node)
        except asyncio.CancelledError:
            logger.info(f"Task for node {node.node_id} was cancelled")
            raise
        except Exception as e:
            logger.error(f"Error processing node {node.node_id}: {e}")
            await tree.update_state(
                node.node_id, MessageState.ERROR, error_message=str(e)
            )
        finally:
            tree._current_node_id = None
            # Check if there are more messages in the queue
            await self._process_next(tree, processor)

    async def _process_next(
        self,
        tree: MessageTree,
        processor: Callable[[str, MessageNode], Awaitable[None]],
    ) -> None:
        """Process the next message in queue, if any."""
        async with tree._lock:
            next_node_id = await tree.dequeue()

            if not next_node_id:
                # No more messages, mark tree as free
                tree._is_processing = False
                logger.debug(f"Tree {tree.root_id} queue empty, marking as free")
                return

            tree._current_node_id = next_node_id
            logger.info(f"Processing next queued node {next_node_id}")

            # Process next node (outside lock)
            node = tree.get_node(next_node_id)
            if node:
                tree._current_task = asyncio.create_task(
                    self.process_node(tree, node, processor)
                )

    async def enqueue_and_start(
        self,
        tree: MessageTree,
        node_id: str,
        processor: Callable[[str, MessageNode], Awaitable[None]],
    ) -> bool:
        """
        Enqueue a node or start processing immediately.

        Args:
            tree: The message tree
            node_id: Node to process
            processor: Async function to process the node

        Returns:
            True if queued, False if processing immediately
        """
        async with tree._lock:
            if tree._is_processing:
                # Tree is busy, queue the message
                await tree._queue.put(node_id)
                queue_size = tree._queue.qsize()
                logger.info(f"Queued node {node_id}, position {queue_size}")
                return True
            else:
                # Tree is free, start processing
                tree._is_processing = True
                tree._current_node_id = node_id

                # Process outside the lock
                node = tree.get_node(node_id)
                if node:
                    tree._current_task = asyncio.create_task(
                        self.process_node(tree, node, processor)
                    )
                return False

    def cancel_current(self, tree: MessageTree) -> bool:
        """Cancel the currently running task in a tree."""
        if tree._current_task and not tree._current_task.done():
            tree._current_task.cancel()
            return True
        return False


# =============================================================================
# Queue Manager
# =============================================================================


class TreeQueueManager:
    """
    Manages multiple message trees. Facade that coordinates components.

    Each new conversation creates a new tree.
    Replies to existing messages add nodes to existing trees.

    Components:
        - TreeRepository: Data access layer
        - TreeQueueProcessor: Async queue processing
    """

    def __init__(self):
        self._repository = TreeRepository()
        self._processor = TreeQueueProcessor()
        self._lock = asyncio.Lock()

        logger.info("TreeQueueManager initialized")

    @property
    def _trees(self):
        """Access internal tree dict for backward compatibility."""
        return self._repository._trees

    @property
    def _node_to_tree(self):
        """Access internal node mapping for backward compatibility."""
        return self._repository._node_to_tree

    async def create_tree(
        self,
        node_id: str,
        incoming: IncomingMessage,
        status_message_id: str,
    ) -> MessageTree:
        """
        Create a new tree with a root node.

        Args:
            node_id: ID for the root node
            incoming: The incoming message
            status_message_id: Bot's status message ID

        Returns:
            The created MessageTree
        """
        async with self._lock:
            root_node = MessageNode(
                node_id=node_id,
                incoming=incoming,
                status_message_id=status_message_id,
                state=MessageState.PENDING,
            )

            tree = MessageTree(root_node)
            self._repository.add_tree(node_id, tree)

            logger.info(f"Created new tree with root {node_id}")
            return tree

    async def add_to_tree(
        self,
        parent_node_id: str,
        node_id: str,
        incoming: IncomingMessage,
        status_message_id: str,
    ) -> tuple[MessageTree, MessageNode]:
        """
        Add a reply as a child node to an existing tree.

        Args:
            parent_node_id: ID of the parent message
            node_id: ID for the new node
            incoming: The incoming reply message
            status_message_id: Bot's status message ID

        Returns:
            Tuple of (tree, new_node)
        """
        async with self._lock:
            if parent_node_id not in self._repository._node_to_tree:
                raise ValueError(f"Parent node {parent_node_id} not found in any tree")

            root_id = self._repository._node_to_tree[parent_node_id]
            tree = self._repository._trees[root_id]

        # Add node (tree has its own lock) - outside manager lock to avoid deadlock
        node = await tree.add_node(
            node_id=node_id,
            incoming=incoming,
            status_message_id=status_message_id,
            parent_id=parent_node_id,
        )

        async with self._lock:
            self._repository.register_node(node_id, root_id)

        logger.info(f"Added node {node_id} to tree {root_id}")
        return tree, node

    def get_tree(self, root_id: str) -> Optional[MessageTree]:
        """Get a tree by its root ID."""
        return self._repository.get_tree(root_id)

    def get_tree_for_node(self, node_id: str) -> Optional[MessageTree]:
        """Get the tree containing a given node."""
        return self._repository.get_tree_for_node(node_id)

    def get_node(self, node_id: str) -> Optional[MessageNode]:
        """Get a node from any tree."""
        return self._repository.get_node(node_id)

    def resolve_parent_node_id(self, msg_id: str) -> Optional[str]:
        """Resolve a message ID to the actual parent node ID."""
        return self._repository.resolve_parent_node_id(msg_id)

    def is_tree_busy(self, root_id: str) -> bool:
        """Check if a tree is currently processing."""
        return self._repository.is_tree_busy(root_id)

    def is_node_tree_busy(self, node_id: str) -> bool:
        """Check if the tree containing a node is busy."""
        return self._repository.is_node_tree_busy(node_id)

    async def enqueue(
        self,
        node_id: str,
        processor: Callable[[str, MessageNode], Awaitable[None]],
    ) -> bool:
        """
        Enqueue a node for processing.

        If the tree is not busy, processing starts immediately.
        If busy, the message is queued.

        Args:
            node_id: Node to process
            processor: Async function to process the node

        Returns:
            True if queued, False if processing immediately
        """
        tree = self._repository.get_tree_for_node(node_id)
        if not tree:
            logger.error(f"No tree found for node {node_id}")
            return False

        return await self._processor.enqueue_and_start(tree, node_id, processor)

    def get_queue_size(self, node_id: str) -> int:
        """Get queue size for the tree containing a node."""
        return self._repository.get_queue_size(node_id)

    def get_pending_children(self, node_id: str) -> List[MessageNode]:
        """Get all pending child nodes (recursively) of a given node."""
        return self._repository.get_pending_children(node_id)

    async def mark_node_error(
        self,
        node_id: str,
        error_message: str,
        propagate_to_children: bool = True,
    ) -> List[MessageNode]:
        """
        Mark a node as ERROR and optionally propagate to pending children.

        Args:
            node_id: The node to mark as error
            error_message: Error description
            propagate_to_children: If True, also mark pending children as error

        Returns:
            List of all nodes marked as error (including children)
        """
        tree = self._repository.get_tree_for_node(node_id)
        if not tree:
            return []

        affected = []
        node = tree.get_node(node_id)
        if node:
            await tree.update_state(
                node_id, MessageState.ERROR, error_message=error_message
            )
            affected.append(node)

        if propagate_to_children:
            pending_children = self._repository.get_pending_children(node_id)
            for child in pending_children:
                await tree.update_state(
                    child.node_id,
                    MessageState.ERROR,
                    error_message=f"Parent failed: {error_message}",
                )
                affected.append(child)

        return affected

    def cancel_tree(self, root_id: str) -> List[MessageNode]:
        """
        Cancel all queued and in-progress messages in a tree.

        Updates node states to ERROR and returns list of affected nodes
        that were actually active or in the current processing queue.
        """
        tree = self._repository.get_tree(root_id)
        if not tree:
            return []

        cancelled_nodes = []

        # 1. Cancel running task via processor
        if self._processor.cancel_current(tree):
            if tree._current_node_id:
                node = tree.get_node(tree._current_node_id)
                if node and node.state not in (
                    MessageState.COMPLETED,
                    MessageState.ERROR,
                ):
                    node.state = MessageState.ERROR
                    node.error_message = "Cancelled by user"
                    cancelled_nodes.append(node)

        # 2. Clear queue and update states
        while not tree._queue.empty():
            try:
                node_id = tree._queue.get_nowait()
                node = tree.get_node(node_id)
                if node:
                    node.state = MessageState.ERROR
                    node.error_message = "Cancelled by user"
                    cancelled_nodes.append(node)
            except asyncio.QueueEmpty:
                break

        # 3. Cleanup: Mark ANY other PENDING or IN_PROGRESS nodes as ERROR
        # This handles stale nodes from previous sessions without counting them as "cancelled now"
        # unless they were in the active queue above.
        cleanup_count = 0
        for node in tree.all_nodes():
            if (
                node.state in (MessageState.PENDING, MessageState.IN_PROGRESS)
                and node not in cancelled_nodes
            ):
                node.state = MessageState.ERROR
                node.error_message = "Stale task cleaned up"
                cleanup_count += 1

        tree._is_processing = False
        tree._current_node_id = None

        if cancelled_nodes:
            logger.info(
                f"Cancelled {len(cancelled_nodes)} active nodes in tree {root_id}"
            )
        if cleanup_count:
            logger.info(f"Cleaned up {cleanup_count} stale nodes in tree {root_id}")

        return cancelled_nodes

    async def cancel_all(self) -> List[MessageNode]:
        """Cancel all messages in all trees (async wrapper)."""
        async with self._lock:
            return self.cancel_all_sync()

    def cancel_all_sync(self) -> List[MessageNode]:
        """
        Cancel all messages in all trees (synchronous/locked version).
        NOTE: Must be called with self._lock held.
        """
        all_cancelled = []
        for root_id in list(self._repository.tree_ids()):
            all_cancelled.extend(self.cancel_tree(root_id))
        return all_cancelled

    def cleanup_stale_nodes(self) -> int:
        """
        Mark any PENDING or IN_PROGRESS nodes in all trees as ERROR.
        Used on startup to reconcile restored state.
        """
        count = 0
        for tree in self._repository.all_trees():
            for node in tree.all_nodes():
                if node.state in (MessageState.PENDING, MessageState.IN_PROGRESS):
                    node.state = MessageState.ERROR
                    node.error_message = "Lost during server restart"
                    count += 1
        if count:
            logger.info(f"Cleaned up {count} stale nodes during startup")
        return count

    def register_node(self, node_id: str, root_id: str) -> None:
        """Register a node ID to a tree (for external mapping)."""
        self._repository.register_node(node_id, root_id)

    def to_dict(self) -> dict:
        """Serialize all trees."""
        return self._repository.to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> "TreeQueueManager":
        """Deserialize from dictionary."""
        manager = cls()
        manager._repository = TreeRepository.from_dict(data)
        return manager
