"""Async queue processor for message trees.

Handles the async processing lifecycle of tree nodes.
"""

import asyncio
import logging
from typing import Callable, Awaitable, Optional

from .tree_data import MessageTree, MessageNode, MessageState

logger = logging.getLogger(__name__)


class TreeQueueProcessor:
    """
    Handles async queue processing for a single tree.

    Separates the async processing logic from the data management.
    """

    def __init__(
        self,
        queue_update_callback: Optional[
            Callable[[MessageTree], Awaitable[None]]
        ] = None,
    ):
        self._queue_update_callback = queue_update_callback

    def set_queue_update_callback(
        self,
        queue_update_callback: Optional[
            Callable[[MessageTree], Awaitable[None]]
        ],
    ) -> None:
        """Update the callback used to refresh queue positions."""
        self._queue_update_callback = queue_update_callback

    async def _notify_queue_updated(self, tree: MessageTree) -> None:
        """Invoke queue update callback if set."""
        if not self._queue_update_callback:
            return
        try:
            await self._queue_update_callback(tree)
        except Exception as e:
            logger.warning(f"Queue update callback failed: {e}")

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
        next_node_id = None
        node = None
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

        # Queue positions changed after dequeue; refresh queued status messages.
        if next_node_id:
            await self._notify_queue_updated(tree)

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
