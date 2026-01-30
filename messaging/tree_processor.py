"""Async queue processor for message trees.

Handles the async processing lifecycle of tree nodes.
"""

import asyncio
import logging
from typing import Callable, Awaitable

from .tree_data import MessageTree, MessageNode, MessageState

logger = logging.getLogger(__name__)


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
