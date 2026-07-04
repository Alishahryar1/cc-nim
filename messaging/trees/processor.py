"""Async processing loop for a single messaging tree queue."""

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger

from config.settings import get_settings
from core.anthropic import get_user_facing_error_message

from ..safe_diagnostics import format_exception_for_log
from .node import MessageNode, MessageState
from .runtime import MessageTree


class TreeQueueProcessor:
    """Per-tree async queue processing owned by TreeQueueManager."""

    def __init__(
        self,
        queue_update_callback: Callable[[MessageTree], Awaitable[None]] | None = None,
        node_started_callback: Callable[[MessageTree, str], Awaitable[None]]
        | None = None,
    ) -> None:
        self._queue_update_callback = queue_update_callback
        self._node_started_callback = node_started_callback

    def set_queue_update_callback(
        self,
        queue_update_callback: Callable[[MessageTree], Awaitable[None]] | None,
    ) -> None:
        """Update the callback used to refresh queue positions."""
        self._queue_update_callback = queue_update_callback

    def set_node_started_callback(
        self,
        node_started_callback: Callable[[MessageTree, str], Awaitable[None]] | None,
    ) -> None:
        """Update the callback used when a queued node starts processing."""
        self._node_started_callback = node_started_callback

    async def _notify_queue_updated(self, tree: MessageTree) -> None:
        """Invoke queue update callback if set."""
        if not self._queue_update_callback:
            return
        try:
            await self._queue_update_callback(tree)
        except Exception as e:
            d = get_settings().log_messaging_error_details
            logger.warning(
                "Queue update callback failed: {}",
                format_exception_for_log(e, log_full_message=d),
            )

    async def notify_queue_updated(self, tree: MessageTree) -> None:
        """Invoke the queue update callback after external queue mutations."""
        await self._notify_queue_updated(tree)

    async def _notify_node_started(self, tree: MessageTree, node_id: str) -> None:
        """Invoke node started callback if set."""
        if not self._node_started_callback:
            return
        try:
            await self._node_started_callback(tree, node_id)
        except Exception as e:
            d = get_settings().log_messaging_error_details
            logger.warning(
                "Node started callback failed: {}",
                format_exception_for_log(e, log_full_message=d),
            )

    async def process_node(
        self,
        tree: MessageTree,
        node: MessageNode,
        processor: Callable[[str, MessageNode], Awaitable[None]],
    ) -> None:
        """Process a single node and then check the queue."""
        if node.state == MessageState.ERROR:
            logger.info(
                f"Skipping node {node.node_id} as it is already in state {node.state}"
            )
            await self._process_next(tree, processor)
            return

        try:
            await processor(node.node_id, node)
        except asyncio.CancelledError:
            logger.info(f"Task for node {node.node_id} was cancelled")
            raise
        except Exception as e:
            d = get_settings().log_messaging_error_details
            logger.error(
                "Error processing node {}: {}",
                node.node_id,
                format_exception_for_log(e, log_full_message=d),
            )
            await tree.update_state(
                node.node_id,
                MessageState.ERROR,
                error_message=get_user_facing_error_message(e),
            )
        finally:
            async with tree.with_lock():
                tree.clear_current_node()
            await self._process_next(tree, processor)

    async def _process_next(
        self,
        tree: MessageTree,
        processor: Callable[[str, MessageNode], Awaitable[None]],
    ) -> None:
        """Process the next message in queue, if any."""
        next_node_id = None
        node: MessageNode | None = None
        discarded_stale_ids = False
        async with tree.with_lock():
            while True:
                next_node_id = await tree.dequeue()

                if not next_node_id:
                    tree.set_processing_state(None, False)
                    logger.debug(f"Tree {tree.root_id} queue empty, marking as free")
                    break

                node = tree.get_node(next_node_id)
                if node:
                    tree.set_processing_state(next_node_id, True)
                    logger.info(f"Processing next queued node {next_node_id}")
                    tree.set_current_task(
                        asyncio.create_task(self.process_node(tree, node, processor))
                    )
                    break

                discarded_stale_ids = True
                logger.debug(
                    "Skipping stale queued node {} in tree {}",
                    next_node_id,
                    tree.root_id,
                )

        if next_node_id and node:
            await self._notify_node_started(tree, next_node_id)
            await self._notify_queue_updated(tree)
        elif discarded_stale_ids:
            await self._notify_queue_updated(tree)

    async def enqueue_and_start(
        self,
        tree: MessageTree,
        node_id: str,
        processor: Callable[[str, MessageNode], Awaitable[None]],
    ) -> bool:
        """
        Enqueue a node or start processing immediately.

        Returns True if queued, False if processing immediately.
        """
        async with tree.with_lock():
            if tree.is_processing:
                tree.put_queue_unlocked(node_id)
                queue_size = tree.get_queue_size()
                logger.info(f"Queued node {node_id}, position {queue_size}")
                return True

            tree.set_processing_state(node_id, True)

            node = tree.get_node(node_id)
            if node:
                tree.set_current_task(
                    asyncio.create_task(self.process_node(tree, node, processor))
                )
            return False

    def cancel_current(self, tree: MessageTree) -> bool:
        """Cancel the currently running task in a tree."""
        return tree.cancel_current_task()


__all__ = ["TreeQueueProcessor"]
