"""Runtime state for one messaging conversation tree."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from loguru import logger

from ..models import IncomingMessage
from .graph import MessageTreeGraph
from .node import MessageNode, MessageState
from .queue import MessageNodeQueue
from .snapshot import TreeSnapshot


class MessageTree:
    """Runtime aggregate for one ordered messaging conversation tree."""

    def __init__(
        self,
        root_node: MessageNode,
        *,
        queue: MessageNodeQueue | None = None,
        graph: MessageTreeGraph | None = None,
    ) -> None:
        self._graph = graph or MessageTreeGraph(root_node)
        self._queue = queue or MessageNodeQueue()
        self._lock = asyncio.Lock()
        self._is_processing = False
        self._current_node_id: str | None = None
        self._current_task: asyncio.Task | None = None
        logger.debug("Created MessageTree with root {}", self.root_id)

    @property
    def root_id(self) -> str:
        return self._graph.root_id

    @property
    def is_processing(self) -> bool:
        return self._is_processing

    @property
    def current_node_id(self) -> str | None:
        return self._current_node_id

    async def add_node(
        self,
        node_id: str,
        incoming: IncomingMessage,
        status_message_id: str,
        parent_id: str,
    ) -> MessageNode:
        async with self._lock:
            return self._graph.add_node(
                node_id=node_id,
                incoming=incoming,
                status_message_id=status_message_id,
                parent_id=parent_id,
            )

    def get_node(self, node_id: str) -> MessageNode | None:
        return self._graph.get_node(node_id)

    def get_root(self) -> MessageNode:
        return self._graph.get_root()

    def get_children(self, node_id: str) -> list[MessageNode]:
        return self._graph.get_children(node_id)

    def get_parent(self, node_id: str) -> MessageNode | None:
        return self._graph.get_parent(node_id)

    def get_parent_session_id(self, node_id: str) -> str | None:
        return self._graph.get_parent_session_id(node_id)

    async def update_state(
        self,
        node_id: str,
        state: MessageState,
        session_id: str | None = None,
        error_message: str | None = None,
    ) -> None:
        async with self._lock:
            self._graph.update_node_state(
                node_id,
                state,
                session_id=session_id,
                error_message=error_message,
            )

    async def enqueue(self, node_id: str) -> int:
        async with self._lock:
            self._queue.put_nowait(node_id)
            position = self._queue.qsize()
            logger.debug("Enqueued node {}, position {}", node_id, position)
            return position

    async def dequeue(self) -> str | None:
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def get_queue_snapshot(self) -> list[str]:
        async with self._lock:
            return self._queue.snapshot()

    def get_queue_size(self) -> int:
        return self._queue.qsize()

    def remove_from_queue(self, node_id: str) -> bool:
        return self._queue.remove_if_present(node_id)

    @asynccontextmanager
    async def with_lock(self):
        async with self._lock:
            yield

    def set_processing_state(self, node_id: str | None, is_processing: bool) -> None:
        self._is_processing = is_processing
        self._current_node_id = node_id if is_processing else None

    def clear_current_node(self) -> None:
        self._current_node_id = None

    def is_current_node(self, node_id: str) -> bool:
        return self._current_node_id == node_id

    def put_queue_unlocked(self, node_id: str) -> None:
        self._queue.put_nowait(node_id)

    def set_current_task(self, task: asyncio.Task | None) -> None:
        self._current_task = task

    def cancel_current_task(self) -> bool:
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            return True
        return False

    def set_node_error_sync(self, node: MessageNode, error_message: str) -> None:
        node.mark_error(error_message)

    def drain_queue_and_mark_cancelled(
        self, error_message: str = "Cancelled by user"
    ) -> list[MessageNode]:
        nodes: list[MessageNode] = []
        for node_id in self._queue.drain():
            node = self._graph.get_node(node_id)
            if node:
                self.set_node_error_sync(node, error_message)
                nodes.append(node)
        return nodes

    def reset_processing_state(self) -> None:
        self._is_processing = False
        self._current_node_id = None

    def all_nodes(self) -> list[MessageNode]:
        return self._graph.all_nodes()

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    def find_node_by_status_message(self, status_msg_id: str) -> MessageNode | None:
        return self._graph.find_node_by_status_message(status_msg_id)

    def get_descendants(self, node_id: str) -> list[str]:
        return self._graph.get_descendants(node_id)

    def remove_branch(self, branch_root_id: str) -> list[MessageNode]:
        return self._graph.remove_branch(branch_root_id)

    def snapshot(self) -> TreeSnapshot:
        return self._graph.snapshot()

    @classmethod
    def from_snapshot(cls, snapshot: TreeSnapshot) -> MessageTree:
        graph = MessageTreeGraph.from_snapshot(snapshot)
        return cls(graph.get_root(), graph=graph)
