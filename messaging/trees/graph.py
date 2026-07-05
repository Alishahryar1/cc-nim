"""In-memory graph for one messaging conversation tree."""

from loguru import logger

from ..models import IncomingMessage
from .node import MessageNode, MessageState
from .snapshot import TreeSnapshot, node_from_snapshot, node_to_snapshot


class MessageTreeGraph:
    """Own parent/child links, node lookup, and status-message lookup."""

    def __init__(self, root_node: MessageNode) -> None:
        self.root_id = root_node.node_id
        self._nodes: dict[str, MessageNode] = {root_node.node_id: root_node}
        self._status_to_node: dict[str, str] = {
            root_node.status_message_id: root_node.node_id
        }

    def add_node(
        self,
        *,
        node_id: str,
        incoming: IncomingMessage,
        status_message_id: str,
        parent_id: str,
    ) -> MessageNode:
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
        self._status_to_node[status_message_id] = node_id
        self._nodes[parent_id].children_ids.append(node_id)
        logger.debug("Added node {} as child of {}", node_id, parent_id)
        return node

    def get_node(self, node_id: str) -> MessageNode | None:
        return self._nodes.get(node_id)

    def get_root(self) -> MessageNode:
        return self._nodes[self.root_id]

    def get_children(self, node_id: str) -> list[MessageNode]:
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[cid] for cid in node.children_ids if cid in self._nodes]

    def get_parent(self, node_id: str) -> MessageNode | None:
        node = self._nodes.get(node_id)
        if not node or not node.parent_id:
            return None
        return self._nodes.get(node.parent_id)

    def get_parent_session_id(self, node_id: str) -> str | None:
        parent = self.get_parent(node_id)
        return parent.session_id if parent else None

    def update_node_state(
        self,
        node_id: str,
        state: MessageState,
        *,
        session_id: str | None = None,
        error_message: str | None = None,
    ) -> bool:
        node = self._nodes.get(node_id)
        if not node:
            logger.warning("Node {} not found for state update", node_id)
            return False
        node.update_state(
            state,
            session_id=session_id,
            error_message=error_message,
        )
        logger.debug("Node {} state -> {}", node_id, state.value)
        return True

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def find_node_by_status_message(self, status_msg_id: str) -> MessageNode | None:
        node_id = self._status_to_node.get(status_msg_id)
        return self._nodes.get(node_id) if node_id else None

    def all_nodes(self) -> list[MessageNode]:
        return list(self._nodes.values())

    def get_descendants(self, node_id: str) -> list[str]:
        if node_id not in self._nodes:
            return []
        result: list[str] = []
        stack = [node_id]
        while stack:
            current_id = stack.pop()
            result.append(current_id)
            node = self._nodes.get(current_id)
            if node:
                stack.extend(node.children_ids)
        return result

    def remove_branch(self, branch_root_id: str) -> list[MessageNode]:
        if branch_root_id not in self._nodes:
            return []

        parent = self.get_parent(branch_root_id)
        removed: list[MessageNode] = []
        for node_id in self.get_descendants(branch_root_id):
            node = self._nodes.get(node_id)
            if not node:
                continue
            removed.append(node)
            del self._nodes[node_id]
            self._status_to_node.pop(node.status_message_id, None)

        if parent and branch_root_id in parent.children_ids:
            parent.children_ids = [
                child_id
                for child_id in parent.children_ids
                if child_id != branch_root_id
            ]

        logger.debug("Removed branch {} ({} nodes)", branch_root_id, len(removed))
        return removed

    def snapshot(self) -> TreeSnapshot:
        return TreeSnapshot(
            root_id=self.root_id,
            nodes={
                node_id: node_to_snapshot(node) for node_id, node in self._nodes.items()
            },
        )

    @classmethod
    def from_snapshot(cls, snapshot: TreeSnapshot) -> MessageTreeGraph:
        root_data = snapshot.nodes[snapshot.root_id]
        root_node = node_from_snapshot(root_data)
        graph = cls(root_node)
        for node_id, node_data in snapshot.nodes.items():
            if node_id == snapshot.root_id:
                continue
            if not isinstance(node_data, dict):
                continue
            node = node_from_snapshot(node_data)
            graph._nodes[node.node_id] = node
            graph._status_to_node[node.status_message_id] = node.node_id
        return graph
