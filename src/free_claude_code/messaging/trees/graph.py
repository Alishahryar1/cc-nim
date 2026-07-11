"""In-memory graph for one messaging conversation tree."""

from loguru import logger

from ..models import MessageScope
from .identity import TreeIdentity
from .node import MessageNode, MessageState
from .snapshot import (
    TreeSnapshot,
    node_from_snapshot,
    node_scope_from_snapshot,
    node_to_snapshot,
)


class MessageTreeGraph:
    """Own parent/child links, node lookup, and status-message lookup."""

    def __init__(self, root_node: MessageNode) -> None:
        self.root_id = root_node.node_id
        self.identity = TreeIdentity(
            scope=root_node.scope,
            root_id=root_node.node_id,
        )
        self._nodes: dict[str, MessageNode] = {root_node.node_id: root_node}
        self._status_to_node: dict[str, str] = {
            root_node.status_message_id: root_node.node_id
        }

    def add_node(
        self,
        *,
        node_id: str,
        scope: MessageScope,
        prompt: str,
        status_message_id: str,
        parent_id: str,
    ) -> MessageNode:
        if scope != self.identity.scope:
            raise ValueError("A reply cannot cross platform chat boundaries")
        if parent_id not in self._nodes:
            raise ValueError(f"Parent node {parent_id} not found in tree")
        if node_id in self._nodes:
            raise ValueError(f"Node {node_id} already exists in tree")
        if node_id in self._status_to_node:
            raise ValueError(f"Message reference {node_id} already exists in tree")
        if status_message_id in self._status_to_node:
            raise ValueError(
                f"Status message {status_message_id} already exists in tree"
            )
        if status_message_id in self._nodes:
            raise ValueError(
                f"Message reference {status_message_id} already exists in tree"
            )

        node = MessageNode(
            node_id=node_id,
            scope=scope,
            prompt=prompt,
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

    def get_parent(self, node_id: str) -> MessageNode | None:
        node = self._nodes.get(node_id)
        if not node or not node.parent_id:
            return None
        return self._nodes.get(node.parent_id)

    def get_parent_session_id(self, node_id: str) -> str | None:
        parent = self.get_parent(node_id)
        return parent.session_id if parent else None

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

    def remove_branch(self, branch_root_id: str) -> None:
        if branch_root_id not in self._nodes:
            return

        parent = self.get_parent(branch_root_id)
        removed_count = 0
        for node_id in self.get_descendants(branch_root_id):
            node = self._nodes.get(node_id)
            if not node:
                continue
            removed_count += 1
            del self._nodes[node_id]
            self._status_to_node.pop(node.status_message_id, None)

        if parent and branch_root_id in parent.children_ids:
            parent.children_ids = [
                child_id
                for child_id in parent.children_ids
                if child_id != branch_root_id
            ]

        logger.debug("Removed branch {} ({} nodes)", branch_root_id, removed_count)

    def snapshot(self) -> TreeSnapshot:
        return TreeSnapshot(
            scope=self.identity.scope,
            root_id=self.root_id,
            nodes={
                node_id: node_to_snapshot(node) for node_id, node in self._nodes.items()
            },
        )

    @classmethod
    def from_snapshot(cls, snapshot: TreeSnapshot) -> MessageTreeGraph:
        root_data = snapshot.nodes[snapshot.root_id]
        if not isinstance(root_data, dict):
            raise ValueError("Tree snapshot contains an invalid root node")
        if node_scope_from_snapshot(root_data) not in (None, snapshot.scope):
            raise ValueError("Tree snapshot contains a cross-scope node")
        root_node = node_from_snapshot(root_data, snapshot.scope)
        if root_node.node_id != snapshot.root_id:
            raise ValueError("Tree snapshot root key does not match its node ID")
        graph = cls(root_node)
        reference_owner = {
            root_node.node_id: root_node.node_id,
            root_node.status_message_id: root_node.node_id,
        }
        for snapshot_node_id, node_data in snapshot.nodes.items():
            if snapshot_node_id == snapshot.root_id:
                continue
            if not isinstance(node_data, dict):
                raise ValueError("Tree snapshot contains an invalid node")
            if node_scope_from_snapshot(node_data) not in (None, snapshot.scope):
                raise ValueError("Tree snapshot contains a cross-scope node")
            node = node_from_snapshot(node_data, snapshot.scope)
            if str(snapshot_node_id) != node.node_id:
                raise ValueError("Tree snapshot node key does not match its node ID")
            if node.node_id in graph._nodes:
                raise ValueError(f"Duplicate node {node.node_id} in tree snapshot")
            for reference in {node.node_id, node.status_message_id}:
                owner = reference_owner.get(reference)
                if owner is not None and owner != node.node_id:
                    raise ValueError(
                        f"Duplicate message reference {reference} in tree snapshot"
                    )
                reference_owner[reference] = node.node_id
            graph._nodes[node.node_id] = node
            graph._status_to_node[node.status_message_id] = node.node_id

        if root_node.parent_id is not None:
            raise ValueError("Tree snapshot root cannot have a parent")
        for node in graph._nodes.values():
            if node.node_id == graph.root_id:
                continue
            if node.parent_id is None or node.parent_id not in graph._nodes:
                raise ValueError(f"Node {node.node_id} has no valid parent")
            graph._nodes[node.parent_id].children_ids.append(node.node_id)
        if set(graph.get_descendants(graph.root_id)) != set(graph._nodes):
            raise ValueError("Tree snapshot contains a disconnected branch")
        return graph
