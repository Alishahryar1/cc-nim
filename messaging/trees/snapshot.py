"""Serializable messaging conversation snapshots."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..models import IncomingMessage
from .node import MessageNode, MessageState


@dataclass(frozen=True)
class TreeSnapshot:
    """Persisted representation of one conversation tree."""

    root_id: str
    nodes: dict[str, dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        return {"root_id": self.root_id, "nodes": dict(self.nodes)}

    @classmethod
    def from_json(cls, data: Any) -> TreeSnapshot | None:
        if not isinstance(data, dict):
            return None
        root_id = data.get("root_id")
        nodes = data.get("nodes")
        if root_id is None or not isinstance(nodes, dict):
            return None
        return cls(root_id=str(root_id), nodes=dict(nodes))

    def lookup_ids(self) -> set[str]:
        lookup_ids: set[str] = set()
        for node_key, node_data in self.nodes.items():
            lookup_ids.add(str(node_key))
            if not isinstance(node_data, dict):
                continue
            node_id = node_data.get("node_id")
            if node_id is not None:
                lookup_ids.add(str(node_id))
            status_message_id = node_data.get("status_message_id")
            if status_message_id is not None:
                lookup_ids.add(str(status_message_id))
        return lookup_ids


@dataclass(frozen=True)
class ConversationSnapshot:
    """Persisted conversation trees plus derived lookup helpers."""

    trees: dict[str, TreeSnapshot] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.trees

    def to_json(self) -> dict[str, Any]:
        return {
            "trees": {
                root_id: tree_snapshot.to_json()
                for root_id, tree_snapshot in self.trees.items()
            }
        }

    @classmethod
    def from_json(cls, data: Any) -> ConversationSnapshot:
        if not isinstance(data, dict):
            return cls()
        raw_trees = data.get("trees", {})
        if not isinstance(raw_trees, dict):
            return cls()

        trees: dict[str, TreeSnapshot] = {}
        for raw_root_id, raw_tree in raw_trees.items():
            snapshot = TreeSnapshot.from_json(raw_tree)
            if snapshot is None:
                continue
            root_id = str(raw_root_id) if raw_root_id is not None else snapshot.root_id
            trees[root_id] = snapshot
        return cls(trees=trees)

    def derive_node_to_tree(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for root_id, tree_snapshot in self.trees.items():
            for lookup_id in tree_snapshot.lookup_ids():
                mapping[lookup_id] = root_id
        return mapping

    def with_tree(self, tree_snapshot: TreeSnapshot) -> ConversationSnapshot:
        trees = dict(self.trees)
        trees[tree_snapshot.root_id] = tree_snapshot
        return ConversationSnapshot(trees=trees)

    def without_tree(self, root_id: str) -> ConversationSnapshot:
        trees = dict(self.trees)
        trees.pop(root_id, None)
        return ConversationSnapshot(trees=trees)


def node_to_snapshot(node: MessageNode) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "incoming": {
            "text": node.incoming.text,
            "chat_id": node.incoming.chat_id,
            "user_id": node.incoming.user_id,
            "message_id": node.incoming.message_id,
            "platform": node.incoming.platform,
            "reply_to_message_id": node.incoming.reply_to_message_id,
            "message_thread_id": node.incoming.message_thread_id,
            "username": node.incoming.username,
        },
        "status_message_id": node.status_message_id,
        "state": node.state.value,
        "parent_id": node.parent_id,
        "session_id": node.session_id,
        "children_ids": list(node.children_ids),
        "created_at": node.created_at.isoformat(),
        "completed_at": node.completed_at.isoformat() if node.completed_at else None,
        "error_message": node.error_message,
    }


def node_from_snapshot(data: dict[str, Any]) -> MessageNode:
    incoming_data = data["incoming"]
    incoming = IncomingMessage(
        text=incoming_data["text"],
        chat_id=incoming_data["chat_id"],
        user_id=incoming_data["user_id"],
        message_id=incoming_data["message_id"],
        platform=incoming_data["platform"],
        reply_to_message_id=incoming_data.get("reply_to_message_id"),
        message_thread_id=incoming_data.get("message_thread_id"),
        username=incoming_data.get("username"),
    )
    return MessageNode(
        node_id=data["node_id"],
        incoming=incoming,
        status_message_id=data["status_message_id"],
        state=MessageState(data["state"]),
        parent_id=data.get("parent_id"),
        session_id=data.get("session_id"),
        children_ids=list(data.get("children_ids", [])),
        created_at=datetime.fromisoformat(data["created_at"]),
        completed_at=datetime.fromisoformat(data["completed_at"])
        if data.get("completed_at")
        else None,
        error_message=data.get("error_message"),
    )
