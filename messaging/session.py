"""
Session Store for Messaging Platforms

Provides persistent storage for mapping platform messages to Claude CLI session IDs
and message trees for conversation continuation.
"""

import contextlib
import json
import os
import tempfile
import threading
from datetime import UTC, datetime
from typing import Any

from loguru import logger


class SessionStore:
    """
    Persistent storage for message ↔ Claude session mappings and message trees.

    Uses a JSON file for storage with thread-safe operations.
    Platform-agnostic: works with any messaging platform.
    """

    def __init__(
        self,
        storage_path: str = "sessions.json",
        *,
        message_log_cap: int | None = None,
    ):
        self.storage_path = storage_path
        self._lock = threading.Lock()
        self._trees: dict[str, dict] = {}  # root_id -> tree data
        self._node_to_tree: dict[str, str] = {}  # node_id -> root_id
        # Per-chat message ID log used to support best-effort UI clearing (/clear).
        # Key: "{platform}:{chat_id}" -> list of records
        self._message_log: dict[str, list[dict[str, Any]]] = {}
        self._message_log_ids: dict[str, set[str]] = {}
        self._dirty = False
        self._save_timer: threading.Timer | None = None
        self._save_debounce_secs = 0.5
        self._message_log_cap: int | None = message_log_cap
        self._load()

    def _make_chat_key(self, platform: str, chat_id: str) -> str:
        return f"{platform}:{chat_id}"

    def _load(self) -> None:
        """Load sessions and trees from disk."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, encoding="utf-8") as f:
                data = json.load(f)

            # Load trees
            self._trees = data.get("trees", {})
            self._node_to_tree = data.get("node_to_tree", {})

            # Load message log (optional/backward compatible)
            raw_log = data.get("message_log", {}) or {}
            if isinstance(raw_log, dict):
                self._message_log = {}
                self._message_log_ids = {}
                for chat_key, items in raw_log.items():
                    if not isinstance(chat_key, str) or not isinstance(items, list):
                        continue
                    cleaned: list[dict[str, Any]] = []
                    seen: set[str] = set()
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        mid = it.get("message_id")
                        if mid is None:
                            continue
                        mid_s = str(mid)
                        if mid_s in seen:
                            continue
                        seen.add(mid_s)
                        cleaned.append(
                            {
                                "message_id": mid_s,
                                "ts": str(it.get("ts") or ""),
                                "direction": str(it.get("direction") or ""),
                                "kind": str(it.get("kind") or ""),
                            }
                        )
                    self._message_log[chat_key] = cleaned
                    self._message_log_ids[chat_key] = seen

            logger.info(
                f"Loaded {len(self._trees)} trees and "
                f"{sum(len(v) for v in self._message_log.values())} msg_ids from {self.storage_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")

    def _snapshot(self) -> dict:
        """Snapshot current state for serialization. Caller must hold self._lock."""
        return {
            "trees": dict(self._trees),
            "node_to_tree": dict(self._node_to_tree),
            "message_log": {k: list(v) for k, v in self._message_log.items()},
        }

    def _write_data(self, data: dict) -> None:
        """Atomically write data dict to disk. Must be called WITHOUT holding self._lock."""
        abs_target = os.path.abspath(self.storage_path)
        dir_name = os.path.dirname(abs_target) or "."
        fd, tmp_path = tempfile.mkstemp(
            dir=dir_name, prefix=".sessions.", suffix=".tmp.json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, abs_target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def _schedule_save(self) -> None:
        """Schedule a debounced save. Caller must hold self._lock."""
        self._dirty = True
        if self._save_timer is not None:
            self._save_timer.cancel()
            self._save_timer = None
        self._save_timer = threading.Timer(
            self._save_debounce_secs, self._save_from_timer
        )
        self._save_timer.daemon = True
        self._save_timer.start()

    def _save_from_timer(self) -> None:
        """Timer callback: save if dirty. Runs in timer thread."""
        with self._lock:
            if not self._dirty:
                self._save_timer = None
                return
            snapshot = self._snapshot()
            self._dirty = False
            self._save_timer = None
        try:
            self._write_data(snapshot)
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
            with self._lock:
                self._dirty = True

    def _flush_save(self) -> dict:
        """Cancel pending timer and snapshot current state. Caller must hold self._lock.
        Returns snapshot dict; caller must call _write_data(snapshot) outside the lock."""
        if self._save_timer is not None:
            self._save_timer.cancel()
            self._save_timer = None
        self._dirty = False
        return self._snapshot()

    def flush_pending_save(self) -> None:
        """Flush any pending debounced save. Call on shutdown to avoid losing data."""
        with self._lock:
            snapshot = self._flush_save()
        try:
            self._write_data(snapshot)
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
            with self._lock:
                self._dirty = True

    def record_message_id(
        self,
        platform: str,
        chat_id: str,
        message_id: str,
        direction: str,
        kind: str,
    ) -> None:
        """Record a message_id for later best-effort deletion (/clear)."""
        if message_id is None:
            return

        chat_key = self._make_chat_key(str(platform), str(chat_id))
        mid = str(message_id)

        with self._lock:
            seen = self._message_log_ids.setdefault(chat_key, set())
            if mid in seen:
                return

            rec = {
                "message_id": mid,
                "ts": datetime.now(UTC).isoformat(),
                "direction": str(direction),
                "kind": str(kind),
            }
            self._message_log.setdefault(chat_key, []).append(rec)
            seen.add(mid)

            # Optional cap to prevent unbounded growth if configured.
            if self._message_log_cap is not None and self._message_log_cap > 0:
                items = self._message_log.get(chat_key, [])
                if len(items) > self._message_log_cap:
                    self._message_log[chat_key] = items[-self._message_log_cap :]
                    self._message_log_ids[chat_key] = {
                        str(x.get("message_id")) for x in self._message_log[chat_key]
                    }

            self._schedule_save()

    def get_message_ids_for_chat(self, platform: str, chat_id: str) -> list[str]:
        """Get all recorded message IDs for a chat (in insertion order)."""
        chat_key = self._make_chat_key(str(platform), str(chat_id))
        with self._lock:
            items = self._message_log.get(chat_key, [])
            return [
                str(x.get("message_id"))
                for x in items
                if x.get("message_id") is not None
            ]

    def clear_all(self) -> None:
        """Clear all stored sessions/trees/mappings and persist an empty store."""
        with self._lock:
            self._trees.clear()
            self._node_to_tree.clear()
            self._message_log.clear()
            self._message_log_ids.clear()
            snapshot = self._flush_save()
        try:
            self._write_data(snapshot)
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
            with self._lock:
                self._dirty = True

    # ==================== Tree Methods ====================

    @staticmethod
    def _tree_lookup_ids(tree_data: dict) -> set[str]:
        """Return lookup IDs represented by serialized tree nodes."""
        lookup_ids: set[str] = set()
        nodes = tree_data.get("nodes", {})
        if not isinstance(nodes, dict):
            return lookup_ids

        for node_key, node_data in nodes.items():
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

    def _remove_tree_lookup_ids_unlocked(self, root_id: str) -> None:
        """Remove all lookup IDs currently pointing at a root. Caller holds lock."""
        stale_lookup_ids = [
            lookup_id
            for lookup_id, mapped_root_id in self._node_to_tree.items()
            if mapped_root_id == root_id
        ]
        for lookup_id in stale_lookup_ids:
            self._node_to_tree.pop(lookup_id, None)

    def save_tree(self, root_id: str, tree_data: dict) -> None:
        """
        Save a message tree.

        Args:
            root_id: Root node ID of the tree
            tree_data: Serialized tree data from tree.to_dict()
        """
        with self._lock:
            self._trees[root_id] = tree_data

            self._remove_tree_lookup_ids_unlocked(root_id)
            for lookup_id in self._tree_lookup_ids(tree_data):
                self._node_to_tree[lookup_id] = root_id

            self._schedule_save()
            logger.debug(f"Saved tree {root_id}")

    def get_tree(self, root_id: str) -> dict | None:
        """Get a tree by its root ID."""
        with self._lock:
            return self._trees.get(root_id)

    def register_node(self, node_id: str, root_id: str) -> None:
        """Register a node ID to a tree root."""
        with self._lock:
            self._node_to_tree[node_id] = root_id
            self._schedule_save()

    def remove_node_mappings(self, node_ids: list[str]) -> None:
        """Remove node IDs from the node-to-tree mapping."""
        with self._lock:
            for nid in node_ids:
                self._node_to_tree.pop(nid, None)
            self._schedule_save()

    def remove_tree(self, root_id: str) -> None:
        """Remove a tree and all its node mappings from the store."""
        with self._lock:
            tree_data = self._trees.pop(root_id, None)
            if tree_data:
                self._remove_tree_lookup_ids_unlocked(root_id)
                self._schedule_save()

    def get_all_trees(self) -> dict[str, dict]:
        """Get all stored trees (public accessor)."""
        with self._lock:
            return dict(self._trees)

    def get_node_mapping(self) -> dict[str, str]:
        """Get the node-to-tree mapping (public accessor)."""
        with self._lock:
            return dict(self._node_to_tree)

    def sync_from_tree_data(
        self, trees: dict[str, dict], node_to_tree: dict[str, str]
    ) -> None:
        """Sync internal tree state from external data and persist."""
        with self._lock:
            self._trees = trees
            self._node_to_tree = node_to_tree
            self._schedule_save()
