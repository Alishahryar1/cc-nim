"""Edge case tests for messaging/session.py SessionStore."""

import json
from unittest.mock import patch

import pytest

from messaging.session import SessionStore


@pytest.fixture
def tmp_store(tmp_path):
    """Create a SessionStore using a temp file."""
    path = str(tmp_path / "sessions.json")
    return SessionStore(storage_path=path)


def _tree_node(node_id: str, status_message_id: str) -> dict:
    return {
        "node_id": node_id,
        "status_message_id": status_message_id,
    }


class TestSessionStoreLoadEdgeCases:
    """Tests for loading corrupted/malformed data."""

    def test_load_corrupted_json(self, tmp_path):
        """Corrupted JSON file is handled gracefully (logs error, starts empty)."""
        path = str(tmp_path / "sessions.json")
        with open(path, "w") as f:
            f.write("{invalid json")

        store = SessionStore(storage_path=path)
        assert len(store._trees) == 0

    def test_load_truncated_json(self, tmp_path):
        """Truncated JSON file is handled gracefully."""
        path = str(tmp_path / "sessions.json")
        with open(path, "w") as f:
            f.write('{"sessions": {"s1": {"session_id": "s1"')

        store = SessionStore(storage_path=path)
        assert len(store._trees) == 0

    def test_load_empty_file(self, tmp_path):
        """Empty file is handled gracefully."""
        path = str(tmp_path / "sessions.json")
        with open(path, "w") as f:
            f.write("")

        store = SessionStore(storage_path=path)
        assert len(store._trees) == 0

    def test_load_nonexistent_file(self, tmp_path):
        """Non-existent file starts with empty state."""
        path = str(tmp_path / "nonexistent.json")
        store = SessionStore(storage_path=path)
        assert len(store._trees) == 0

    def test_load_legacy_sessions_ignored(self, tmp_path):
        """Legacy sessions in file are ignored; trees and message_log load."""
        path = str(tmp_path / "sessions.json")
        data = {
            "sessions": {
                "s1": {
                    "session_id": "s1",
                    "chat_id": 12345,
                    "initial_msg_id": 100,
                    "last_msg_id": 200,
                    "platform": "telegram",
                    "created_at": "2025-01-01T00:00:00+00:00",
                    "updated_at": "2025-01-01T00:00:00+00:00",
                }
            },
            "trees": {"r1": {"root_id": "r1", "nodes": {"r1": {}}}},
            "node_to_tree": {"r1": "r1"},
            "message_log": {},
        }
        with open(path, "w") as f:
            json.dump(data, f)

        store = SessionStore(storage_path=path)
        assert store.get_tree("r1") is not None


class TestSessionStoreSaveEdgeCases:
    """Tests for save failure handling."""

    def test_save_io_error_handled(self, tmp_store):
        """Write failure during atomic replace is surfaced to callers."""
        tmp_store.save_tree("r1", {"root_id": "r1", "nodes": {"r1": {}}})
        with (
            patch("messaging.session.os.replace", side_effect=OSError("disk full")),
            pytest.raises(OSError),
        ):
            tmp_store._write_data(tmp_store._snapshot())


class TestSessionStoreTreeMappings:
    def test_save_tree_rebuilds_lookup_ids_for_that_root(self, tmp_path):
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)
        store.register_node("unrelated_status", "other_root")

        store.save_tree(
            "root",
            {
                "root_id": "root",
                "nodes": {
                    "root": _tree_node("root", "root_status"),
                    "child": _tree_node("child", "child_status"),
                },
            },
        )

        mapping = store.get_node_mapping()
        assert mapping["root"] == "root"
        assert mapping["root_status"] == "root"
        assert mapping["child"] == "root"
        assert mapping["child_status"] == "root"

        store.save_tree(
            "root",
            {
                "root_id": "root",
                "nodes": {
                    "root": _tree_node("root", "root_status"),
                },
            },
        )

        mapping = store.get_node_mapping()
        assert mapping["root"] == "root"
        assert mapping["root_status"] == "root"
        assert "child" not in mapping
        assert "child_status" not in mapping
        assert mapping["unrelated_status"] == "other_root"

    def test_remove_tree_removes_all_lookup_ids_for_that_root(self, tmp_path):
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)
        store.register_node("old_status", "root")
        store.register_node("unrelated_status", "other_root")
        store.save_tree(
            "root",
            {
                "root_id": "root",
                "nodes": {
                    "root": _tree_node("root", "root_status"),
                    "child": _tree_node("child", "child_status"),
                },
            },
        )

        store.remove_tree("root")

        mapping = store.get_node_mapping()
        assert "root" not in mapping
        assert "root_status" not in mapping
        assert "child" not in mapping
        assert "child_status" not in mapping
        assert "old_status" not in mapping
        assert mapping["unrelated_status"] == "other_root"


class TestSessionStoreAtomicWrites:
    """Atomic persistence: failed replace must not truncate the prior file."""

    def test_failed_replace_keeps_prior_bytes_and_marks_dirty(self, tmp_path):
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)
        store.save_tree("r1", {"root_id": "r1", "nodes": {"r1": {}}})
        store.flush_pending_save()
        with open(path, encoding="utf-8") as f:
            disk_after_first = f.read()

        store.save_tree("r2", {"root_id": "r2", "nodes": {"r2": {}}})

        with patch(
            "messaging.session.os.replace", side_effect=OSError("replace failed")
        ):
            store.flush_pending_save()

        with open(path, encoding="utf-8") as f:
            disk_after_failed = f.read()
        assert disk_after_failed == disk_after_first
        assert store._dirty is True
        assert store.get_tree("r2") is not None


class TestSessionStoreClearAll:
    def test_clear_all_wipes_state_and_persists(self, tmp_path):
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)

        store.save_tree(
            "root1",
            {
                "root_id": "root1",
                "nodes": {
                    "root1": {
                        "node_id": "root1",
                        "incoming": {
                            "text": "hello",
                            "chat_id": "c1",
                            "user_id": "u1",
                            "message_id": "m1",
                            "platform": "telegram",
                            "reply_to_message_id": None,
                            "username": None,
                        },
                        "status_message_id": "status1",
                        "state": "pending",
                        "parent_id": None,
                        "session_id": None,
                        "children_ids": [],
                        "created_at": "2025-01-01T00:00:00+00:00",
                        "completed_at": None,
                        "error_message": None,
                    }
                },
            },
        )

        store.clear_all()

        assert store.get_all_trees() == {}
        assert store.get_node_mapping() == {}

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["trees"] == {}
        assert data["node_to_tree"] == {}
        assert data["message_log"] == {}

        store2 = SessionStore(storage_path=path)
        assert len(store2._trees) == 0

    def test_message_log_persists_and_dedups(self, tmp_path):
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)

        store.record_message_id("telegram", "c1", "1", direction="in", kind="command")
        store.record_message_id("telegram", "c1", "2", direction="out", kind="command")
        store.record_message_id("telegram", "c1", "2", direction="out", kind="command")

        ids = store.get_message_ids_for_chat("telegram", "c1")
        assert ids == ["1", "2"]

        store.flush_pending_save()
        store2 = SessionStore(storage_path=path)
        assert store2.get_message_ids_for_chat("telegram", "c1") == ["1", "2"]
