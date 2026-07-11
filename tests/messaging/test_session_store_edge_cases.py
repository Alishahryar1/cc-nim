"""Edge case tests for the messaging session store."""

import json
import threading
from collections.abc import Callable
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

import free_claude_code.messaging.session.persistence as persistence_module
from free_claude_code.messaging.models import MessageScope
from free_claude_code.messaging.session import SessionStore
from free_claude_code.messaging.session.persistence import DebouncedJsonPersistence
from free_claude_code.messaging.trees import TreeIdentity, TreeSnapshot

TELEGRAM_C1 = MessageScope(platform="telegram", chat_id="c1")


def _identity(root_id: str, scope: MessageScope = TELEGRAM_C1) -> TreeIdentity:
    return TreeIdentity(scope=scope, root_id=root_id)


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


class FakeTimer:
    instances: ClassVar[list[FakeTimer]] = []

    def __init__(
        self,
        interval: float,
        function: Callable[..., None],
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.interval = interval
        self.function = function
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.daemon = False
        self.canceled = False
        self.started = False
        self.instances.append(self)

    def cancel(self) -> None:
        self.canceled = True

    def start(self) -> None:
        self.started = True

    def fire(self, *, force: bool = False) -> None:
        if self.canceled and not force:
            return
        self.function(*self.args, **self.kwargs)


class RecordingPersistence(DebouncedJsonPersistence):
    def __init__(
        self,
        storage_path: str,
        *,
        snapshot: Callable[[], dict[str, Any]],
        on_dirty: Callable[[bool], None],
    ) -> None:
        self.writes: list[dict[str, Any]] = []
        super().__init__(storage_path, snapshot=snapshot, on_dirty=on_dirty)

    def _write_file(self, data: dict[str, Any]) -> None:
        self.writes.append(data)


class TestSessionStoreLoadEdgeCases:
    """Tests for loading corrupted/malformed data."""

    def test_load_corrupted_json(self, tmp_path):
        """Corrupted JSON file is handled gracefully (logs error, starts empty)."""
        path = str(tmp_path / "sessions.json")
        with open(path, "w") as f:
            f.write("{invalid json")

        store = SessionStore(storage_path=path)
        assert store.load_conversation_snapshot().is_empty

    def test_load_truncated_json(self, tmp_path):
        """Truncated JSON file is handled gracefully."""
        path = str(tmp_path / "sessions.json")
        with open(path, "w") as f:
            f.write('{"sessions": {"s1": {"session_id": "s1"')

        store = SessionStore(storage_path=path)
        assert store.load_conversation_snapshot().is_empty

    def test_load_empty_file(self, tmp_path):
        """Empty file is handled gracefully."""
        path = str(tmp_path / "sessions.json")
        with open(path, "w") as f:
            f.write("")

        store = SessionStore(storage_path=path)
        assert store.load_conversation_snapshot().is_empty

    def test_load_nonexistent_file(self, tmp_path):
        """Non-existent file starts with empty state."""
        path = str(tmp_path / "nonexistent.json")
        store = SessionStore(storage_path=path)
        assert store.load_conversation_snapshot().is_empty

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
            "trees": {
                "r1": {
                    "root_id": "r1",
                    "nodes": {
                        "r1": {
                            "node_id": "r1",
                            "incoming": {
                                "platform": TELEGRAM_C1.platform,
                                "chat_id": TELEGRAM_C1.chat_id,
                            },
                        }
                    },
                }
            },
            "node_to_tree": {"r1": "r1"},
            "message_log": {},
        }
        with open(path, "w") as f:
            json.dump(data, f)

        store = SessionStore(storage_path=path)
        assert store.load_conversation_snapshot().get_tree(_identity("r1")) is not None


class TestSessionStoreSaveEdgeCases:
    """Tests for save failure handling."""

    def test_save_io_error_handled(self, tmp_store):
        """Write failure marks pending state dirty without crashing callers."""
        tmp_store.save_tree_snapshot(
            TreeSnapshot(scope=TELEGRAM_C1, root_id="r1", nodes={"r1": {}})
        )
        with patch(
            "free_claude_code.messaging.session.persistence.os.replace",
            side_effect=OSError("disk full"),
        ):
            tmp_store.flush_pending_save()
        assert tmp_store.dirty is True

    def test_stale_timer_callback_cannot_clear_newer_timer(self, tmp_path, monkeypatch):
        """An already-running old timer cannot consume the newest save."""
        FakeTimer.instances = []
        monkeypatch.setattr(persistence_module.threading, "Timer", FakeTimer)

        dirty_states: list[bool] = []
        snapshot_count = 0

        def snapshot() -> dict[str, Any]:
            nonlocal snapshot_count
            snapshot_count += 1
            return {"snapshot": snapshot_count}

        persistence = RecordingPersistence(
            str(tmp_path / "sessions.json"),
            snapshot=snapshot,
            on_dirty=dirty_states.append,
        )

        persistence.schedule_save()
        first_timer = FakeTimer.instances[0]
        persistence.schedule_save()
        second_timer = FakeTimer.instances[1]

        first_timer.fire(force=True)
        assert persistence.writes == []
        assert dirty_states[-1] is True
        assert second_timer.canceled is False

        second_timer.fire()
        assert persistence.writes == [{"snapshot": 1}]
        assert dirty_states[-1] is False

    def test_running_old_write_finishes_before_newer_flush(self, tmp_path, monkeypatch):
        """A claimed old snapshot cannot land after a newer flushed snapshot."""
        FakeTimer.instances = []
        monkeypatch.setattr(persistence_module.threading, "Timer", FakeTimer)

        state = {"version": "old"}
        dirty_states: list[bool] = []
        old_write_started = threading.Event()
        release_old_write = threading.Event()

        class BlockingPersistence(RecordingPersistence):
            def _write_file(self, data: dict[str, Any]) -> None:
                if data == {"version": "old"}:
                    old_write_started.set()
                    release_old_write.wait(timeout=2)
                super()._write_file(data)

        persistence = BlockingPersistence(
            str(tmp_path / "sessions.json"),
            snapshot=lambda: dict(state),
            on_dirty=dirty_states.append,
        )
        persistence.schedule_save()
        old_writer = threading.Thread(target=FakeTimer.instances[0].fire)
        old_writer.start()
        assert old_write_started.wait(timeout=2)

        state["version"] = "new"
        persistence.schedule_save()
        new_writer = threading.Thread(target=persistence.flush)
        new_writer.start()
        assert new_writer.is_alive()

        release_old_write.set()
        old_writer.join(timeout=2)
        new_writer.join(timeout=2)

        assert not old_writer.is_alive()
        assert not new_writer.is_alive()
        assert persistence.writes == [{"version": "old"}, {"version": "new"}]
        assert dirty_states[-1] is False


class TestSessionStoreTreeSnapshots:
    def test_unscoped_tree_without_legacy_ingress_is_reported_and_skipped(
        self, tmp_path
    ):
        path = tmp_path / "sessions.json"
        path.write_text(
            json.dumps(
                {
                    "conversation": {
                        "trees": [
                            {
                                "root_id": "root",
                                "nodes": {
                                    "root": {
                                        "node_id": "root",
                                        "status_message_id": "status",
                                        "state": "completed",
                                    }
                                },
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )

        with patch(
            "free_claude_code.messaging.trees.snapshot.logger.warning"
        ) as warning:
            store = SessionStore(storage_path=str(path))

        assert store.load_conversation_snapshot().is_empty
        warning.assert_called_once_with(
            "Skipping messaging tree snapshot without recoverable scope: root_id={}",
            "root",
        )

    def test_snapshot_ingress_and_egress_are_deeply_detached(self, tmp_path):
        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        snapshot = TreeSnapshot(
            scope=TELEGRAM_C1,
            root_id="root",
            nodes={"root": {"node_id": "root", "state": "completed"}},
        )
        store.save_tree_snapshot(snapshot)
        snapshot.nodes["root"]["state"] = "mutated-after-save"

        loaded = store.load_conversation_snapshot()
        loaded_tree = loaded.get_tree(_identity("root"))
        assert loaded_tree is not None
        assert loaded_tree.nodes["root"]["state"] == "completed"
        loaded_tree.nodes["root"]["state"] = "mutated-after-load"

        reloaded = store.load_conversation_snapshot().get_tree(_identity("root"))
        assert reloaded is not None
        assert reloaded.nodes["root"]["state"] == "completed"

    def test_save_tree_replaces_snapshot_for_scoped_identity(self, tmp_path):
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)

        store.save_tree_snapshot(
            TreeSnapshot(
                scope=TELEGRAM_C1,
                root_id="root",
                nodes={
                    "root": _tree_node("root", "root_status"),
                    "child": _tree_node("child", "child_status"),
                },
            )
        )

        saved = store.load_conversation_snapshot().get_tree(_identity("root"))
        assert saved is not None
        assert saved.lookup_ids() == {
            "root",
            "root_status",
            "child",
            "child_status",
        }

        store.save_tree_snapshot(
            TreeSnapshot(
                scope=TELEGRAM_C1,
                root_id="root",
                nodes={
                    "root": _tree_node("root", "root_status"),
                },
            )
        )

        replaced = store.load_conversation_snapshot().get_tree(_identity("root"))
        assert replaced is not None
        assert replaced.lookup_ids() == {"root", "root_status"}

    def test_remove_tree_removes_only_scoped_identity(self, tmp_path):
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)
        store.save_tree_snapshot(
            TreeSnapshot(
                scope=TELEGRAM_C1,
                root_id="root",
                nodes={
                    "root": _tree_node("root", "root_status"),
                    "child": _tree_node("child", "child_status"),
                },
            )
        )

        store.remove_tree_snapshot(_identity("root"))

        assert store.load_conversation_snapshot().get_tree(_identity("root")) is None


class TestSessionStoreAtomicWrites:
    """Atomic persistence: failed replace must not truncate the prior file."""

    def test_failed_replace_keeps_prior_bytes_and_marks_dirty(self, tmp_path):
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)
        store.save_tree_snapshot(
            TreeSnapshot(scope=TELEGRAM_C1, root_id="r1", nodes={"r1": {}})
        )
        store.flush_pending_save()
        with open(path, encoding="utf-8") as f:
            disk_after_first = f.read()

        store.save_tree_snapshot(
            TreeSnapshot(scope=TELEGRAM_C1, root_id="r2", nodes={"r2": {}})
        )

        with patch(
            "free_claude_code.messaging.session.persistence.os.replace",
            side_effect=OSError("replace failed"),
        ):
            store.flush_pending_save()

        with open(path, encoding="utf-8") as f:
            disk_after_failed = f.read()
        assert disk_after_failed == disk_after_first
        assert store.dirty is True
        assert store.load_conversation_snapshot().get_tree(_identity("r2")) is not None


class TestSessionStoreClearAll:
    def test_clear_all_wipes_state_and_persists(self, tmp_path):
        path = str(tmp_path / "sessions.json")
        store = SessionStore(storage_path=path)

        store.save_tree_snapshot(
            TreeSnapshot(
                scope=TELEGRAM_C1,
                root_id="root1",
                nodes={
                    "root1": {
                        "node_id": "root1",
                        "incoming": {
                            "text": "hello",
                            "chat_id": "c1",
                            "user_id": "u1",
                            "message_id": "m1",
                            "platform": "telegram",
                            "reply_to_message_id": None,
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
            )
        )

        store.clear_all()

        assert store.load_conversation_snapshot().is_empty

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["conversation"]["trees"] == []
        assert data["message_log"] == {}

        store2 = SessionStore(storage_path=path)
        assert store2.load_conversation_snapshot().is_empty

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
