"""Tests for messaging/ module."""

import pytest
import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

# --- Existing Tests ---


class TestMessagingModels:
    """Test messaging models."""

    def test_incoming_message_creation(self):
        """Test IncomingMessage dataclass."""
        from messaging.models import IncomingMessage

        msg = IncomingMessage(
            text="Hello",
            chat_id="123",
            user_id="456",
            message_id="789",
            platform="telegram",
        )
        assert msg.text == "Hello"
        assert msg.chat_id == "123"
        assert msg.platform == "telegram"
        assert msg.is_reply() is False

    def test_incoming_message_with_reply(self):
        """Test IncomingMessage as a reply."""
        from messaging.models import IncomingMessage

        msg = IncomingMessage(
            text="Reply text",
            chat_id="123",
            user_id="456",
            message_id="789",
            platform="discord",
            reply_to_message_id="100",
        )
        assert msg.is_reply() is True
        assert msg.reply_to_message_id == "100"


class TestMessagingBase:
    """Test MessagingPlatform ABC."""

    def test_platform_is_abstract(self):
        """Verify MessagingPlatform cannot be instantiated."""
        from messaging.base import MessagingPlatform

        with pytest.raises(TypeError):
            MessagingPlatform()


class TestSessionStore:
    """Test SessionStore."""

    def test_session_store_init(self, tmp_path):
        """Test SessionStore initialization."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        assert store._sessions == {}

    def test_save_and_get_session(self, tmp_path):
        """Test saving and retrieving a session."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))

        store.save_session(
            session_id="sess_123",
            chat_id="chat_456",
            initial_msg_id="msg_789",
            platform="telegram",
        )

        # Retrieve by message
        found = store.get_session_by_msg("chat_456", "msg_789", "telegram")
        assert found == "sess_123"

        # Verify persistence file created
        assert os.path.exists(str(tmp_path / "sessions.json"))

    def test_update_last_message(self, tmp_path):
        """Test updating last message in session."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))

        store.save_session("sess_1", "chat_1", "msg_1", "telegram")
        store.update_last_message("sess_1", "msg_2")

        # Should find session by new message too
        found = store.get_session_by_msg("chat_1", "msg_2", "telegram")
        assert found == "sess_1"

        # Original message mapping should still work
        found_old = store.get_session_by_msg("chat_1", "msg_1", "telegram")
        assert found_old == "sess_1"

        # Verify record updated
        record = store.get_session_record("sess_1")
        assert record is not None
        assert record.last_msg_id == "msg_2"

    def test_update_last_message_unknown_session(self, tmp_path):
        """Test updating unknown session does nothing."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        store.update_last_message("unknown", "msg_x")
        # Should log warning but not crash

    def test_get_session_record(self, tmp_path):
        """Test getting full session record."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        store.save_session("sess_1", "chat_1", "msg_1", "telegram")

        record = store.get_session_record("sess_1")
        assert record is not None
        assert record.session_id == "sess_1"
        assert record.platform == "telegram"

    def test_session_not_found(self, tmp_path):
        """Test getting non-existent session returns None."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))

        found = store.get_session_by_msg("notexist", "notexist", "telegram")
        assert found is None

    def test_rename_session(self, tmp_path):
        """Test renaming a session."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        store.save_session("old_id", "c1", "m1", "telegram")
        store.update_last_message("old_id", "m2")

        success = store.rename_session("old_id", "new_id")
        assert success is True

        # Verify old id gone
        assert store.get_session_record("old_id") is None

        # Verify new id exists
        rec = store.get_session_record("new_id")
        assert rec is not None
        assert rec.session_id == "new_id"

        # Verify mappings point to new id
        assert store.get_session_by_msg("c1", "m1", "telegram") == "new_id"
        assert store.get_session_by_msg("c1", "m2", "telegram") == "new_id"

    def test_rename_unknown_session(self, tmp_path):
        """Test renaming unknown session fails."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        success = store.rename_session("unknown", "new")
        assert success is False

    def test_cleanup_old_sessions(self, tmp_path):
        """Test cleaning up expired sessions."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))

        # Create an old session manually
        old_date = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
        store.save_session("old_sess", "c_old", "m_old")
        # Manipulate the created_at directly
        store._sessions["old_sess"].created_at = old_date

        # Create a new session
        store.save_session("new_sess", "c_new", "m_new")

        # Cleanup
        removed = store.cleanup_old_sessions(max_age_days=30)
        assert removed == 1

        assert store.get_session_record("old_sess") is None
        assert store.get_session_by_msg("c_old", "m_old") is None
        assert store.get_session_record("new_sess") is not None

    def test_cleanup_old_sessions_invalid_date(self, tmp_path):
        """Test cleanup handles invalid date formats gracefully."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        store.save_session("bad_date_sess", "c", "m")
        store._sessions["bad_date_sess"].created_at = "not-a-date"

        # Should not crash
        store.cleanup_old_sessions(30)
        # Should still exist because parsing failed so it wasn't removed (or default behavior)
        # The code tries parsing, excepts, and continues, so it isn't removed.
        assert store.get_session_record("bad_date_sess") is not None

    # --- Tree Tests ---

    def test_save_and_get_tree(self, tmp_path):
        """Test saving and retrieving trees."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))

        tree_data = {
            "root": "r1",
            "nodes": {"r1": {"content": "root"}, "n1": {"content": "child"}},
        }
        store.save_tree("r1", tree_data)

        loaded = store.get_tree("r1")
        assert loaded == tree_data

        # Verify node mapping
        assert store.get_tree_root_for_node("r1") == "r1"
        assert store.get_tree_root_for_node("n1") == "r1"

        # Verify get_tree_by_node
        assert store.get_tree_by_node("n1") == tree_data
        assert store.get_tree_by_node("unknown") is None

    def test_update_tree_node(self, tmp_path):
        """Test updating a specific node in a tree."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))

        store.save_tree("r1", {"nodes": {"r1": {}}})

        # Add new node
        store.update_tree_node("r1", "n2", {"data": "test"})

        tree = store.get_tree("r1")
        assert tree is not None
        assert "n2" in tree["nodes"]
        assert tree["nodes"]["n2"]["data"] == "test"
        assert store.get_tree_root_for_node("n2") == "r1"

    def test_update_tree_node_unknown_tree(self, tmp_path):
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        store.update_tree_node("unknown_root", "n1", {})
        # Should not crash

    def test_register_node(self, tmp_path):
        """Test manual node registration."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))
        store.register_node("n_manual", "r_manual")
        assert store.get_tree_root_for_node("n_manual") == "r_manual"

    def test_cleanup_old_trees(self, tmp_path):
        """Test cleaning up expired trees."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))

        old_date = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()

        # Old tree
        store.save_tree(
            "old_root", {"nodes": {"old_root": {"created_at": old_date}, "child": {}}}
        )

        # New tree
        store.save_tree(
            "new_root",
            {
                "nodes": {
                    "new_root": {"created_at": datetime.now(timezone.utc).isoformat()}
                }
            },
        )

        removed = store.cleanup_old_trees(30)
        assert removed == 1

        assert store.get_tree("old_root") is None
        assert (
            store.get_tree_root_for_node("child") is None
        )  # Node mapping should be gone
        assert store.get_tree("new_root") is not None

    # --- Persistence & Edge Cases ---

    def test_load_existing_legacy_format(self, tmp_path):
        """Test loading legacy session format (int IDs)."""
        from messaging.session import SessionStore

        data = {
            "sessions": {
                "s1": {
                    "session_id": "s1",
                    "chat_id": 123,  # Legacy int
                    "initial_msg_id": 100,  # Legacy int
                    "last_msg_id": 101,  # Legacy int
                    "created_at": "2024-01-01",
                    "updated_at": "2024-01-01",
                    # platform missing -> should default to telegram
                }
            }
        }

        p = tmp_path / "sessions.json"
        with open(p, "w") as f:
            json.dump(data, f)

        store = SessionStore(storage_path=str(p))
        rec = store.get_session_record("s1")
        assert rec is not None

        assert rec.chat_id == "123"  # Converted to str
        assert rec.platform == "telegram"  # Defaulted
        assert store.get_session_by_msg("123", "100", "telegram") == "s1"

    def test_load_corrupt_file(self, tmp_path):
        """Test loading corrupt/invalid json file."""
        p = tmp_path / "sessions.json"
        with open(p, "w") as f:
            f.write("{invalid json")

        from messaging.session import SessionStore

        # Should log error and start empty, avoiding crash
        store = SessionStore(storage_path=str(p))
        assert store._sessions == {}

    def test_save_error_handling(self, tmp_path):
        """Test error during save."""
        from messaging.session import SessionStore

        store = SessionStore(storage_path=str(tmp_path / "sessions.json"))

        # Mock open to raise exception
        with patch("builtins.open", side_effect=IOError("Disk full")):
            store.save_session("s1", "c1", "m1")

        # Should log error but not crash. Session should be in memory though.
        assert "s1" in store._sessions


class TestTreeQueueManager:
    """Test TreeQueueManager."""

    def test_tree_queue_manager_init(self):
        """Test TreeQueueManager initialization."""
        from messaging.tree_queue import TreeQueueManager

        mgr = TreeQueueManager()
        assert mgr._trees == {}

    def test_tree_not_busy_initially(self):
        """Test tree is not busy when no messages."""
        from messaging.tree_queue import TreeQueueManager

        mgr = TreeQueueManager()
        assert mgr.is_tree_busy("nonexistent") is False

    def test_get_queue_size_empty(self):
        """Test queue size is 0 for non-existent node."""
        from messaging.tree_queue import TreeQueueManager

        mgr = TreeQueueManager()
        assert mgr.get_queue_size("nonexistent") == 0

    @pytest.mark.asyncio
    async def test_create_tree_and_enqueue(self):
        """Test creating a tree and enqueueing."""
        from messaging.tree_queue import TreeQueueManager
        from messaging.models import IncomingMessage

        mgr = TreeQueueManager()
        processed = []

        async def processor(node_id, node):
            processed.append(node_id)

        incoming = IncomingMessage(
            text="test", chat_id="1", user_id="1", message_id="1", platform="test"
        )

        tree = await mgr.create_tree("1", incoming, "status_1")
        was_queued = await mgr.enqueue("1", processor)

        # First message should process immediately, not queue
        assert was_queued is False

    def test_cancel_tree_empty(self):
        """Test cancelling non-existent tree."""
        from messaging.tree_queue import TreeQueueManager

        mgr = TreeQueueManager()
        cancelled = mgr.cancel_tree("nonexistent")
        assert cancelled == []
