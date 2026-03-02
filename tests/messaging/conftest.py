import pytest
from unittest.mock import AsyncMock, MagicMock

from messaging.models import IncomingMessage


@pytest.fixture
def messaging_platform():
    mp = MagicMock()
    mp.queue_send_message = AsyncMock()
    mp.queue_edit_message = AsyncMock()
    mp.queue_delete_message = AsyncMock()
    mp.cancel_pending_voice = AsyncMock()
    mp.record_message_id = AsyncMock()
    mp.fire_and_forget = AsyncMock()
    return mp


@pytest.fixture
def cli_manager():
    cm = MagicMock()
    cm.get_or_create_session = AsyncMock()
    cm.stop_all = AsyncMock()
    cm.get_stats = AsyncMock()
    return cm


@pytest.fixture
def session_store():
    ss = MagicMock()
    ss.save_tree = AsyncMock()
    ss.clear_all = AsyncMock()
    ss.get_message_ids_for_chat = AsyncMock(return_value=[])
    ss.remove_node_mappings = AsyncMock()
    ss.remove_tree = AsyncMock()
    ss.record_message_id = AsyncMock()
    return ss


@pytest.fixture
def message_queue():
    return MagicMock()


@pytest.fixture
def voice_processor():
    return MagicMock()


@pytest.fixture
def incoming_message_factory():
    def _factory(**kwargs):
        # Required fields with defaults
        text = kwargs.pop("text", "test text")
        chat_id = kwargs.pop("chat_id", "test_chat")
        user_id = kwargs.pop("user_id", "test_user")
        message_id = kwargs.pop("message_id", "test_msg")
        platform = kwargs.pop("platform", "telegram")
        return IncomingMessage(
            text=text,
            chat_id=chat_id,
            user_id=user_id,
            message_id=message_id,
            platform=platform,
            **kwargs
        )
    return _factory
