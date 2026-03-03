import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from messaging.models import IncomingMessage


@pytest.fixture
def messaging_platform():
    mp = MagicMock()
    mp.queue_send_message = AsyncMock()
    mp.queue_edit_message = AsyncMock()
    mp.queue_delete_message = AsyncMock()
    mp.cancel_pending_voice = AsyncMock(return_value=None)
    mp.record_message_id = MagicMock()

    def _fire_and_forget(task):
        if asyncio.iscoroutine(task):
            return asyncio.create_task(task)
        return None

    mp.fire_and_forget = MagicMock(side_effect=_fire_and_forget)
    return mp


@pytest.fixture
def cli_manager():
    cm = MagicMock()
    cm.get_or_create_session = AsyncMock()
    cm.register_real_session_id = AsyncMock(return_value=True)
    cm.stop_all = AsyncMock()
    cm.remove_session = AsyncMock(return_value=True)
    cm.get_stats = MagicMock(return_value={"active_sessions": 0})
    return cm


@pytest.fixture
def session_store():
    ss = MagicMock()
    ss.save_tree = MagicMock()
    ss.clear_all = MagicMock()
    ss.get_message_ids_for_chat = MagicMock(return_value=[])
    ss.remove_node_mappings = MagicMock()
    ss.remove_tree = MagicMock()
    ss.record_message_id = MagicMock()
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
            **kwargs,
        )

    return _factory
