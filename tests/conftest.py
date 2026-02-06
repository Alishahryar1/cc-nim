import pytest
import asyncio
import os
import sys

# Set mock environment BEFORE any imports that use Settings
os.environ.setdefault("NVIDIA_NIM_API_KEY", "test_key")
os.environ.setdefault("MODEL", "test-model")
os.environ["PTB_TIMEDELTA"] = "1"

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import AsyncMock, MagicMock
from providers.base import ProviderConfig
from providers.nvidia_nim import NvidiaNimProvider
from config.nim import NimSettings
from messaging.base import CLISession, SessionManagerInterface, MessagingPlatform
from messaging.models import IncomingMessage
from messaging.session import SessionStore


@pytest.fixture
def provider_config():
    return ProviderConfig(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        rate_limit=10,
        rate_window=60,
        nim_settings=NimSettings(),
    )


@pytest.fixture
def nim_provider(provider_config):
    return NvidiaNimProvider(provider_config)


@pytest.fixture
def mock_cli_session():
    session = MagicMock(spec=CLISession)
    session.start_task = MagicMock()  # This will return an async generator
    session.is_busy = False
    return session


@pytest.fixture
def mock_cli_manager():
    manager = MagicMock(spec=SessionManagerInterface)
    manager.get_or_create_session = AsyncMock()
    manager.register_real_session_id = AsyncMock(return_value=True)
    manager.stop_all = AsyncMock()
    manager.get_stats = MagicMock(
        return_value={"active_sessions": 0, "max_sessions": 5}
    )
    return manager


@pytest.fixture
def mock_platform():
    platform = MagicMock(spec=MessagingPlatform)
    platform.send_message = AsyncMock(return_value="msg_123")
    platform.edit_message = AsyncMock()
    platform.queue_send_message = AsyncMock(return_value="msg_123")
    platform.queue_edit_message = AsyncMock()

    def _fire_and_forget(task):
        if asyncio.iscoroutine(task):
            # Create a task to avoid "coroutine was never awaited" warning
            return asyncio.create_task(task)
        return None

    platform.fire_and_forget = MagicMock(side_effect=_fire_and_forget)
    return platform


@pytest.fixture
def mock_session_store():
    store = MagicMock(spec=SessionStore)
    store.save_tree = MagicMock()
    store.get_tree = MagicMock(return_value=None)
    store.register_node = MagicMock()
    return store


@pytest.fixture
def incoming_message_factory():
    def _create(**kwargs):
        defaults = {
            "text": "hello",
            "chat_id": "chat_1",
            "user_id": "user_1",
            "message_id": "msg_1",
            "platform": "telegram",
        }
        defaults.update(kwargs)
        if "timestamp" in defaults and isinstance(defaults["timestamp"], str):
            from datetime import datetime

            defaults["timestamp"] = datetime.fromisoformat(defaults["timestamp"])
        return IncomingMessage(**defaults)  # type: ignore

    return _create
