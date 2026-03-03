import asyncio
import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# Set mock environment BEFORE any imports that use Settings
os.environ.setdefault("NVIDIA_NIM_API_KEY", "test_key")
os.environ.setdefault("MODEL", "nvidia_nim/test-model")
os.environ["PTB_TIMEDELTA"] = "1"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.nim import NimSettings
from messaging.models import IncomingMessage
from messaging.platforms.base import (
    CLISession,
    MessagingPlatform,
    SessionManagerInterface,
)
from messaging.session import SessionStore
from providers.base import ProviderConfig
from providers.nvidia_nim import NvidiaNimProvider


@pytest.fixture
def provider_config():
    return ProviderConfig(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def nim_provider(provider_config):
    return NvidiaNimProvider(provider_config, nim_settings=NimSettings())


@pytest.fixture
def open_router_provider(provider_config):
    from providers.open_router import OpenRouterProvider

    return OpenRouterProvider(provider_config)


@pytest.fixture
def lmstudio_provider(provider_config):
    from providers.lmstudio import LMStudioProvider

    lmstudio_config = ProviderConfig(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
        rate_limit=provider_config.rate_limit,
        rate_window=provider_config.rate_window,
    )
    return LMStudioProvider(lmstudio_config)


@pytest.fixture
def mock_cli_session():
    session = MagicMock(spec=CLISession)
    session.start_task = MagicMock()
    session.is_busy = False
    return session


@pytest.fixture
def mock_cli_manager():
    manager = MagicMock(spec=SessionManagerInterface)
    manager.get_or_create_session = AsyncMock()
    manager.register_real_session_id = AsyncMock(return_value=True)
    manager.stop_all = AsyncMock()
    manager.remove_session = AsyncMock(return_value=True)
    manager.get_stats = MagicMock(return_value={"active_sessions": 0})
    return manager


@pytest.fixture
def mock_platform():
    platform = MagicMock(spec=MessagingPlatform)
    platform.send_message = AsyncMock(return_value="msg_123")
    platform.edit_message = AsyncMock()
    platform.delete_message = AsyncMock()
    platform.queue_send_message = AsyncMock(return_value="msg_123")
    platform.queue_edit_message = AsyncMock()
    platform.queue_delete_message = AsyncMock()
    platform.cancel_pending_voice = AsyncMock(return_value=None)

    def _fire_and_forget(task):
        if asyncio.iscoroutine(task):
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
    store.clear_all = MagicMock()
    store.record_message_id = MagicMock()
    store.get_message_ids_for_chat = MagicMock(return_value=[])
    store.remove_node_mappings = MagicMock()
    store.remove_tree = MagicMock()
    return store


@pytest.fixture
def incoming_message_factory():
    _valid_keys = frozenset(
        {
            "text",
            "chat_id",
            "user_id",
            "message_id",
            "platform",
            "reply_to_message_id",
            "message_thread_id",
            "username",
            "timestamp",
            "raw_event",
            "status_message_id",
        }
    )

    def _create(**kwargs):
        defaults: dict[str, Any] = {
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
        filtered = {k: v for k, v in defaults.items() if k in _valid_keys}
        return IncomingMessage(**filtered)

    return _create


@pytest.fixture(autouse=True)
def _propagate_loguru_to_caplog():
    """Route loguru logs to pytest's LogCaptureHandler without recursion."""
    from loguru import logger as loguru_logger

    class _PropagateHandler:
        _in_write = False

        def write(self, message):
            if self._in_write:
                return
            self._in_write = True
            record = message.record
            try:
                level = min(record["level"].no, logging.CRITICAL)
                log_record = logging.LogRecord(
                    name=record["name"],
                    level=level,
                    pathname=record["file"].path,
                    lineno=record["line"],
                    msg=record["message"],
                    args=(),
                    exc_info=None,
                )
                root_logger = logging.getLogger()
                for handler in root_logger.handlers:
                    if handler.__class__.__name__ == "LogCaptureHandler":
                        handler.handle(log_record)
            finally:
                self._in_write = False

    handler_id = loguru_logger.add(_PropagateHandler(), format="{message}")
    yield
    with contextlib.suppress(ValueError):
        loguru_logger.remove(handler_id)
