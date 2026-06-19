import asyncio
import contextlib
import logging
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from config.settings import Settings

# Set mock environment BEFORE any imports that use Settings
os.environ.setdefault("NVIDIA_NIM_API_KEY", "test_key")
os.environ.setdefault("MODEL", "nvidia_nim/test-model")
os.environ["PTB_TIMEDELTA"] = "1"
# Ensure tests don't pick up a server API key from the repo .env
# (tests expect endpoints to be unauthenticated by default)
os.environ["ANTHROPIC_AUTH_TOKEN"] = ""

Settings.model_config = {**Settings.model_config, "env_file": None}


@pytest.fixture(autouse=True)
def _isolate_from_dotenv(monkeypatch):
    """Prevent Pydantic BaseSettings from reading the .env file during tests."""
    monkeypatch.setattr(
        Settings, "model_config", {**Settings.model_config, "env_file": None}
    )


@pytest.fixture
def provider_config():
    from providers.base import ProviderConfig

    return ProviderConfig(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture
def nim_provider(provider_config):
    from config.nim import NimSettings
    from providers.nvidia_nim import NvidiaNimProvider

    return NvidiaNimProvider(provider_config, nim_settings=NimSettings())


@pytest.fixture
def open_router_provider(provider_config):
    from providers.open_router import OpenRouterProvider

    return OpenRouterProvider(provider_config)


@pytest.fixture
def lmstudio_provider(provider_config):
    from providers.base import ProviderConfig
    from providers.lmstudio import LMStudioProvider

    lmstudio_config = ProviderConfig(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
        rate_limit=provider_config.rate_limit,
        rate_window=provider_config.rate_window,
    )
    return LMStudioProvider(lmstudio_config)


@pytest.fixture
def llamacpp_provider(provider_config):
    from providers.base import ProviderConfig
    from providers.llamacpp import LlamaCppProvider

    llamacpp_config = ProviderConfig(
        api_key="llamacpp",
        base_url="http://localhost:8080/v1",
        rate_limit=10,
        rate_window=60,
    )
    return LlamaCppProvider(llamacpp_config)


@pytest.fixture
def mock_cli_session():
    from messaging.managed_protocols import ManagedClaudeSessionProtocol

    session = MagicMock(spec=ManagedClaudeSessionProtocol)
    session.start_task = MagicMock()  # This will return an async generator
    session.is_busy = False
    return session


@pytest.fixture
def mock_cli_manager():
    from messaging.managed_protocols import ManagedClaudeSessionManagerProtocol

    manager = MagicMock(spec=ManagedClaudeSessionManagerProtocol)
    manager.get_or_create_session = AsyncMock()
    manager.register_real_session_id = AsyncMock(return_value=True)
    manager.stop_all = AsyncMock()
    manager.remove_session = AsyncMock(return_value=True)
    manager.get_stats = MagicMock(return_value={"active_sessions": 0})
    return manager


@pytest.fixture
def mock_platform():
    from messaging.platforms.ports import OutboundMessenger

    platform = MagicMock(spec=OutboundMessenger)
    platform.send_message = AsyncMock(return_value="msg_123")
    platform.edit_message = AsyncMock()
    platform.delete_message = AsyncMock()
    platform.queue_send_message = AsyncMock(return_value="msg_123")
    platform.queue_edit_message = AsyncMock()
    platform.queue_delete_message = AsyncMock()

    async def _queue_delete_messages(
        chat_id: str, message_ids: list[str], *, fire_and_forget: bool = True
    ) -> None:
        qdm = platform.queue_delete_message
        for mid in message_ids:
            await qdm(chat_id, mid, fire_and_forget=fire_and_forget)

    platform.queue_delete_messages = AsyncMock(side_effect=_queue_delete_messages)
    platform.cancel_pending_voice = AsyncMock(return_value=None)

    def _fire_and_forget(task):
        if asyncio.iscoroutine(task):
            # Create a task to avoid "coroutine was never awaited" warning
            return asyncio.create_task(task)
        return None

    platform.fire_and_forget = MagicMock(side_effect=_fire_and_forget)
    return platform


@pytest.fixture
def mock_session_store():
    from messaging.session import SessionStore

    store = MagicMock(spec=SessionStore)
    store.save_tree = MagicMock()
    store.get_tree = MagicMock(return_value=None)
    store.register_node = MagicMock()
    store.clear_all = MagicMock()
    store.record_message_id = MagicMock()
    store.get_message_ids_for_chat = MagicMock(return_value=[])
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
        from messaging.models import IncomingMessage

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
    """Route loguru logs to stdlib logging so pytest caplog captures them."""
    from loguru import logger as loguru_logger

    class _PropagateHandler:
        def write(self, message):
            record = message.record
            level = record["level"].no
            stdlib_level = min(level, logging.CRITICAL)
            py_logger = logging.getLogger(record["name"])
            py_logger.log(stdlib_level, record["message"])

    handler_id = loguru_logger.add(_PropagateHandler(), format="{message}")
    yield
    with contextlib.suppress(ValueError):
        loguru_logger.remove(
            handler_id
        )  # Handler already removed (e.g. by test_logging_config)
