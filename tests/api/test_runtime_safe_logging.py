"""Tests for safe default logging in :mod:`api.runtime`."""

import importlib
import logging
from unittest.mock import MagicMock, patch

import pytest

from tests.api.test_app_lifespan_and_errors import _app_settings


@pytest.mark.asyncio
async def test_messaging_start_failure_default_logs_exclude_traceback(caplog):
    api_runtime_mod = importlib.import_module("api.runtime")
    settings = _app_settings(
        messaging_platform="telegram",
        telegram_bot_token="t",
        allowed_telegram_user_id="1",
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir="",
        claude_workspace="./agent_workspace",
        host="127.0.0.1",
        port=8082,
        log_api_error_tracebacks=False,
    )
    runtime = api_runtime_mod.AppRuntime(app=MagicMock(), settings=settings)

    with (
        patch(
            "messaging.platforms.factory.create_messaging_components",
            side_effect=RuntimeError("SECRET_RUNTIME_DETAIL"),
        ),
        caplog.at_level(logging.ERROR),
    ):
        await runtime._start_messaging_if_configured()

    blob = " | ".join(r.getMessage() for r in caplog.records)
    assert "SECRET_RUNTIME_DETAIL" not in blob
    assert "exc_type=RuntimeError" in blob


@pytest.mark.asyncio
async def test_best_effort_default_logs_exclude_exception_text(caplog):
    api_runtime_mod = importlib.import_module("api.runtime")

    async def boom():
        raise ValueError("SECRET_SHUTDOWN")

    with caplog.at_level(logging.WARNING):
        await api_runtime_mod.best_effort("test_step", boom(), log_verbose_errors=False)

    blob = " | ".join(r.getMessage() for r in caplog.records)
    assert "SECRET_SHUTDOWN" not in blob
    assert "exc_type=ValueError" in blob


@pytest.mark.asyncio
async def test_best_effort_verbose_includes_exception_text(caplog):
    api_runtime_mod = importlib.import_module("api.runtime")

    async def boom():
        raise ValueError("VISIBLE_SHUTDOWN")

    with caplog.at_level(logging.WARNING):
        await api_runtime_mod.best_effort("test_step", boom(), log_verbose_errors=True)

    blob = " | ".join(r.getMessage() for r in caplog.records)
    assert "VISIBLE_SHUTDOWN" in blob
