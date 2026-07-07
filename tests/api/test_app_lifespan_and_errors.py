import importlib
from collections.abc import MutableMapping
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config.settings import Settings
from messaging.trees import ConversationSnapshot, TreeSnapshot
from providers.exceptions import ServiceUnavailableError
from providers.runtime import ProviderRuntime

_RUNTIME_EXTRAS = {
    "model": "nvidia_nim/test-model",
    "model_opus": None,
    "model_sonnet": None,
    "model_haiku": None,
    "telegram_proxy_url": "",
    "voice_note_enabled": True,
    "whisper_model": "base",
    "whisper_device": "cpu",
    "huggingface_api_key": "",
    "nvidia_nim_api_key": "",
    "messaging_rate_limit": 1,
    "messaging_rate_window": 1.0,
    "max_message_log_entries_per_chat": None,
    "debug_platform_edits": False,
    "debug_subagent_stack": False,
    "log_api_error_tracebacks": False,
    "log_raw_messaging_content": False,
    "log_raw_cli_diagnostics": False,
    "log_messaging_error_details": False,
}


def _app_settings(**kwargs):
    """Minimal settings namespace for AppRuntime (matches typed :class:`Settings` fields used)."""
    data = {**_RUNTIME_EXTRAS, **kwargs}
    return SimpleNamespace(**data)


def _fake_messaging_components(runtime: MagicMock | None = None) -> SimpleNamespace:
    runtime = runtime or MagicMock()
    runtime.name = getattr(runtime, "name", "fake")
    runtime.on_message = getattr(runtime, "on_message", MagicMock())
    runtime.start = getattr(runtime, "start", AsyncMock())
    runtime.stop = getattr(runtime, "stop", AsyncMock())
    outbound = MagicMock()
    outbound.queue_send_message = AsyncMock(return_value="msg")
    outbound.queue_edit_message = AsyncMock()
    outbound.queue_delete_messages = AsyncMock()
    outbound.fire_and_forget = MagicMock()
    return SimpleNamespace(
        name=runtime.name,
        runtime=runtime,
        outbound=outbound,
        voice_cancellation=None,
    )


@pytest.fixture(autouse=True)
def _redirect_fcc_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


def test_warn_if_process_auth_token_logs_warning(monkeypatch):
    api_runtime_mod = importlib.import_module("api.runtime")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "process-token")
    monkeypatch.setitem(Settings.model_config, "env_file", ())
    settings = Settings.model_construct()

    with patch.object(api_runtime_mod.logger, "warning") as warning:
        api_runtime_mod.warn_if_process_auth_token(settings)

    warning.assert_called_once()
    assert "ANTHROPIC_AUTH_TOKEN" in warning.call_args.args[0]


def test_warn_if_process_auth_token_skips_explicit_dotenv_config(monkeypatch, tmp_path):
    api_runtime_mod = importlib.import_module("api.runtime")
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_AUTH_TOKEN=\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "process-token")
    monkeypatch.setitem(Settings.model_config, "env_file", (env_file,))
    settings = Settings.model_construct()

    with patch.object(api_runtime_mod.logger, "warning") as warning:
        api_runtime_mod.warn_if_process_auth_token(settings)

    warning.assert_not_called()


@pytest.mark.asyncio
async def test_runtime_startup_logs_admin_url_without_printed_server_banner(tmp_path):
    import api.runtime as api_runtime_mod

    settings = _app_settings(
        messaging_platform="none",
        telegram_bot_token=None,
        allowed_telegram_user_id=None,
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir=str(tmp_path / "workspace"),
        claude_workspace=str(tmp_path / "data"),
        host="127.0.0.1",
        port=9099,
    )
    runtime = api_runtime_mod.AppRuntime(
        app=FastAPI(), settings=cast(Settings, settings)
    )
    uvicorn_logger = MagicMock()

    with (
        patch("builtins.print") as printed,
        patch.object(
            api_runtime_mod.logging, "getLogger", return_value=uvicorn_logger
        ) as get_logger,
        patch.object(api_runtime_mod.logger, "info") as app_info,
        patch.object(ProviderRuntime, "validate_configured_models", new=AsyncMock()),
        patch.object(ProviderRuntime, "start_model_list_refresh"),
        patch.object(ProviderRuntime, "cleanup", new=AsyncMock()),
        patch(
            "messaging.platforms.factory.create_messaging_components",
            return_value=None,
        ),
    ):
        await runtime.startup()
        await runtime.shutdown()

    printed.assert_not_called()
    get_logger.assert_called_with("uvicorn.error")
    uvicorn_logger.info.assert_called_once_with(
        "Admin UI: %s (local-only)",
        "http://127.0.0.1:9099/admin",
    )
    logged = " ".join(str(arg) for call in app_info.call_args_list for arg in call.args)
    assert "Server URL:" not in logged


def test_create_app_provider_error_handler_returns_anthropic_format():
    from api.app import create_app
    from providers.exceptions import AuthenticationError

    app = create_app()

    @app.get("/raise_provider")
    async def _raise_provider():
        raise AuthenticationError("bad key")

    api_app_mod = importlib.import_module("api.app")
    settings = _app_settings(
        messaging_platform="telegram",
        telegram_bot_token=None,
        allowed_telegram_user_id=None,
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir="",
        claude_workspace="./agent_workspace",
        host="127.0.0.1",
        port=8082,
    )
    with (
        patch.object(api_app_mod, "get_settings", return_value=settings),
        patch.object(ProviderRuntime, "cleanup", new=AsyncMock()),
    ):
        with TestClient(app) as client:
            resp = client.get("/raise_provider")
        assert resp.status_code == 401
    body = resp.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "authentication_error"


def test_create_app_provider_error_default_logs_exclude_provider_message():
    """Provider errors must not log exc.message by default."""
    from api.app import create_app
    from providers.exceptions import AuthenticationError

    app = create_app()
    secret = "provider-upstream-secret-detail"

    @app.get("/raise_provider_secret")
    async def _raise():
        raise AuthenticationError(secret)

    api_app_mod = importlib.import_module("api.app")
    settings = _app_settings(
        messaging_platform="telegram",
        telegram_bot_token=None,
        allowed_telegram_user_id=None,
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir="",
        claude_workspace="./agent_workspace",
        host="127.0.0.1",
        port=8082,
        log_api_error_tracebacks=False,
    )
    with (
        patch.object(api_app_mod, "get_settings", return_value=settings),
        patch.object(ProviderRuntime, "cleanup", new=AsyncMock()),
        patch.object(api_app_mod.logger, "error") as log_err,
    ):
        with TestClient(app) as client:
            resp = client.get("/raise_provider_secret")
        assert resp.status_code == 401

    blob = " ".join(str(a) for c in log_err.call_args_list for a in c.args)
    blob += repr([c.kwargs for c in log_err.call_args_list])
    assert secret not in blob
    assert "authentication_error" in blob


def test_create_app_general_exception_handler_returns_500():
    from api.app import create_app

    app = create_app()

    @app.get("/raise_general")
    async def _raise_general():
        raise RuntimeError("boom")

    api_app_mod = importlib.import_module("api.app")
    settings = _app_settings(
        messaging_platform="telegram",
        telegram_bot_token=None,
        allowed_telegram_user_id=None,
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir="",
        claude_workspace="./agent_workspace",
        host="127.0.0.1",
        port=8082,
    )
    with (
        patch.object(api_app_mod, "get_settings", return_value=settings),
        patch.object(ProviderRuntime, "cleanup", new=AsyncMock()),
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/raise_general")
        assert resp.status_code == 500
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "api_error"


def test_create_app_general_exception_default_logs_exclude_exception_message():
    """Unhandled errors must not log exception text by default (may echo user content)."""
    from api.app import create_app

    app = create_app()

    secret = "user-provided-secret-token-xyzzy"

    @app.get("/raise_secret")
    async def _raise_secret():
        raise ValueError(secret)

    api_app_mod = importlib.import_module("api.app")
    settings = _app_settings(
        messaging_platform="telegram",
        telegram_bot_token=None,
        allowed_telegram_user_id=None,
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir="",
        claude_workspace="./agent_workspace",
        host="127.0.0.1",
        port=8082,
        log_api_error_tracebacks=False,
    )
    with (
        patch.object(api_app_mod, "get_settings", return_value=settings),
        patch.object(ProviderRuntime, "cleanup", new=AsyncMock()),
        patch.object(api_app_mod.logger, "error") as log_err,
    ):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/raise_secret")
        assert resp.status_code == 500

    flattened: list[str] = []
    for call in log_err.call_args_list:
        flattened.extend(str(arg) for arg in call.args)
        flattened.append(repr(call.kwargs))
    blob = " ".join(flattened)
    assert secret not in blob
    assert "ValueError" in blob


@pytest.mark.parametrize(
    "messaging_enabled", [True, False], ids=["with_platform", "no_platform"]
)
def test_app_lifespan_sets_state_and_cleans_up(tmp_path, messaging_enabled):
    from api.app import create_app

    app = create_app()

    settings = _app_settings(
        messaging_platform="telegram",
        telegram_bot_token="token" if messaging_enabled else None,
        allowed_telegram_user_id="123",
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir=str(tmp_path / "workspace"),
        claude_workspace=str(tmp_path / "data"),
        host="127.0.0.1",
        port=8082,
    )

    fake_platform = MagicMock()
    fake_platform.name = "fake"
    fake_platform.on_message = MagicMock()
    fake_platform.start = AsyncMock()
    fake_platform.stop = AsyncMock()
    fake_components = _fake_messaging_components(fake_platform)

    snapshot = (
        ConversationSnapshot(trees={"t": TreeSnapshot(root_id="t", nodes={})})
        if messaging_enabled
        else ConversationSnapshot()
    )
    session_store = MagicMock()
    session_store.load_conversation_snapshot.return_value = snapshot
    session_store.save_conversation_snapshot = MagicMock()

    fake_queue = MagicMock()
    fake_queue.cleanup_stale_nodes.return_value = 1
    fake_queue.snapshot.return_value = snapshot

    cli_manager = MagicMock()
    cli_manager.stop_all = AsyncMock()

    api_app_mod = importlib.import_module("api.app")

    runtime_cleanup = AsyncMock()
    with (
        patch.object(api_app_mod, "get_settings", return_value=settings),
        patch.object(ProviderRuntime, "cleanup", new=runtime_cleanup),
        patch(
            "messaging.platforms.factory.create_messaging_components",
            return_value=fake_components if messaging_enabled else None,
        ) as create_components,
        patch("messaging.session.SessionStore", return_value=session_store),
        patch("cli.managed.ManagedClaudeSessionManager", return_value=cli_manager),
        patch(
            "messaging.trees.TreeQueueManager.from_snapshot",
            return_value=fake_queue,
        ),
        TestClient(app),
    ):
        pass

    if messaging_enabled:
        create_components.assert_called_once()
        fake_platform.on_message.assert_called_once()
        fake_platform.start.assert_awaited_once()
        fake_platform.stop.assert_awaited_once()
        cli_manager.stop_all.assert_awaited_once()
        assert getattr(app.state, "messaging_workflow", None) is not None
        session_store.save_conversation_snapshot.assert_called_once_with(snapshot)
    else:
        fake_platform.start.assert_not_awaited()
        fake_platform.stop.assert_not_awaited()
        cli_manager.stop_all.assert_not_awaited()
        assert getattr(app.state, "messaging_runtime", "missing") is None

    runtime_cleanup.assert_awaited_once()


def test_app_lifespan_cleanup_continues_if_platform_stop_raises(tmp_path):
    from api.app import create_app

    app = create_app()

    settings = _app_settings(
        messaging_platform="telegram",
        telegram_bot_token="token",
        allowed_telegram_user_id="123",
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir=str(tmp_path / "workspace"),
        claude_workspace=str(tmp_path / "data"),
        host="127.0.0.1",
        port=8082,
    )

    fake_platform = MagicMock()
    fake_platform.name = "fake"
    fake_platform.on_message = MagicMock()
    fake_platform.start = AsyncMock()
    fake_platform.stop = AsyncMock(side_effect=RuntimeError("stop failed"))
    fake_components = _fake_messaging_components(fake_platform)

    session_store = MagicMock()
    session_store.load_conversation_snapshot.return_value = ConversationSnapshot()

    cli_manager = MagicMock()
    cli_manager.stop_all = AsyncMock()

    api_app_mod = importlib.import_module("api.app")
    runtime_cleanup = AsyncMock()
    with (
        patch.object(api_app_mod, "get_settings", return_value=settings),
        patch.object(ProviderRuntime, "cleanup", new=runtime_cleanup),
        patch(
            "messaging.platforms.factory.create_messaging_components",
            return_value=fake_components,
        ),
        patch("messaging.session.SessionStore", return_value=session_store),
        patch("cli.managed.ManagedClaudeSessionManager", return_value=cli_manager),
        TestClient(app),
    ):
        pass

    fake_platform.stop.assert_awaited_once()
    cli_manager.stop_all.assert_awaited_once()
    runtime_cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_startup_validation_failure_does_not_block_server(tmp_path):
    import api.runtime as api_runtime_mod

    settings = _app_settings(
        messaging_platform="none",
        telegram_bot_token=None,
        allowed_telegram_user_id=None,
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir=str(tmp_path / "workspace"),
        claude_workspace=str(tmp_path / "data"),
        host="127.0.0.1",
        port=8082,
    )
    app = FastAPI()
    runtime = api_runtime_mod.AppRuntime(
        app=app,
        settings=cast(Settings, settings),
    )

    validation = AsyncMock(side_effect=ServiceUnavailableError("bad model"))
    cleanup = AsyncMock()
    with (
        patch.object(ProviderRuntime, "validate_configured_models", new=validation),
        patch.object(ProviderRuntime, "cleanup", new=cleanup),
        patch.object(api_runtime_mod.logger, "warning") as log_warning,
        patch(
            "messaging.platforms.factory.create_messaging_components",
            return_value=None,
        ) as create_components,
    ):
        await runtime.startup()
        await runtime.shutdown()

    validation.assert_awaited_once_with()
    cleanup.assert_awaited_once()
    create_components.assert_called_once()
    logged = " ".join(
        str(arg) for call in log_warning.call_args_list for arg in call.args
    )
    assert "validation failed" in logged
    assert "bad model" in logged
    assert "Traceback" not in logged
    assert app.state.startup_validation_error == "bad model"


@pytest.mark.asyncio
async def test_graceful_asgi_lifespan_model_validation_failure_starts(tmp_path):
    import api.app as api_app_mod

    settings = _app_settings(
        messaging_platform="none",
        telegram_bot_token=None,
        allowed_telegram_user_id=None,
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir=str(tmp_path / "workspace"),
        claude_workspace=str(tmp_path / "data"),
        host="127.0.0.1",
        port=8082,
    )
    app = api_app_mod.GracefulLifespanApp(FastAPI())
    sent: list[MutableMapping[str, Any]] = []
    received = [
        {"type": "lifespan.startup"},
        {"type": "lifespan.shutdown"},
    ]

    async def receive() -> MutableMapping[str, Any]:
        return received.pop(0)

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)

    validation = AsyncMock(side_effect=ServiceUnavailableError("bad model"))
    cleanup = AsyncMock()
    with (
        patch.object(api_app_mod, "get_settings", return_value=settings),
        patch.object(ProviderRuntime, "validate_configured_models", new=validation),
        patch.object(ProviderRuntime, "cleanup", new=cleanup),
    ):
        await app({"type": "lifespan"}, receive, send)

    assert sent == [
        {"type": "lifespan.startup.complete"},
        {"type": "lifespan.shutdown.complete"},
    ]


def test_app_lifespan_messaging_import_error_no_crash(tmp_path, caplog):
    """Messaging import failure logs warning and continues without crash."""
    from api.app import create_app

    app = create_app()

    settings = _app_settings(
        messaging_platform="telegram",
        telegram_bot_token="token",
        allowed_telegram_user_id="123",
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir=str(tmp_path / "workspace"),
        claude_workspace=str(tmp_path / "data"),
        host="127.0.0.1",
        port=8082,
    )

    api_app_mod = importlib.import_module("api.app")
    runtime_cleanup = AsyncMock()
    with (
        patch.object(api_app_mod, "get_settings", return_value=settings),
        patch.object(ProviderRuntime, "cleanup", new=runtime_cleanup),
        patch(
            "messaging.platforms.factory.create_messaging_components",
            side_effect=ImportError("discord not installed"),
        ),
        TestClient(app),
    ):
        pass

    assert getattr(app.state, "messaging_runtime", None) is None
    runtime_cleanup.assert_awaited_once()


def test_app_lifespan_platform_start_exception_cleanup_still_runs(tmp_path):
    """Exception during platform.start() logs error, cleanup still runs."""
    from api.app import create_app

    app = create_app()

    settings = _app_settings(
        messaging_platform="telegram",
        telegram_bot_token="token",
        allowed_telegram_user_id="123",
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir=str(tmp_path / "workspace"),
        claude_workspace=str(tmp_path / "data"),
        host="127.0.0.1",
        port=8082,
    )

    fake_platform = MagicMock()
    fake_platform.name = "fake"
    fake_platform.on_message = MagicMock()
    fake_platform.start = AsyncMock(side_effect=RuntimeError("start failed"))
    fake_platform.stop = AsyncMock()
    fake_components = _fake_messaging_components(fake_platform)

    session_store = MagicMock()
    session_store.load_conversation_snapshot.return_value = ConversationSnapshot()

    cli_manager = MagicMock()
    cli_manager.stop_all = AsyncMock()

    api_app_mod = importlib.import_module("api.app")
    runtime_cleanup = AsyncMock()
    with (
        patch.object(api_app_mod, "get_settings", return_value=settings),
        patch.object(ProviderRuntime, "cleanup", new=runtime_cleanup),
        patch(
            "messaging.platforms.factory.create_messaging_components",
            return_value=fake_components,
        ),
        patch("messaging.session.SessionStore", return_value=session_store),
        patch("cli.managed.ManagedClaudeSessionManager", return_value=cli_manager),
        TestClient(app),
    ):
        pass

    runtime_cleanup.assert_awaited_once()


def test_app_lifespan_flush_pending_save_exception_warning_only(tmp_path):
    """Session store flush exception on shutdown is logged as warning, no crash."""
    from api.app import create_app

    app = create_app()

    settings = _app_settings(
        messaging_platform="telegram",
        telegram_bot_token="token",
        allowed_telegram_user_id="123",
        discord_bot_token=None,
        allowed_discord_channels=None,
        allowed_dir=str(tmp_path / "workspace"),
        claude_workspace=str(tmp_path / "data"),
        host="127.0.0.1",
        port=8082,
    )

    fake_platform = MagicMock()
    fake_platform.name = "fake"
    fake_platform.on_message = MagicMock()
    fake_platform.start = AsyncMock()
    fake_platform.stop = AsyncMock()
    fake_components = _fake_messaging_components(fake_platform)

    session_store = MagicMock()
    session_store.load_conversation_snapshot.return_value = ConversationSnapshot()
    session_store.flush_pending_save = MagicMock(side_effect=OSError("disk full"))

    cli_manager = MagicMock()
    cli_manager.stop_all = AsyncMock()

    api_app_mod = importlib.import_module("api.app")
    runtime_cleanup = AsyncMock()
    with (
        patch.object(api_app_mod, "get_settings", return_value=settings),
        patch.object(ProviderRuntime, "cleanup", new=runtime_cleanup),
        patch(
            "messaging.platforms.factory.create_messaging_components",
            return_value=fake_components,
        ),
        patch("messaging.session.SessionStore", return_value=session_store),
        patch("cli.managed.ManagedClaudeSessionManager", return_value=cli_manager),
        TestClient(app),
    ):
        pass

    session_store.flush_pending_save.assert_called_once()
    runtime_cleanup.assert_awaited_once()


def test_create_app_writes_server_log_under_fcc_home(monkeypatch, tmp_path):
    """App logging uses ~/.fcc/logs/server.log regardless of cwd."""
    from loguru import logger

    import config.logging_config as logging_config_mod
    from api.app import create_app
    from config.paths import server_log_path

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(logging_config_mod, "_configured", False)

    create_app(lifespan_enabled=False)
    logger.info("canonical log path test")
    logger.complete()

    canonical_log = server_log_path()
    assert canonical_log == tmp_path / ".fcc" / "logs" / "server.log"
    assert canonical_log.is_file()
    assert "canonical log path test" in canonical_log.read_text(encoding="utf-8")
    assert not (run_dir / "logs" / "server.log").exists()


def test_create_app_honors_process_log_file_override(monkeypatch, tmp_path):
    """Smoke subprocesses can redirect app logs without changing Settings."""
    from loguru import logger

    import config.logging_config as logging_config_mod
    from api.app import create_app
    from config.paths import server_log_path

    custom_log = tmp_path / "smoke" / "server.log"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("LOG_FILE", str(custom_log))
    monkeypatch.setattr(logging_config_mod, "_configured", False)

    create_app(lifespan_enabled=False)
    logger.info("process log path test")
    logger.complete()

    assert custom_log.is_file()
    assert "process log path test" in custom_log.read_text(encoding="utf-8")
    assert not server_log_path().exists()
