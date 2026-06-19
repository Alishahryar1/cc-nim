"""Tests for cli/entrypoints.py — fcc-init scaffolding logic."""

import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from config.settings import Settings


def _launcher_settings(
    *,
    port: int = 8082,
    token: str = "freecc",
) -> Settings:
    return Settings.model_construct(
        host="0.0.0.0",
        port=port,
        anthropic_auth_token=token,
    )


def _run_init(tmp_home: Path) -> tuple[str, Path]:
    """Run init() with home directory redirected to tmp_home. Returns (printed output, env_file path)."""
    from cli.entrypoints import init

    env_file = tmp_home / ".fcc" / ".env"
    printed: list[str] = []

    with (
        patch("pathlib.Path.home", return_value=tmp_home),
        patch(
            "builtins.print",
            side_effect=lambda *a: printed.append(" ".join(str(x) for x in a)),
        ),
    ):
        init()

    return "\n".join(printed), env_file


def test_init_creates_env_file(tmp_path: Path) -> None:
    """init() creates .env from the bundled template when it doesn't exist yet."""
    output, env_file = _run_init(tmp_path)

    assert env_file.exists()
    assert env_file.stat().st_size > 0
    assert str(env_file) in output


def test_init_copies_template_content(tmp_path: Path) -> None:
    """init() writes the canonical root env.example content, not an empty file."""
    template = (Path(__file__).resolve().parents[2] / ".env.example").read_text(
        encoding="utf-8"
    )
    _, env_file = _run_init(tmp_path)

    assert env_file.read_text("utf-8") == template


def test_init_migrates_home_checkout_env_before_template(tmp_path: Path) -> None:
    """init() preserves users who kept config in ~/free-claude-code/.env."""
    legacy_env = tmp_path / "free-claude-code" / ".env"
    legacy_env.parent.mkdir(parents=True)
    legacy_env.write_text("MODEL=deepseek/deepseek-chat\n", encoding="utf-8")

    output, env_file = _run_init(tmp_path)

    assert env_file.read_text("utf-8") == "MODEL=deepseek/deepseek-chat\n"
    assert f"Config migrated from {legacy_env}" in output


def test_init_migrates_legacy_xdg_env_before_template(tmp_path: Path) -> None:
    """init() preserves users who kept config in ~/.config/free-claude-code/.env."""
    legacy_env = tmp_path / ".config" / "free-claude-code" / ".env"
    legacy_env.parent.mkdir(parents=True)
    legacy_env.write_text("MODEL=open_router/free-model\n", encoding="utf-8")

    output, env_file = _run_init(tmp_path)

    assert env_file.read_text("utf-8") == "MODEL=open_router/free-model\n"
    assert f"Config migrated from {legacy_env}" in output


def test_legacy_env_migration_does_not_overwrite_managed_env(
    tmp_path: Path,
) -> None:
    """Legacy migration never overwrites an existing ~/.fcc/.env."""
    from cli.entrypoints import _migrate_legacy_env_if_missing

    managed_env = tmp_path / ".fcc" / ".env"
    managed_env.parent.mkdir(parents=True)
    managed_env.write_text("MODEL=nvidia_nim/current\n", encoding="utf-8")
    legacy_env = tmp_path / "free-claude-code" / ".env"
    legacy_env.parent.mkdir(parents=True)
    legacy_env.write_text("MODEL=deepseek/legacy\n", encoding="utf-8")

    with patch("pathlib.Path.home", return_value=tmp_path):
        migrated_from = _migrate_legacy_env_if_missing()

    assert migrated_from is None
    assert managed_env.read_text("utf-8") == "MODEL=nvidia_nim/current\n"


def test_env_template_loader_uses_root_template_in_source_checkout() -> None:
    """Source checkout fallback uses the root .env.example as the single source."""
    from cli.entrypoints import _load_env_template

    template = (Path(__file__).resolve().parents[2] / ".env.example").read_text(
        encoding="utf-8"
    )

    assert _load_env_template() == template


def test_init_creates_parent_directories(tmp_path: Path) -> None:
    """init() creates ~/.fcc/ even if it doesn't exist."""
    config_dir = tmp_path / ".fcc"
    assert not config_dir.exists()

    _run_init(tmp_path)

    assert config_dir.is_dir()


def test_init_skips_if_env_already_exists(tmp_path: Path) -> None:
    """init() does not overwrite an existing .env and prints a warning."""
    # Create it first
    _run_init(tmp_path)

    env_file = tmp_path / ".fcc" / ".env"
    env_file.write_text("existing content", encoding="utf-8")

    output, _ = _run_init(tmp_path)

    assert env_file.read_text("utf-8") == "existing content"
    assert "already exists" in output


def test_init_prints_next_step_hint(tmp_path: Path) -> None:
    """init() tells the user to run fcc-server after editing .env."""
    output, _ = _run_init(tmp_path)

    assert "fcc-server" in output


def test_cli_scripts_are_registered() -> None:
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    scripts = pyproject["project"]["scripts"]
    assert scripts["fcc-server"] == "cli.entrypoints:serve"
    assert scripts["free-claude-code"] == "cli.entrypoints:serve"
    assert scripts["fcc-claude"] == "cli.entrypoints:launch_claude"


def test_schedule_open_admin_browser_opens_when_health_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opening /admin runs after /health preflight succeeds."""
    monkeypatch.delenv("FCC_OPEN_BROWSER", raising=False)
    from api.admin_urls import local_admin_url
    from cli import entrypoints

    settings = _launcher_settings(port=31337)
    opened_urls: list[str] = []

    class ImmediateThread:
        def __init__(self, target=None, **_kwargs: object) -> None:
            self._target = target

        def start(self) -> None:
            assert self._target is not None
            self._target()

    with (
        patch.object(entrypoints.threading, "Thread", ImmediateThread),
        patch.object(entrypoints, "_preflight_proxy", return_value=None),
        patch.object(
            entrypoints.webbrowser,
            "open",
            side_effect=lambda url: opened_urls.append(url),
        ),
        patch.object(entrypoints.time, "sleep"),
    ):
        entrypoints._schedule_open_admin_browser(settings)

    assert opened_urls == [local_admin_url(settings)]


def test_schedule_open_admin_browser_skips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FCC_OPEN_BROWSER", "0")
    from cli import entrypoints

    settings = _launcher_settings()

    with patch.object(entrypoints.threading, "Thread") as thread_cls:
        entrypoints._schedule_open_admin_browser(settings)

    thread_cls.assert_not_called()


def test_serve_supervisor_restarts_when_app_requests_restart() -> None:
    from cli import entrypoints

    settings = _launcher_settings()
    get_settings = MagicMock(side_effect=[settings, settings])
    get_settings.cache_clear = MagicMock()
    servers: list[object] = []

    class FakeServer:
        def __init__(self, config):
            self.config = config
            self.should_exit = False
            servers.append(self)

        def run(self):
            if len(servers) == 1:
                self.config.app.app.state.admin_restart_callback()
                assert self.should_exit is True

    def fake_config(app, **kwargs):
        return SimpleNamespace(app=app, kwargs=kwargs)

    with (
        patch.object(entrypoints, "get_settings", get_settings),
        patch.object(entrypoints.uvicorn, "Config", side_effect=fake_config),
        patch.object(entrypoints.uvicorn, "Server", side_effect=FakeServer),
        patch.object(entrypoints, "_schedule_open_admin_browser"),
        patch.object(entrypoints, "kill_all_best_effort") as kill_all,
    ):
        entrypoints.serve()

    assert len(servers) == 2
    get_settings.cache_clear.assert_called_once()
    kill_all.assert_called_once()


def test_serve_migrates_legacy_env_before_loading_settings(tmp_path: Path) -> None:
    from cli import entrypoints

    legacy_env = tmp_path / "free-claude-code" / ".env"
    legacy_env.parent.mkdir(parents=True)
    legacy_env.write_text("MODEL=deepseek/deepseek-chat\n", encoding="utf-8")
    settings = _launcher_settings()
    get_settings = MagicMock(return_value=settings)
    get_settings.cache_clear = MagicMock()

    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch.object(entrypoints, "get_settings", get_settings),
        patch.object(entrypoints, "_run_supervised_server", return_value=False),
        patch.object(entrypoints, "kill_all_best_effort"),
    ):
        entrypoints.serve()

    assert (tmp_path / ".fcc" / ".env").read_text("utf-8") == (
        "MODEL=deepseek/deepseek-chat\n"
    )
    get_settings.assert_called_once_with()


def test_serve_handles_keyboard_interrupt_without_traceback() -> None:
    from cli import entrypoints

    settings = _launcher_settings()
    get_settings = MagicMock(return_value=settings)
    get_settings.cache_clear = MagicMock()

    with (
        patch.object(entrypoints, "get_settings", get_settings),
        patch.object(
            entrypoints,
            "_run_supervised_server",
            side_effect=KeyboardInterrupt,
        ),
        patch.object(entrypoints, "kill_all_best_effort") as kill_all,
    ):
        entrypoints.serve()

    get_settings.cache_clear.assert_not_called()
    kill_all.assert_called_once()


def test_claude_child_env_targets_current_proxy_config() -> None:
    from cli.entrypoints import _claude_child_env

    env = _claude_child_env(
        _launcher_settings(port=9090, token=" proxy-token "),
        {
            "PATH": "keep",
            "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
            "ANTHROPIC_AUTH_TOKEN": "old-token",
            "ANTHROPIC_API_KEY": "official-key",
        },
    )

    assert env["PATH"] == "keep"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9090"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "proxy-token"
    assert env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "190000"
    assert "ANTHROPIC_API_KEY" not in env


def test_claude_child_env_threads_custom_auto_compact_window() -> None:
    """A configured window (e.g. 1M for glm-5.2[1m]) reaches the child env verbatim."""
    from cli.entrypoints import _claude_child_env

    settings = _launcher_settings()
    settings.claude_code_auto_compact_window = 1000000
    env = _claude_child_env(settings, {})

    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "1000000"


def test_claude_child_env_removes_blank_configured_auth_token() -> None:
    from cli.entrypoints import _claude_child_env

    env = _claude_child_env(
        _launcher_settings(token=""),
        {
            "ANTHROPIC_AUTH_TOKEN": "inherited-token",
            "ANTHROPIC_API_KEY": "official-key",
        },
    )

    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_launch_claude_passes_args_and_child_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cli.entrypoints import launch_claude

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "old-token")
    monkeypatch.setenv("KEEP_ME", "yes")
    settings = _launcher_settings(port=9191, token="proxy-token")

    with (
        patch("cli.entrypoints.get_settings", return_value=settings),
        patch("cli.entrypoints._preflight_proxy", return_value=None),
        patch("cli.entrypoints.shutil.which", return_value="resolved-claude.cmd"),
        patch("cli.entrypoints.subprocess.Popen") as popen,
        patch("cli.entrypoints.register_pid") as register_pid,
        patch("cli.entrypoints.unregister_pid") as unregister_pid,
        pytest.raises(SystemExit) as exc_info,
    ):
        process = popen.return_value
        process.pid = 12345
        process.wait.return_value = 7
        launch_claude(["--model", "sonnet"])

    assert exc_info.value.code == 7
    popen.assert_called_once()
    assert popen.call_args.args[0] == ["resolved-claude.cmd", "--model", "sonnet"]
    child_env = popen.call_args.kwargs["env"]
    assert child_env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9191"
    assert child_env["ANTHROPIC_AUTH_TOKEN"] == "proxy-token"
    assert child_env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
    assert child_env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "190000"
    assert child_env["KEEP_ME"] == "yes"
    register_pid.assert_called_once_with(12345)
    unregister_pid.assert_called_once_with(12345)


def test_launch_claude_keyboard_interrupt_kills_child_tree() -> None:
    from cli.entrypoints import launch_claude

    settings = _launcher_settings(port=9191, token="proxy-token")

    with (
        patch("cli.entrypoints.get_settings", return_value=settings),
        patch("cli.entrypoints._preflight_proxy", return_value=None),
        patch("cli.entrypoints.shutil.which", return_value="resolved-claude.cmd"),
        patch("cli.entrypoints.subprocess.Popen") as popen,
        patch("cli.entrypoints.register_pid"),
        patch("cli.entrypoints.kill_pid_tree_best_effort") as kill_tree,
        patch("cli.entrypoints.unregister_pid") as unregister_pid,
        pytest.raises(KeyboardInterrupt),
    ):
        process = popen.return_value
        process.pid = 12345
        process.wait.side_effect = [KeyboardInterrupt, 0]

        launch_claude([])

    kill_tree.assert_called_once_with(12345)
    unregister_pid.assert_called_once_with(12345)


def test_launch_claude_exits_when_command_cannot_be_resolved(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cli.entrypoints import launch_claude

    settings = _launcher_settings()
    with (
        patch("cli.entrypoints.get_settings", return_value=settings),
        patch("cli.entrypoints._preflight_proxy", return_value=None),
        patch("cli.entrypoints.shutil.which", return_value=None),
        patch("cli.entrypoints.subprocess.Popen") as popen,
        pytest.raises(SystemExit) as exc_info,
    ):
        launch_claude([])

    assert exc_info.value.code == 127
    popen.assert_not_called()
    captured = capsys.readouterr()
    assert "Could not find Claude Code command: claude" in captured.err
    assert "npm install -g @anthropic-ai/claude-code" in captured.err


def test_launch_claude_unreachable_proxy_exits_with_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cli.entrypoints import launch_claude

    settings = _launcher_settings(port=9393)
    with (
        patch("cli.entrypoints.get_settings", return_value=settings),
        patch("cli.entrypoints._preflight_proxy", return_value="connection refused"),
        patch("cli.entrypoints.subprocess.run") as run,
        pytest.raises(SystemExit) as exc_info,
    ):
        launch_claude([])

    assert exc_info.value.code == 1
    run.assert_not_called()
    captured = capsys.readouterr()
    assert "http://127.0.0.1:9393" in captured.err
    assert "fcc-server" in captured.err


def test_server_is_stale_returns_false_when_health_unreachable() -> None:
    from cli.entrypoints import _server_is_stale

    with patch("cli.entrypoints._fetch_health", return_value=None):
        assert _server_is_stale("http://127.0.0.1:9999") is False


def test_server_is_stale_returns_false_when_no_last_activity() -> None:
    from cli.entrypoints import _server_is_stale

    with patch(
        "cli.entrypoints._fetch_health",
        return_value={"status": "healthy"},
    ):
        assert _server_is_stale("http://127.0.0.1:9999") is False


def test_server_is_stale_returns_false_when_recently_active() -> None:
    import time as _time

    from cli.entrypoints import _server_is_stale

    with patch(
        "cli.entrypoints._fetch_health",
        return_value={"status": "healthy", "last_activity_epoch": _time.monotonic()},
    ):
        assert _server_is_stale("http://127.0.0.1:9999") is False


def test_server_is_stale_returns_true_when_idle_beyond_threshold() -> None:
    import time as _time

    from cli.entrypoints import SERVER_STALE_THRESHOLD_SECONDS, _server_is_stale

    old_epoch = _time.monotonic() - SERVER_STALE_THRESHOLD_SECONDS - 1
    with patch(
        "cli.entrypoints._fetch_health",
        return_value={"status": "healthy", "last_activity_epoch": old_epoch},
    ):
        assert _server_is_stale("http://127.0.0.1:9999") is True


def test_launch_claude_restarts_stale_server_before_preflight() -> None:
    from cli.entrypoints import launch_claude

    settings = _launcher_settings(port=9191, token="proxy-token")

    with (
        patch("cli.entrypoints.get_settings", return_value=settings),
        patch("cli.entrypoints._server_is_stale", return_value=True),
        patch("cli.entrypoints._restart_stale_server") as restart,
        patch("cli.entrypoints._preflight_proxy", return_value=None),
        patch("cli.entrypoints.shutil.which", return_value="resolved-claude.cmd"),
        patch("cli.entrypoints.subprocess.Popen") as popen,
        patch("cli.entrypoints.register_pid"),
        patch("cli.entrypoints.unregister_pid"),
        pytest.raises(SystemExit),
    ):
        process = popen.return_value
        process.pid = 12345
        process.wait.return_value = 0
        launch_claude([])

    restart.assert_called_once_with(settings)


def test_launch_claude_skips_restart_when_not_stale() -> None:
    from cli.entrypoints import launch_claude

    settings = _launcher_settings(port=9191, token="proxy-token")

    with (
        patch("cli.entrypoints.get_settings", return_value=settings),
        patch("cli.entrypoints._server_is_stale", return_value=False),
        patch("cli.entrypoints._restart_stale_server") as restart,
        patch("cli.entrypoints._preflight_proxy", return_value=None),
        patch("cli.entrypoints.shutil.which", return_value="resolved-claude.cmd"),
        patch("cli.entrypoints.subprocess.Popen") as popen,
        patch("cli.entrypoints.register_pid"),
        patch("cli.entrypoints.unregister_pid"),
        pytest.raises(SystemExit),
    ):
        process = popen.return_value
        process.pid = 12345
        process.wait.return_value = 0
        launch_claude([])

    restart.assert_not_called()
