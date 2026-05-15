"""Tests for cli/entrypoints.py — fcc-init scaffolding logic."""

import os
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
    claude_bin: str = "claude-test",
) -> Settings:
    return Settings.model_construct(
        host="0.0.0.0",
        port=port,
        anthropic_auth_token=token,
        claude_cli_bin=claude_bin,
    )


def _run_init(tmp_home: Path) -> tuple[str, Path]:
    """Run init() with home directory redirected to tmp_home. Returns (printed output, env_file path)."""
    from cli.entrypoints import init

    env_file = tmp_home / ".config" / "free-claude-code" / ".env"
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
    if os.name == "posix":
        assert env_file.stat().st_mode & 0o777 == 0o600


def test_init_copies_template_content(tmp_path: Path) -> None:
    """init() writes the canonical root env.example content, not an empty file."""
    template = (Path(__file__).resolve().parents[2] / ".env.example").read_text(
        encoding="utf-8"
    )
    _, env_file = _run_init(tmp_path)

    assert env_file.read_text("utf-8") == template


def test_env_template_loader_uses_root_template_in_source_checkout() -> None:
    """Source checkout fallback uses the root .env.example as the single source."""
    from cli.entrypoints import _load_env_template

    template = (Path(__file__).resolve().parents[2] / ".env.example").read_text(
        encoding="utf-8"
    )

    assert _load_env_template() == template


def test_init_creates_parent_directories(tmp_path: Path) -> None:
    """init() creates ~/.config/free-claude-code/ even if it doesn't exist."""
    config_dir = tmp_path / ".config" / "free-claude-code"
    assert not config_dir.exists()

    _run_init(tmp_path)

    assert config_dir.is_dir()


def test_init_skips_if_env_already_exists(tmp_path: Path) -> None:
    """init() does not overwrite an existing .env and prints a warning."""
    # Create it first
    _run_init(tmp_path)

    env_file = tmp_path / ".config" / "free-claude-code" / ".env"
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
        patch.object(entrypoints, "kill_all_best_effort") as kill_all,
    ):
        entrypoints.serve()

    assert len(servers) == 2
    get_settings.cache_clear.assert_called_once()
    kill_all.assert_called_once()


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
    assert "ANTHROPIC_API_KEY" not in env


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

    settings = _launcher_settings(claude_bin="claude-missing")
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
    assert "Could not find Claude Code command: claude-missing" in captured.err
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
