"""Tests for cli/entrypoints.py — fcc-init scaffolding logic."""

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.error import URLError
from urllib.request import Request

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
        model="nvidia_nim/test-model",
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


class _JsonResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> _JsonResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


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
    assert scripts["fcc-claude"] == "cli.launchers.claude:launch"
    assert scripts["fcc-codex"] == "cli.launchers.codex:launch"


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
        patch.object(entrypoints, "preflight_proxy", return_value=None),
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
    from cli.launchers.claude import build_claude_launcher_env

    env = build_claude_launcher_env(
        proxy_root_url="http://127.0.0.1:9090",
        auth_token=" proxy-token ",
        base_env={
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


def test_claude_child_env_uses_sentinel_for_blank_configured_auth_token() -> None:
    from cli.launchers.claude import build_claude_launcher_env

    env = build_claude_launcher_env(
        proxy_root_url="http://127.0.0.1:8082",
        auth_token="",
        base_env={
            "ANTHROPIC_AUTH_TOKEN": "inherited-token",
            "ANTHROPIC_API_KEY": "official-key",
        },
    )

    assert env["ANTHROPIC_AUTH_TOKEN"] == "fcc-no-auth"
    assert "ANTHROPIC_API_KEY" not in env


def test_launch_claude_passes_args_and_child_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cli.launchers.claude import launch

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "old-token")
    monkeypatch.setenv("KEEP_ME", "yes")
    settings = _launcher_settings(port=9191, token="proxy-token")

    with (
        patch("cli.launchers.claude.get_settings", return_value=settings),
        patch("cli.launchers.claude.preflight_proxy", return_value=None),
        patch("cli.launchers.common.shutil.which", return_value="resolved-claude.cmd"),
        patch("cli.launchers.common.subprocess.Popen") as popen,
        patch("cli.launchers.common.register_pid") as register_pid,
        patch("cli.launchers.common.unregister_pid") as unregister_pid,
        pytest.raises(SystemExit) as exc_info,
    ):
        process = popen.return_value
        process.pid = 12345
        process.wait.return_value = 7
        launch(["--model", "sonnet"])

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


def test_launch_codex_passes_responses_config_and_child_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cli.launchers.codex import launch

    monkeypatch.setenv("OPENAI_API_KEY", "official-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("CODEX_HOME", "keep-home")
    settings = _launcher_settings(port=9191, token="proxy-token")
    catalog_path = tmp_path / "codex-model-catalog.json"
    requests: list[Request] = []

    def fake_urlopen(request: Request, *, timeout: float) -> _JsonResponse:
        requests.append(request)
        assert timeout == 1.5
        return _JsonResponse(
            {
                "data": [
                    {
                        "id": "anthropic/nvidia_nim/provider-model",
                        "display_name": "NVIDIA model",
                    },
                    {
                        "id": ("claude-3-freecc-no-thinking/nvidia_nim/provider-model"),
                        "display_name": "NVIDIA model (no thinking)",
                    },
                    {
                        "id": "claude-opus-4-20250514",
                        "display_name": "Claude Opus 4",
                    },
                ]
            }
        )

    with (
        patch("cli.launchers.codex.get_settings", return_value=settings),
        patch("cli.launchers.codex.preflight_proxy", return_value=None),
        patch("cli.launchers.common.shutil.which", return_value="resolved-codex.cmd"),
        patch(
            "cli.launchers.codex.codex_model_catalog_path", return_value=catalog_path
        ),
        patch("cli.launchers.codex.urlopen", side_effect=fake_urlopen),
        patch("cli.launchers.common.subprocess.Popen") as popen,
        patch("cli.launchers.common.register_pid") as register_pid,
        patch("cli.launchers.common.unregister_pid") as unregister_pid,
        pytest.raises(SystemExit) as exc_info,
    ):
        process = popen.return_value
        process.pid = 12345
        process.wait.return_value = 0
        launch(["exec", "hello"])

    assert exc_info.value.code == 0
    command = popen.call_args.args[0]
    assert command[0] == "resolved-codex.cmd"
    assert 'model_provider="fcc"' in command
    assert 'model_providers.fcc.base_url="http://127.0.0.1:9191/v1"' in command
    assert 'model_providers.fcc.wire_api="responses"' in command
    assert f"model_catalog_json={json.dumps(str(catalog_path))}" in command
    assert command[-2:] == ["exec", "hello"]
    assert len(requests) == 1
    request = requests[0]
    assert request.full_url == "http://127.0.0.1:9191/v1/models"
    headers = {key.lower(): value for key, value in request.header_items()}
    assert headers["x-api-key"] == "proxy-token"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert [model["slug"] for model in catalog["models"]] == [
        "nvidia_nim/provider-model"
    ]
    child_env = popen.call_args.kwargs["env"]
    assert child_env["FCC_CODEX_API_KEY"] == "proxy-token"
    assert child_env["CODEX_HOME"] == "keep-home"
    assert "OPENAI_API_KEY" not in child_env
    assert "OPENAI_BASE_URL" not in child_env
    register_pid.assert_called_once_with(12345)
    unregister_pid.assert_called_once_with(12345)


def test_launch_codex_catalog_failure_warns_and_continues(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from cli.launchers.codex import launch

    settings = _launcher_settings(port=9191, token="proxy-token")

    with (
        patch("cli.launchers.codex.get_settings", return_value=settings),
        patch("cli.launchers.codex.preflight_proxy", return_value=None),
        patch("cli.launchers.common.shutil.which", return_value="resolved-codex.cmd"),
        patch(
            "cli.launchers.codex.codex_model_catalog_path",
            return_value=tmp_path / "codex-model-catalog.json",
        ),
        patch("cli.launchers.codex.urlopen", side_effect=URLError("boom")),
        patch("cli.launchers.common.subprocess.Popen") as popen,
        patch("cli.launchers.common.register_pid"),
        patch("cli.launchers.common.unregister_pid"),
        pytest.raises(SystemExit) as exc_info,
    ):
        process = popen.return_value
        process.pid = 12345
        process.wait.return_value = 0
        launch(["exec", "hello"])

    assert exc_info.value.code == 0
    command = popen.call_args.args[0]
    assert not any("model_catalog_json=" in arg for arg in command)
    captured = capsys.readouterr()
    assert "could not prepare Codex model catalog" in captured.err
    assert "launching without model picker catalog" in captured.err


def test_launch_claude_keyboard_interrupt_kills_child_tree() -> None:
    from cli.launchers.claude import launch

    settings = _launcher_settings(port=9191, token="proxy-token")

    with (
        patch("cli.launchers.claude.get_settings", return_value=settings),
        patch("cli.launchers.claude.preflight_proxy", return_value=None),
        patch("cli.launchers.common.shutil.which", return_value="resolved-claude.cmd"),
        patch("cli.launchers.common.subprocess.Popen") as popen,
        patch("cli.launchers.common.register_pid"),
        patch("cli.launchers.common.kill_pid_tree_best_effort") as kill_tree,
        patch("cli.launchers.common.unregister_pid") as unregister_pid,
        pytest.raises(KeyboardInterrupt),
    ):
        process = popen.return_value
        process.pid = 12345
        process.wait.side_effect = [KeyboardInterrupt, 0]

        launch([])

    kill_tree.assert_called_once_with(12345)
    unregister_pid.assert_called_once_with(12345)


def test_launch_claude_exits_when_command_cannot_be_resolved(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cli.launchers.claude import launch

    settings = _launcher_settings()
    with (
        patch("cli.launchers.claude.get_settings", return_value=settings),
        patch("cli.launchers.claude.preflight_proxy", return_value=None),
        patch("cli.launchers.common.shutil.which", return_value=None),
        patch("cli.launchers.common.subprocess.Popen") as popen,
        pytest.raises(SystemExit) as exc_info,
    ):
        launch([])

    assert exc_info.value.code == 127
    popen.assert_not_called()
    captured = capsys.readouterr()
    assert "Could not find Claude Code command: claude" in captured.err
    assert "npm install -g @anthropic-ai/claude-code" in captured.err


def test_launch_claude_unreachable_proxy_exits_with_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cli.launchers.claude import launch

    settings = _launcher_settings(port=9393)
    with (
        patch("cli.launchers.claude.get_settings", return_value=settings),
        patch(
            "cli.launchers.claude.preflight_proxy", return_value="connection refused"
        ),
        patch("cli.launchers.common.subprocess.Popen") as popen,
        pytest.raises(SystemExit) as exc_info,
    ):
        launch([])

    assert exc_info.value.code == 1
    popen.assert_not_called()
    captured = capsys.readouterr()
    assert "http://127.0.0.1:9393" in captured.err
    assert "fcc-server" in captured.err
