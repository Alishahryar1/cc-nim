"""Unit tests for the `fcc-codex-desktop` launcher."""

import sys
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from free_claude_code.cli.launchers.codex_desktop import (
    codex_config_path,
    ephemeral_codex_config,
    launch,
    prepare_codex_config_content,
    resolve_codex_desktop_binary,
)


def test_codex_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CODEX_HOME", raising=False)
    assert codex_config_path() == Path.home() / ".codex" / "config.toml"

    custom_dir = tmp_path / "custom_codex"
    monkeypatch.setenv("CODEX_HOME", str(custom_dir))
    assert codex_config_path() == custom_dir / "config.toml"


def test_resolve_codex_desktop_binary_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_exe = tmp_path / "codex-desktop-fake"
    fake_exe.touch()
    fake_exe.chmod(0o755)

    monkeypatch.setenv("CODEX_DESKTOP_PATH", str(fake_exe))
    assert resolve_codex_desktop_binary() == str(fake_exe)

    monkeypatch.delenv("CODEX_DESKTOP_PATH", raising=False)
    monkeypatch.setenv("CODEX_PATH", str(fake_exe))
    assert resolve_codex_desktop_binary() == str(fake_exe)


def test_resolve_codex_desktop_binary_darwin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CODEX_DESKTOP_PATH", raising=False)
    monkeypatch.delenv("CODEX_PATH", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")

    fake_app = tmp_path / "Applications" / "Codex.app" / "Contents" / "MacOS" / "Codex"
    fake_app.parent.mkdir(parents=True, exist_ok=True)
    fake_app.touch()

    original_is_file = Path.is_file

    def mock_is_file(self: Path) -> bool:
        if str(self) == "/Applications/Codex.app/Contents/MacOS/Codex":
            return True
        return original_is_file(self)

    monkeypatch.setattr(Path, "is_file", mock_is_file)
    assert (
        resolve_codex_desktop_binary() == "/Applications/Codex.app/Contents/MacOS/Codex"
    )


def test_resolve_codex_desktop_binary_win32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CODEX_DESKTOP_PATH", raising=False)
    monkeypatch.delenv("CODEX_PATH", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")

    fake_local_appdata = tmp_path / "LocalAppData"
    monkeypatch.setenv("LOCALAPPDATA", str(fake_local_appdata))

    expected_exe = fake_local_appdata / "Programs" / "Codex" / "Codex.exe"
    expected_exe.parent.mkdir(parents=True, exist_ok=True)
    expected_exe.touch()

    assert resolve_codex_desktop_binary() == str(expected_exe)


def test_resolve_codex_desktop_binary_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CODEX_DESKTOP_PATH", raising=False)
    monkeypatch.delenv("CODEX_PATH", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")

    fake_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    fake_bin = fake_home / ".local" / "bin" / "codex-desktop"
    fake_bin.parent.mkdir(parents=True, exist_ok=True)
    fake_bin.touch()

    assert resolve_codex_desktop_binary() == str(fake_bin)


def test_resolve_codex_desktop_binary_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CODEX_DESKTOP_PATH", raising=False)
    monkeypatch.delenv("CODEX_PATH", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "home", lambda: Path("/nonexistent/home"))
    monkeypatch.setattr("shutil.which", lambda name: None)

    monkeypatch.setattr(Path, "is_file", lambda self: False)

    with pytest.raises(SystemExit) as exc_info:
        resolve_codex_desktop_binary()

    assert exc_info.value.code == 127
    stderr = capsys.readouterr().err
    assert "Could not find Codex Desktop command: codex-desktop" in stderr
    assert "Install Codex Desktop" in stderr


def test_prepare_codex_config_content_new() -> None:
    content = prepare_codex_config_content(
        None,
        api_url="http://127.0.0.1:8311/v1",
        model="gpt-4o",
    )
    parsed = tomllib.loads(content)

    assert parsed["model_provider"] == "fcc"
    assert parsed["model"] == "gpt-4o"
    assert parsed["model_providers"]["fcc"]["name"] == "Free Claude Code"
    assert parsed["model_providers"]["fcc"]["base_url"] == "http://127.0.0.1:8311/v1"
    assert parsed["model_providers"]["fcc"]["wire_api"] == "responses"
    assert parsed["model_providers"]["fcc"]["auth"]["command"] == "fcc-codex"
    assert parsed["model_providers"]["fcc"]["auth"]["args"] == [
        "--print-proxy-auth-token"
    ]


def test_prepare_codex_config_content_existing() -> None:
    existing = (
        'theme = "dark"\nmodel_provider = "openai"\n\n[custom_settings]\nkey = 123\n'
    )
    content = prepare_codex_config_content(
        existing,
        api_url="http://127.0.0.1:8311/v1",
    )
    parsed = tomllib.loads(content)

    assert parsed["theme"] == "dark"
    assert parsed["model_provider"] == "fcc"
    assert parsed["custom_settings"]["key"] == 123
    assert parsed["model_providers"]["fcc"]["base_url"] == "http://127.0.0.1:8311/v1"


def test_ephemeral_codex_config_creation_and_cleanup(tmp_path: Path) -> None:
    config_file = tmp_path / ".codex" / "config.toml"
    assert not config_file.exists()

    with ephemeral_codex_config(
        config_file,
        proxy_root_url="http://127.0.0.1:8311",
        model="claude-3-5-sonnet",
    ):
        assert config_file.exists()
        content = config_file.read_text(encoding="utf-8")
        parsed = tomllib.loads(content)
        assert parsed["model_provider"] == "fcc"
        assert parsed["model"] == "claude-3-5-sonnet"

    assert not config_file.exists()


def test_ephemeral_codex_config_restoration_on_exception(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    original_text = 'model_provider = "openai"\n'
    config_file.write_text(original_text, encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="launcher crash"),
        ephemeral_codex_config(
            config_file,
            proxy_root_url="http://127.0.0.1:8311",
        ),
    ):
        assert 'model_provider = "fcc"' in config_file.read_text(encoding="utf-8")
        raise RuntimeError("launcher crash")

    assert config_file.read_text(encoding="utf-8") == original_text


def test_launch_preflight_failure(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch(
            "free_claude_code.cli.launchers.codex_desktop.preflight_proxy",
            return_value="connection refused",
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        launch([])

    assert exc_info.value.code == 1
    stderr = capsys.readouterr().err
    assert "Free Claude Code proxy is not reachable at" in stderr
    assert "connection refused" in stderr


def test_launch_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "config.toml"
    monkeypatch.setattr(
        "free_claude_code.cli.launchers.codex_desktop.codex_config_path",
        lambda: config_file,
    )
    monkeypatch.setattr(
        "free_claude_code.cli.launchers.codex_desktop.resolve_codex_desktop_binary",
        lambda: "/usr/bin/codex-desktop",
    )
    monkeypatch.setattr(
        "free_claude_code.cli.launchers.codex_desktop.preflight_proxy",
        lambda url: None,
    )

    mock_run = MagicMock()
    monkeypatch.setattr(
        "free_claude_code.cli.launchers.codex_desktop.run_client_process",
        mock_run,
    )

    launch(["--project", "my-app"])

    mock_run.assert_called_once()
    kwargs = mock_run.call_args.kwargs
    assert kwargs["command"] == ["/usr/bin/codex-desktop", "--project", "my-app"]
    assert kwargs["display_name"] == "Codex Desktop"
    assert not config_file.exists()  # Cleaned up after launch process finished
