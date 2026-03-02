import os
import signal
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import uvicorn

from cc_nim.cli import (
    _get_config_path,
    _get_pid_path,
    init_command,
    start_command,
    stop_command,
)


# ==================== _get_config_path tests ====================

def test_get_config_path_uses_ccproxy_config(monkeypatch, tmp_path):
    config_path = tmp_path / "custom.env"
    config_path.touch()
    monkeypatch.setenv("CCPROXY_CONFIG", str(config_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    assert _get_config_path() == config_path


def test_get_config_path_uses_home_ccenv(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    ccenv = home / ".ccenv"
    ccenv.touch()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("CCPROXY_CONFIG", raising=False)
    assert _get_config_path() == ccenv


def test_get_config_path_fallback_to_cwd(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "project"
    cwd.mkdir()
    dotenv = cwd / ".env"
    dotenv.touch()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(Path, "cwd", lambda: cwd)
    monkeypatch.delenv("CCPROXY_CONFIG", raising=False)
    assert _get_config_path() == dotenv


def test_get_config_path_returns_home_when_none_exist(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "project"
    cwd.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(Path, "cwd", lambda: cwd)
    monkeypatch.delenv("CCPROXY_CONFIG", raising=False)
    result = _get_config_path()
    assert result == home / ".ccenv"


# ==================== _get_pid_path tests ====================

def test_get_pid_path_returns_correct_location(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    assert _get_pid_path() == home / ".cc-nim" / "cc-nim.pid"


# ==================== init_command tests ====================

def test_init_command_creates_config(monkeypatch, tmp_path):
    config_path = tmp_path / "config.env"
    monkeypatch.setattr("cc_nim.cli._get_config_path", lambda: config_path)
    # Ensure config does not exist
    assert not config_path.exists()
    init_command()
    assert config_path.exists()
    content = config_path.read_text()
    assert "NVIDIA_NIM_API_KEY=" in content
    assert "MODEL=" in content
    assert "CLAUDE_WORKSPACE=" in content


def test_init_command_fails_if_config_exists(monkeypatch, tmp_path):
    config_path = tmp_path / "config.env"
    config_path.touch()
    monkeypatch.setattr("cc_nim.cli._get_config_path", lambda: config_path)
    result = init_command()
    assert result == 1


# ==================== start_command tests ====================

def test_start_command_writes_pid_and_starts_uvicorn(monkeypatch, tmp_path):
    pid_path = tmp_path / "cc-nim.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("cc_nim.cli._get_pid_path", lambda: pid_path)
    mock_settings = MagicMock()
    mock_settings.host = "0.0.0.0"
    mock_settings.port = 8082
    mock_get_settings = MagicMock(return_value=mock_settings)
    monkeypatch.setattr("cc_nim.cli.get_settings", mock_get_settings)
    monkeypatch.setattr(os, "getpid", lambda: 12345)
    mock_uvicorn_run = MagicMock()
    monkeypatch.setattr(uvicorn, "run", mock_uvicorn_run)
    monkeypatch.setattr("atexit.register", lambda func: None)  # Fixed target
    # Prevent PID file cleanup so we can assert its existence after start_command
    monkeypatch.setattr(pid_path, "unlink", lambda *args, **kwargs: None)
    # Mock Path.cwd to project root (where server.py exists)
    monkeypatch.setattr(Path, "cwd", lambda: Path(__file__).parent.parent.parent)
    if pid_path.exists():
        pid_path.unlink()
    # Ensure config exists
    config_path = tmp_path / "config.env"
    config_path.touch()
    monkeypatch.setattr("cc_nim.cli._get_config_path", lambda: config_path)
    start_command()
    assert pid_path.exists()
    assert pid_path.read_text().strip() == "12345"
    mock_uvicorn_run.assert_called_once()
    # Check that uvicorn.run was called with expected arguments
    call_kwargs = mock_uvicorn_run.call_args[1]
    assert call_kwargs["host"] == "0.0.0.0"
    assert call_kwargs["port"] == 8082
    assert call_kwargs["log_level"] == "info"
    assert call_kwargs["timeout_graceful_shutdown"] == 5


def test_start_command_removes_stale_pid_and_continues(monkeypatch, tmp_path):
    pid_path = tmp_path / "cc-nim.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text("12345")
    monkeypatch.setattr("cc_nim.cli._get_pid_path", lambda: pid_path)
    mock_settings = MagicMock()
    mock_settings.port = 8082
    mock_settings.host = "0.0.0.0"
    mock_get_settings = MagicMock(return_value=mock_settings)
    monkeypatch.setattr("cc_nim.cli.get_settings", mock_get_settings)
    monkeypatch.setattr(os, "getpid", lambda: 99999)
    # Simulate stale PID by making os.kill raise OSError
    def mock_kill(pid, sig):
        raise OSError("No such process")
    monkeypatch.setattr(os, "kill", mock_kill)
    mock_uvicorn_run = MagicMock()
    monkeypatch.setattr(uvicorn, "run", mock_uvicorn_run)
    monkeypatch.setattr("atexit.register", lambda func: None)  # Fixed target
    # Prevent PID file cleanup in finally so we can assert existence after start_command
    monkeypatch.setattr(pid_path, "unlink", lambda *args, **kwargs: None)
    # Provide cwd so server module can be found
    monkeypatch.setattr(Path, "cwd", lambda: Path(__file__).parent.parent.parent)
    # Ensure config exists
    config_path = tmp_path / "config.env"
    config_path.touch()
    monkeypatch.setattr("cc_nim.cli._get_config_path", lambda: config_path)
    start_command()
    assert pid_path.exists()
    assert pid_path.read_text().strip() == "99999"
    mock_uvicorn_run.assert_called_once()


def test_start_command_exits_if_process_already_running(monkeypatch, tmp_path):
    pid_path = tmp_path / "cc-nim.pid"
    pid_path.write_text("99999")
    monkeypatch.setattr("cc_nim.cli._get_pid_path", lambda: pid_path)
    mock_settings = MagicMock()
    mock_settings.port = 8082
    mock_get_settings = MagicMock(return_value=mock_settings)
    monkeypatch.setattr("cc_nim.cli.get_settings", mock_get_settings)
    # Simulate os.kill that doesn't raise -> process exists
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    # Ensure config exists
    config_path = tmp_path / "config.env"
    config_path.touch()
    monkeypatch.setattr("cc_nim.cli._get_config_path", lambda: config_path)
    result = start_command()
    assert result == 1
    assert pid_path.exists()


# ==================== stop_command tests ====================

def test_stop_command_stops_via_brew_success(monkeypatch, tmp_path):
    pid_path = tmp_path / "cc-nim.pid"
    pid_path.write_text("12345")
    monkeypatch.setattr("cc_nim.cli._get_pid_path", lambda: pid_path)
    result = MagicMock(returncode=0)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: result)
    # Ensure os.kill is not called
    def raise_assertion(*args, **kwargs):
        raise AssertionError("os.kill should not be called")
    monkeypatch.setattr(os, "kill", raise_assertion)
    stop_command()
    assert not pid_path.exists()


def test_stop_command_uses_kill_when_brew_fails(monkeypatch, tmp_path):
    pid_path = tmp_path / "cc-nim.pid"
    pid_path.write_text("12345")
    monkeypatch.setattr("cc_nim.cli._get_pid_path", lambda: pid_path)
    result = MagicMock(returncode=1, stdout=b"", stderr=b"")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: result)
    mock_kill = MagicMock()
    monkeypatch.setattr(os, "kill", mock_kill)
    monkeypatch.setattr(signal, "SIGTERM", signal.SIGTERM)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    stop_command()
    mock_kill.assert_called_once_with(12345, signal.SIGTERM)
    assert not pid_path.exists()


def test_stop_command_fails_when_no_pid_and_brew_fails(monkeypatch):
    pid_path = Path("/nonexistent/cc-nim.pid")
    monkeypatch.setattr("cc_nim.cli._get_pid_path", lambda: pid_path)
    result = MagicMock(returncode=1)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: result)
    ret = stop_command()
    assert ret == 1


def test_stop_command_handles_kill_error(monkeypatch, tmp_path):
    pid_path = tmp_path / "cc-nim.pid"
    pid_path.write_text("12345")
    monkeypatch.setattr("cc_nim.cli._get_pid_path", lambda: pid_path)
    result = MagicMock(returncode=1)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: result)
    def kill_error(pid, sig):
        raise OSError("Operation not permitted")
    monkeypatch.setattr(os, "kill", kill_error)
    monkeypatch.setattr(signal, "SIGTERM", signal.SIGTERM)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    ret = stop_command()
    assert ret == 1
