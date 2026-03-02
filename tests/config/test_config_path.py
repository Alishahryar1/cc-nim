"""Tests for config path resolution in settings module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from config.settings import _compute_config_path


@pytest.fixture
def mock_home(tmp_path):
    with patch("config.settings.Path.home", return_value=tmp_path):
        yield tmp_path


def test_uses_ccproxy_config_when_set(mock_home):
    """Should use CCPROXY_CONFIG when set, regardless of other files."""
    test_path = Path("/custom/path/.env")
    with patch.dict(os.environ, {"CCPROXY_CONFIG": str(test_path)}):
        result = _compute_config_path()
        assert result == test_path


def test_uses_home_ccenv_when_exists(mock_home):
    """Should use ~/.ccenv if it exists and no CCPROXY_CONFIG."""
    home_ccenv = mock_home / ".ccenv"
    home_ccenv.touch()
    with patch.dict(os.environ, {}, clear=True):
        result = _compute_config_path()
        assert result == home_ccenv


def test_falls_back_to_project_env_when_home_missing(mock_home):
    """Should fall back to .env in cwd if ~/.ccenv does not exist."""
    # Ensure home/.ccenv does not exist
    with patch.dict(os.environ, {}, clear=True):
        # mock cwd to a temp dir containing .env
        tmp_cwd = mock_home / "project"
        tmp_cwd.mkdir()
        dotenv = tmp_cwd / ".env"
        dotenv.touch()
        with patch("config.settings.Path.cwd", return_value=tmp_cwd):
            result = _compute_config_path()
            assert result == dotenv


def test_returns_home_ccenv_even_when_missing(mock_home):
    """If no config files exist, returns ~/.ccenv (may be created later)."""
    # No files exist
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("config.settings.Path.cwd") as mock_cwd,
    ):
        # cwd has no .env; Path.cwd().exists() would be True but .env not exist
        mock_cwd.return_value = mock_cwd / "no_env"
        mock_cwd.return_value.mkdir.return_value = None
        result = _compute_config_path()
        assert result == mock_home / ".ccenv"
