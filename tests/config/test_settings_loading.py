"""Tests for settings loading from config file."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from config.settings import Settings, get_settings


@pytest.fixture
def temp_env_file(tmp_path):
    """Create a temporary .env file with sample content."""
    env_file = tmp_path / "test.env"
    env_file.write_text(
        """NVIDIA_NIM_API_KEY=nvapi-test-key
MODEL=nvidia_nim/stepfun-ai/step-3.5-flash
PROVIDER_RATE_LIMIT=50
VOICE_NOTE_ENABLED=false
WHISPER_DEVICE=cuda
DISCORD_BOT_TOKEN=
ALLOWED_DISCORD_CHANNELS=
"""
    )
    return env_file


def test_settings_loads_from_config_file(temp_env_file):
    """Should load settings from the specified config file."""
    with patch.dict(os.environ, {"CCPROXY_CONFIG": str(temp_env_file)}):
        # Clear the lru_cache to force reload
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.nvidia_nim_api_key == "nvapi-test-key"
        assert settings.model == "nvidia_nim/stepfun-ai/step-3.5-flash"
        assert settings.provider_rate_limit == 50
        assert settings.voice_note_enabled is False
        assert settings.whisper_device == "cuda"
        # Empty string fields should become None
        assert settings.discord_bot_token is None
        assert settings.allowed_discord_channels is None


@pytest.fixture
def mock_home(tmp_path):
    with patch("config.settings.Path.home", return_value=tmp_path):
        yield tmp_path


def test_settings_uses_home_ccenv_when_no_override(mock_home, temp_env_file):
    """Should load from ~/.ccenv when CCPROXY_CONFIG is not set."""
    home_ccenv = mock_home / ".ccenv"
    home_ccenv.write_text(temp_env_file.read_text())
    with patch.dict(os.environ, {}, clear=True):
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.nvidia_nim_api_key == "nvapi-test-key"


def test_settings_defaults_when_no_config():
    """Should use default values when no config file exists."""
    # Ensure no config files are found
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("config.settings._compute_config_path") as mock_compute,
    ):
        mock_compute.return_value = Path("/nonexistent/.env")
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.nvidia_nim_api_key == ""
        assert settings.model == "nvidia_nim/meta/llama3-70b-instruct"
        assert settings.provider_rate_limit == 40


def test_optional_str_validator_converts_empty_to_none():
    """Validator should convert empty strings to None for optional fields."""
    # Direct instantiation to test validator
    settings = Settings(discord_bot_token="", allowed_discord_channels="")
    assert settings.discord_bot_token is None
    assert settings.allowed_discord_channels is None
