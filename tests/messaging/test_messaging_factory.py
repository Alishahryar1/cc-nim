"""Tests for messaging platform factory."""

from unittest.mock import MagicMock, patch

from messaging.platforms.factory import (
    MessagingPlatformOptions,
    create_messaging_components,
)


class TestCreateMessagingComponents:
    """Tests for create_messaging_components factory function."""

    def test_telegram_with_token(self):
        """Create Telegram platform when bot_token is provided."""
        mock_runtime = MagicMock()
        mock_runtime.name = "telegram"
        mock_runtime.outbound = MagicMock()
        with (
            patch("messaging.platforms.telegram.TELEGRAM_AVAILABLE", True),
            patch(
                "messaging.platforms.telegram.TelegramRuntime",
                return_value=mock_runtime,
            ) as runtime_cls,
        ):
            result = create_messaging_components(
                "telegram",
                MessagingPlatformOptions(
                    telegram_bot_token="test_token",
                    allowed_telegram_user_id="12345",
                    voice_note_enabled=False,
                    whisper_model="large-v3",
                    whisper_device="cuda",
                ),
            )

        assert result is not None
        assert result.runtime is mock_runtime
        assert result.outbound is mock_runtime.outbound
        assert result.voice_cancellation is mock_runtime
        runtime_cls.assert_called_once_with(
            bot_token="test_token",
            allowed_user_id="12345",
            voice_note_enabled=False,
            whisper_model="large-v3",
            whisper_device="cuda",
            hf_token="",
            nvidia_nim_api_key="",
            messaging_rate_limit=1,
            messaging_rate_window=1.0,
            log_raw_messaging_content=False,
            log_api_error_tracebacks=False,
        )

    def test_telegram_without_token(self):
        """Return None when no bot_token for Telegram."""
        result = create_messaging_components("telegram")
        assert result is None

    def test_telegram_empty_token(self):
        """Return None when bot_token is empty string."""
        result = create_messaging_components(
            "telegram", MessagingPlatformOptions(telegram_bot_token="")
        )
        assert result is None

    def test_discord_with_token(self):
        """Create Discord platform when discord_bot_token is provided."""
        mock_runtime = MagicMock()
        mock_runtime.name = "discord"
        mock_runtime.outbound = MagicMock()
        with (
            patch("messaging.platforms.discord.DISCORD_AVAILABLE", True),
            patch(
                "messaging.platforms.discord.DiscordRuntime",
                return_value=mock_runtime,
            ) as runtime_cls,
        ):
            result = create_messaging_components(
                "discord",
                MessagingPlatformOptions(
                    discord_bot_token="test_token",
                    allowed_discord_channels="123,456",
                    voice_note_enabled=False,
                    whisper_model="small",
                    whisper_device="nvidia_nim",
                ),
            )

        assert result is not None
        assert result.runtime is mock_runtime
        assert result.outbound is mock_runtime.outbound
        assert result.voice_cancellation is mock_runtime
        runtime_cls.assert_called_once_with(
            bot_token="test_token",
            allowed_channel_ids="123,456",
            voice_note_enabled=False,
            whisper_model="small",
            whisper_device="nvidia_nim",
            hf_token="",
            nvidia_nim_api_key="",
            messaging_rate_limit=1,
            messaging_rate_window=1.0,
            log_raw_messaging_content=False,
            log_api_error_tracebacks=False,
        )

    def test_discord_without_token(self):
        """Return None when no discord_bot_token for Discord."""
        result = create_messaging_components("discord")
        assert result is None

    def test_discord_empty_token(self):
        """Return None when discord_bot_token is empty string."""
        result = create_messaging_components(
            "discord",
            MessagingPlatformOptions(
                discord_bot_token="",
                allowed_discord_channels="123",
            ),
        )
        assert result is None

    def test_unknown_platform(self):
        """Return None for unknown platform types."""
        result = create_messaging_components("slack")
        assert result is None

    def test_unknown_platform_with_kwargs(self):
        """Return None for unknown platform even with kwargs."""
        result = create_messaging_components(
            "slack", MessagingPlatformOptions(telegram_bot_token="token")
        )
        assert result is None
