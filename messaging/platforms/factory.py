"""Messaging platform factory.

Creates the appropriate messaging platform adapter based on configuration.
To add a new platform (e.g. Discord, Slack):
1. Create a new class implementing MessagingPlatform in messaging/platforms/
2. Add a case to create_messaging_platform() below
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from .base import MessagingPlatform


@dataclass(frozen=True, slots=True)
class MessagingPlatformOptions:
    """Typed wiring from :class:`~api.runtime.AppRuntime` into platform adapters."""

    telegram_bot_token: str | None = None
    allowed_telegram_user_id: str | None = None
    telegram_proxy: str | None = None
    discord_bot_token: str | None = None
    allowed_discord_channels: str | None = None
    voice_note_enabled: bool = True
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    hf_token: str = ""
    nvidia_nim_api_key: str = ""
    messaging_rate_limit: int = 1
    messaging_rate_window: float = 1.0
    log_raw_messaging_content: bool = False
    log_api_error_tracebacks: bool = False


def create_messaging_platform(
    platform_type: str,
    options: MessagingPlatformOptions | None = None,
) -> MessagingPlatform | None:
    """Create a messaging platform instance based on type.

    Args:
        platform_type: Platform identifier (``telegram``, ``discord``, ``none``).
        options: Token, allowlist, and voice / transcription settings.

    Returns:
        Configured :class:`MessagingPlatform` instance, or None if not configured.
    """
    opts = options or MessagingPlatformOptions()
    if platform_type == "none":
        logger.info("Messaging platform disabled by configuration")
        return None

    if platform_type == "telegram":
        bot_token = opts.telegram_bot_token
        if not bot_token:
            logger.info("No Telegram bot token configured, skipping platform setup")
            return None

        from .telegram import TelegramPlatform

        return TelegramPlatform(
            bot_token=bot_token,
            allowed_user_id=opts.allowed_telegram_user_id,
            telegram_proxy=opts.telegram_proxy,
            voice_note_enabled=opts.voice_note_enabled,
            whisper_model=opts.whisper_model,
            whisper_device=opts.whisper_device,
            hf_token=opts.hf_token,
            nvidia_nim_api_key=opts.nvidia_nim_api_key,
            messaging_rate_limit=opts.messaging_rate_limit,
            messaging_rate_window=opts.messaging_rate_window,
            log_raw_messaging_content=opts.log_raw_messaging_content,
            log_api_error_tracebacks=opts.log_api_error_tracebacks,
        )

    if platform_type == "discord":
        bot_token = opts.discord_bot_token
        if not bot_token:
            logger.info("No Discord bot token configured, skipping platform setup")
            return None

        from .discord import DiscordPlatform

        return DiscordPlatform(
            bot_token=bot_token,
            allowed_channel_ids=opts.allowed_discord_channels,
            voice_note_enabled=opts.voice_note_enabled,
            whisper_model=opts.whisper_model,
            whisper_device=opts.whisper_device,
            hf_token=opts.hf_token,
            nvidia_nim_api_key=opts.nvidia_nim_api_key,
            messaging_rate_limit=opts.messaging_rate_limit,
            messaging_rate_window=opts.messaging_rate_window,
            log_raw_messaging_content=opts.log_raw_messaging_content,
            log_api_error_tracebacks=opts.log_api_error_tracebacks,
        )

    logger.warning(
        f"Unknown messaging platform: '{platform_type}'. "
        "Supported: 'none', 'telegram', 'discord'"
    )
    return None
