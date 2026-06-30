"""Messaging platform component factory."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from .ports import MessagingPlatformComponents


@dataclass(frozen=True, slots=True)
class MessagingPlatformOptions:
    """Typed wiring from app settings into messaging platform runtimes."""

    telegram_bot_token: str | None = None
    allowed_telegram_user_id: str | None = None
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


def create_messaging_components(
    platform_type: str,
    options: MessagingPlatformOptions | None = None,
) -> MessagingPlatformComponents | None:
    """Create runtime/outbound components for the configured messaging platform."""
    opts = options or MessagingPlatformOptions()
    if platform_type == "none":
        logger.info("Messaging platform disabled by configuration")
        return None

    if platform_type == "telegram":
        bot_token = opts.telegram_bot_token
        if not bot_token:
            logger.info("No Telegram bot token configured, skipping platform setup")
            return None

        from .telegram import TelegramRuntime

        runtime = TelegramRuntime(
            bot_token=bot_token,
            allowed_user_id=opts.allowed_telegram_user_id,
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
        return MessagingPlatformComponents(
            name=runtime.name,
            runtime=runtime,
            outbound=runtime.outbound,
            voice_cancellation=runtime,
        )

    if platform_type == "discord":
        bot_token = opts.discord_bot_token
        if not bot_token:
            logger.info("No Discord bot token configured, skipping platform setup")
            return None

        from .discord import DiscordRuntime

        runtime = DiscordRuntime(
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
        return MessagingPlatformComponents(
            name=runtime.name,
            runtime=runtime,
            outbound=runtime.outbound,
            voice_cancellation=runtime,
        )

    logger.warning(
        "Unknown messaging platform: '{}'. Supported: 'none', 'telegram', 'discord'",
        platform_type,
    )
    return None
