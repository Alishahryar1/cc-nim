"""Discord messaging runtime."""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from core.anthropic import format_user_error_preview

from ..models import IncomingMessage
from ..rendering.discord_markdown import format_status_discord
from .discord_inbound import (
    discord_text_message_from_event,
    discord_voice_request_from_event,
    get_audio_attachment,
    parse_allowed_channels,
)
from .discord_io import DiscordMessenger
from .ports import InboundMessageHandler
from .voice_flow import VoiceNoteFlow

_discord_module: Any = None
try:
    import discord as _discord_import

    _discord_module = _discord_import
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False


def _get_discord() -> Any:
    """Return the discord module or raise a setup error."""
    if not DISCORD_AVAILABLE or _discord_module is None:
        raise ImportError(
            "discord.py is required. Install with: pip install discord.py"
        )
    return _discord_module


if DISCORD_AVAILABLE and _discord_module is not None:
    _discord = _discord_module

    class _DiscordClient(_discord.Client):
        """Internal Discord client that forwards events to the runtime."""

        def __init__(
            self,
            runtime: DiscordRuntime,
            intents: _discord.Intents,
        ) -> None:
            super().__init__(intents=intents)
            self._runtime = runtime

        async def on_ready(self) -> None:
            self._runtime._connected = True
            logger.info("Discord platform connected")

        async def on_message(self, message: Any) -> None:
            await self._runtime._handle_client_message(message)
else:
    _DiscordClient = None


class DiscordRuntime:
    """Owns Discord SDK lifecycle and inbound event handoff."""

    name = "discord"

    def __init__(
        self,
        bot_token: str | None = None,
        allowed_channel_ids: str | None = None,
        *,
        voice_note_enabled: bool = True,
        whisper_model: str = "base",
        whisper_device: str = "cpu",
        huggingface_api_key: str = "",
        nvidia_nim_api_key: str = "",
        messaging_rate_limit: int = 1,
        messaging_rate_window: float = 1.0,
        log_raw_messaging_content: bool = False,
        log_api_error_tracebacks: bool = False,
    ) -> None:
        if not DISCORD_AVAILABLE:
            raise ImportError(
                "discord.py is required. Install with: pip install discord.py"
            )

        self.bot_token = bot_token
        self.allowed_channel_ids = parse_allowed_channels(allowed_channel_ids)
        if not self.bot_token:
            logger.warning("DISCORD_BOT_TOKEN not set")

        discord = _get_discord()
        intents = discord.Intents.default()
        intents.message_content = True

        assert _DiscordClient is not None
        self._client = _DiscordClient(self, intents)
        self._message_handler: InboundMessageHandler | None = None
        self._connected = False
        self._limiter: Any | None = None
        self._start_task: asyncio.Task | None = None
        self.outbound = DiscordMessenger(
            get_client=lambda: self._client,
            get_discord=_get_discord,
            get_limiter=lambda: self._limiter,
        )
        self._voice_flow = VoiceNoteFlow(
            voice_note_enabled=voice_note_enabled,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            huggingface_api_key=huggingface_api_key,
            nvidia_nim_api_key=nvidia_nim_api_key,
            log_raw_messaging_content=log_raw_messaging_content,
            log_api_error_tracebacks=log_api_error_tracebacks,
        )
        self._messaging_rate_limit = messaging_rate_limit
        self._messaging_rate_window = messaging_rate_window
        self._log_raw_messaging_content = log_raw_messaging_content
        self._log_api_error_tracebacks = log_api_error_tracebacks

    async def _handle_client_message(self, message: Any) -> None:
        """Adapter entry point used by the internal Discord client."""
        await self._on_discord_message(message)

    async def cancel_pending_voice(
        self, chat_id: str, reply_id: str
    ) -> tuple[str, str] | None:
        """Cancel a pending voice transcription."""
        return await self._voice_flow.cancel_pending_voice(chat_id, reply_id)

    async def _handle_voice_note(
        self, message: Any, attachment: Any, channel_id: str
    ) -> bool:
        """Handle a Discord audio attachment."""
        return await self._voice_flow.handle(
            discord_voice_request_from_event(message, attachment, channel_id),
            message_handler=self._message_handler,
            queue_send_message=self.outbound.queue_send_message,
            queue_delete_messages=self.outbound.queue_delete_messages,
        )

    async def _on_discord_message(self, message: Any) -> None:
        """Handle incoming Discord messages."""
        if message.author.bot:
            return

        channel_id = str(message.channel.id)
        if not self.allowed_channel_ids or channel_id not in self.allowed_channel_ids:
            return

        if not message.content:
            audio_att = get_audio_attachment(message)
            if audio_att:
                await self._handle_voice_note(message, audio_att, channel_id)
            return

        incoming = discord_text_message_from_event(
            message,
            log_raw_messaging_content=self._log_raw_messaging_content,
        )
        if self._message_handler is None:
            return

        try:
            await self._message_handler(incoming)
        except Exception as e:
            if self._log_api_error_tracebacks:
                logger.error("Error handling message: {}", e)
            else:
                logger.error("Error handling message: exc_type={}", type(e).__name__)
            with contextlib.suppress(Exception):
                await self.outbound.send_message(
                    channel_id,
                    format_status_discord("Error:", format_user_error_preview(e)),
                    reply_to=str(message.id),
                )

    async def start(self) -> None:
        """Initialize and connect to Discord."""
        if not self.bot_token:
            raise ValueError("DISCORD_BOT_TOKEN is required")

        from ..limiter import MessagingRateLimiter

        self._limiter = await MessagingRateLimiter.get_instance(
            rate_limit=self._messaging_rate_limit,
            rate_window=self._messaging_rate_window,
        )

        self._start_task = asyncio.create_task(
            self._client.start(self.bot_token),
            name="discord-client-start",
        )

        max_wait = 30
        waited = 0.0
        while not self._connected and waited < max_wait:
            await asyncio.sleep(0.5)
            waited += 0.5

        if not self._connected:
            raise RuntimeError("Discord client failed to connect within timeout")

        logger.info("Discord platform started")

    async def stop(self) -> None:
        """Stop Discord SDK resources."""
        if self._client.is_closed():
            self._connected = False
            return

        await self._client.close()
        if self._start_task and not self._start_task.done():
            try:
                await asyncio.wait_for(self._start_task, timeout=5.0)
            except TimeoutError, asyncio.CancelledError:
                self._start_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._start_task

        self._connected = False
        logger.info("Discord platform stopped")

    def on_message(self, handler: Callable[[IncomingMessage], Awaitable[None]]) -> None:
        """Register the workflow callback for inbound messages."""
        self._message_handler = handler

    @property
    def is_connected(self) -> bool:
        """Return whether Discord startup completed."""
        return self._connected
