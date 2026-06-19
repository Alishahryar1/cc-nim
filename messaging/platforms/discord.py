"""
Discord Platform Adapter

Implements MessagingPlatform for Discord using discord.py.
"""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any, cast

from loguru import logger

from core.anthropic import format_user_error_preview

from ..models import IncomingMessage
from ..rendering.discord_markdown import format_status_discord
from .base import MessagingPlatform
from .outbox import PlatformOutbox
from .voice_flow import (
    VoiceNoteFlow,
    VoiceNoteRequest,
    audio_suffix_from_metadata,
    is_audio_metadata,
)

_discord_module: Any = None
try:
    import discord as _discord_import

    _discord_module = _discord_import
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

DISCORD_MESSAGE_LIMIT = 2000


def _get_discord() -> Any:
    """Return the discord module. Raises if not available."""
    if not DISCORD_AVAILABLE or _discord_module is None:
        raise ImportError(
            "discord.py is required. Install with: pip install discord.py"
        )
    return _discord_module


def _parse_allowed_channels(raw: str | None) -> set[str]:
    """Parse comma-separated channel IDs into a set of strings."""
    if not raw or not raw.strip():
        return set()
    return {s.strip() for s in raw.split(",") if s.strip()}


if DISCORD_AVAILABLE and _discord_module is not None:
    _discord = _discord_module

    class _DiscordClient(_discord.Client):
        """Internal Discord client that forwards events to DiscordPlatform."""

        def __init__(
            self,
            platform: DiscordPlatform,
            intents: _discord.Intents,
        ) -> None:
            super().__init__(intents=intents)
            self._platform = platform

        async def on_ready(self) -> None:
            """Called when the bot is ready."""
            self._platform._connected = True
            logger.info("Discord platform connected")

        async def on_message(self, message: Any) -> None:
            """Handle incoming Discord messages."""
            await self._platform._handle_client_message(message)
else:
    _DiscordClient = None


class DiscordPlatform(MessagingPlatform):
    """
    Discord messaging platform adapter.

    Uses discord.py for Discord access.
    Requires a Bot Token from Discord Developer Portal and message_content intent.
    """

    name = "discord"

    def __init__(
        self,
        bot_token: str | None = None,
        allowed_channel_ids: str | None = None,
        *,
        voice_note_enabled: bool = True,
        whisper_model: str = "base",
        whisper_device: str = "cpu",
        hf_token: str = "",
        nvidia_nim_api_key: str = "",
        messaging_rate_limit: int = 1,
        messaging_rate_window: float = 1.0,
        log_raw_messaging_content: bool = False,
        log_api_error_tracebacks: bool = False,
    ):
        if not DISCORD_AVAILABLE:
            raise ImportError(
                "discord.py is required. Install with: pip install discord.py"
            )

        self.bot_token = bot_token
        self.allowed_channel_ids = _parse_allowed_channels(allowed_channel_ids)

        if not self.bot_token:
            logger.warning("DISCORD_BOT_TOKEN not set")

        discord = _get_discord()
        intents = discord.Intents.default()
        intents.message_content = True

        assert _DiscordClient is not None
        self._client = _DiscordClient(self, intents)
        self._message_handler: Callable[[IncomingMessage], Awaitable[None]] | None = (
            None
        )
        self._connected = False
        self._limiter: Any | None = None
        self._start_task: asyncio.Task | None = None

        async def send_operation(
            chat_id: str,
            text: str,
            reply_to: str | None,
            parse_mode: str | None,
            message_thread_id: str | None,
        ) -> str:
            return await self.send_message(
                chat_id,
                text,
                reply_to,
                parse_mode,
                message_thread_id,
            )

        async def edit_operation(
            chat_id: str,
            message_id: str,
            text: str,
            parse_mode: str | None,
        ) -> None:
            await self.edit_message(chat_id, message_id, text, parse_mode)

        async def delete_operation(chat_id: str, message_id: str) -> None:
            await self.delete_message(chat_id, message_id)

        async def delete_many_operation(
            chat_id: str,
            message_ids: list[str],
        ) -> None:
            await self.delete_messages(chat_id, message_ids)

        self._outbox = PlatformOutbox(
            get_limiter=lambda: self._limiter,
            send=send_operation,
            edit=edit_operation,
            delete=delete_operation,
            delete_many=delete_many_operation,
        )
        self._voice_flow = VoiceNoteFlow(
            voice_note_enabled=voice_note_enabled,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            hf_token=hf_token,
            nvidia_nim_api_key=nvidia_nim_api_key,
            log_raw_messaging_content=log_raw_messaging_content,
            log_api_error_tracebacks=log_api_error_tracebacks,
        )
        self._messaging_rate_limit = messaging_rate_limit
        self._messaging_rate_window = messaging_rate_window
        self._log_raw_messaging_content = log_raw_messaging_content
        self._log_api_error_tracebacks = log_api_error_tracebacks

    async def _handle_client_message(self, message: Any) -> None:
        """Adapter entry point used by the internal discord client."""
        await self._on_discord_message(message)

    async def _register_pending_voice(
        self, chat_id: str, voice_msg_id: str, status_msg_id: str
    ) -> None:
        """Register a voice note as pending transcription."""
        await self._voice_flow.register_pending_voice(
            chat_id,
            voice_msg_id,
            status_msg_id,
        )

    async def cancel_pending_voice(
        self, chat_id: str, reply_id: str
    ) -> tuple[str, str] | None:
        """Cancel a pending voice transcription. Returns (voice_msg_id, status_msg_id) if found."""
        return await self._voice_flow.cancel_pending_voice(chat_id, reply_id)

    async def _is_voice_still_pending(self, chat_id: str, voice_msg_id: str) -> bool:
        """Check if a voice note is still pending (not cancelled)."""
        return await self._voice_flow.is_voice_still_pending(chat_id, voice_msg_id)

    def _get_audio_attachment(self, message: Any) -> Any | None:
        """Return first audio attachment, or None."""
        for att in message.attachments:
            if is_audio_metadata(att.filename, att.content_type):
                return att
        return None

    async def _handle_voice_note(
        self, message: Any, attachment: Any, channel_id: str
    ) -> bool:
        """Handle voice/audio attachment. Returns True if handled."""
        user_id = str(message.author.id)
        message_id = str(message.id)
        reply_to = (
            str(message.reference.message_id)
            if message.reference and message.reference.message_id
            else None
        )
        ct = attachment.content_type or "audio/ogg"

        async def _download_to(tmp_path) -> None:
            await attachment.save(str(tmp_path))

        async def _reply_text(text: str) -> None:
            await message.reply(text)

        return await self._voice_flow.handle(
            VoiceNoteRequest(
                platform="discord",
                chat_id=channel_id,
                user_id=user_id,
                message_id=message_id,
                raw_event=message,
                content_type=ct,
                temp_suffix=audio_suffix_from_metadata(
                    filename=attachment.filename,
                    content_type=attachment.content_type,
                ),
                status_text=format_status_discord("Transcribing voice note..."),
                reply_to_message_id=reply_to,
                username=message.author.display_name,
                download_to=_download_to,
                reply_text=_reply_text,
            ),
            message_handler=self._message_handler,
            queue_send_message=self.queue_send_message,
            queue_delete_message=self.queue_delete_message,
        )

    async def _on_discord_message(self, message: Any) -> None:
        """Handle incoming Discord messages."""
        if message.author.bot:
            return

        channel_id = str(message.channel.id)

        if not self.allowed_channel_ids or channel_id not in self.allowed_channel_ids:
            return

        # Handle voice/audio attachments when message has no text content
        if not message.content:
            audio_att = self._get_audio_attachment(message)
            if audio_att:
                await self._handle_voice_note(message, audio_att, channel_id)
                return
            return

        user_id = str(message.author.id)
        message_id = str(message.id)
        reply_to = (
            str(message.reference.message_id)
            if message.reference and message.reference.message_id
            else None
        )

        raw_content = message.content or ""
        if self._log_raw_messaging_content:
            text_preview = raw_content[:80]
            if len(raw_content) > 80:
                text_preview += "..."
            logger.info(
                "DISCORD_MSG: chat_id={} message_id={} reply_to={} text_preview={!r}",
                channel_id,
                message_id,
                reply_to,
                text_preview,
            )
        else:
            logger.info(
                "DISCORD_MSG: chat_id={} message_id={} reply_to={} text_len={}",
                channel_id,
                message_id,
                reply_to,
                len(raw_content),
            )

        if not self._message_handler:
            return

        incoming = IncomingMessage(
            text=message.content,
            chat_id=channel_id,
            user_id=user_id,
            message_id=message_id,
            platform="discord",
            reply_to_message_id=reply_to,
            username=message.author.display_name,
            raw_event=message,
        )

        try:
            await self._message_handler(incoming)
        except Exception as e:
            if self._log_api_error_tracebacks:
                logger.error("Error handling message: {}", e)
            else:
                logger.error("Error handling message: exc_type={}", type(e).__name__)
            with contextlib.suppress(Exception):
                await self.send_message(
                    channel_id,
                    format_status_discord("Error:", format_user_error_preview(e)),
                    reply_to=message_id,
                )

    def _truncate(self, text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> str:
        """Truncate text to Discord's message limit."""
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

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
        waited = 0
        while not self._connected and waited < max_wait:
            await asyncio.sleep(0.5)
            waited += 0.5

        if not self._connected:
            raise RuntimeError("Discord client failed to connect within timeout")

        logger.info("Discord platform started")

    async def stop(self) -> None:
        """Stop the bot."""
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

    async def _send_message_raw(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = None,
        message_thread_id: str | None = None,
    ) -> str:
        """Send a message to a channel."""
        channel = self._client.get_channel(int(chat_id))
        if not channel or not hasattr(channel, "send"):
            raise RuntimeError(f"Channel {chat_id} not found")

        text = self._truncate(text)
        channel = cast(Any, channel)

        discord = _get_discord()
        if reply_to:
            ref = discord.MessageReference(
                message_id=int(reply_to),
                channel_id=int(chat_id),
            )
            msg = await channel.send(content=text, reference=ref)
        else:
            msg = await channel.send(content=text)

        return str(msg.id)

    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = None,
        message_thread_id: str | None = None,
    ) -> str:
        """Send a message to a channel."""
        return await self._send_message_raw(
            chat_id,
            text,
            reply_to,
            parse_mode,
            message_thread_id,
        )

    async def _edit_message_raw(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = None,
    ) -> None:
        """Edit an existing message."""
        channel = self._client.get_channel(int(chat_id))
        if not channel or not hasattr(channel, "fetch_message"):
            raise RuntimeError(f"Channel {chat_id} not found")

        discord = _get_discord()
        channel = cast(Any, channel)
        try:
            msg = await channel.fetch_message(int(message_id))
        except discord.NotFound:
            return

        text = self._truncate(text)
        await msg.edit(content=text)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = None,
    ) -> None:
        """Edit an existing message."""
        await self._edit_message_raw(chat_id, message_id, text, parse_mode)

    async def _delete_message_raw(
        self,
        chat_id: str,
        message_id: str,
    ) -> None:
        """Delete a message from a channel."""
        channel = self._client.get_channel(int(chat_id))
        if not channel or not hasattr(channel, "fetch_message"):
            return

        discord = _get_discord()
        channel = cast(Any, channel)
        try:
            msg = await channel.fetch_message(int(message_id))
            await msg.delete()
        except discord.NotFound, discord.Forbidden:
            pass

    async def delete_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> None:
        """Delete a message from a channel."""
        await self._delete_message_raw(chat_id, message_id)

    async def _delete_messages_raw(self, chat_id: str, message_ids: list[str]) -> None:
        """Delete multiple messages (best-effort)."""
        for mid in message_ids:
            await self._delete_message_raw(chat_id, mid)

    async def delete_messages(self, chat_id: str, message_ids: list[str]) -> None:
        """Delete multiple messages (best-effort)."""
        await self._delete_messages_raw(chat_id, message_ids)

    async def queue_send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = None,
        fire_and_forget: bool = True,
        message_thread_id: str | None = None,
    ) -> str | None:
        """Enqueue a message to be sent."""
        return await self._outbox.queue_send_message(
            chat_id,
            text,
            reply_to,
            parse_mode,
            fire_and_forget,
            message_thread_id,
        )

    async def queue_edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = None,
        fire_and_forget: bool = True,
    ) -> None:
        """Enqueue a message edit."""
        await self._outbox.queue_edit_message(
            chat_id,
            message_id,
            text,
            parse_mode,
            fire_and_forget,
        )

    async def queue_delete_message(
        self,
        chat_id: str,
        message_id: str,
        fire_and_forget: bool = True,
    ) -> None:
        """Enqueue a message delete."""
        await self._outbox.queue_delete_message(chat_id, message_id, fire_and_forget)

    async def queue_delete_messages(
        self,
        chat_id: str,
        message_ids: list[str],
        fire_and_forget: bool = True,
    ) -> None:
        """Enqueue a bulk delete."""
        await self._outbox.queue_delete_messages(
            chat_id,
            message_ids,
            fire_and_forget,
        )

    def fire_and_forget(self, task: Awaitable[Any]) -> None:
        """Execute a coroutine without awaiting it."""
        self._outbox.fire_and_forget(task)

    def on_message(
        self,
        handler: Callable[[IncomingMessage], Awaitable[None]],
    ) -> None:
        """Register a message handler callback."""
        self._message_handler = handler

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected
