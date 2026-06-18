"""
Telegram Platform Adapter

Implements MessagingPlatform for Telegram using python-telegram-bot.
"""

import asyncio
import contextlib
import os

# Opt-in to future behavior for python-telegram-bot (retry_after as timedelta)
# This must be set BEFORE importing telegram.error
os.environ["PTB_TIMEDELTA"] = "1"

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from core.anthropic import format_user_error_preview

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes

from ..models import IncomingMessage
from ..rendering.telegram_markdown import escape_md_v2, format_status
from .base import MessagingPlatform
from .outbox import PlatformOutbox
from .voice_flow import VoiceNoteFlow, VoiceNoteRequest, audio_suffix_from_metadata

# Optional import - python-telegram-bot may not be installed
try:
    from telegram import Update
    from telegram.error import NetworkError, RetryAfter, TelegramError
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )
    from telegram.request import HTTPXRequest

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


class TelegramPlatform(MessagingPlatform):
    """
    Telegram messaging platform adapter.

    Uses python-telegram-bot (BoT API) for Telegram access.
    Requires a Bot Token from @BotFather.
    """

    name = "telegram"

    def __init__(
        self,
        bot_token: str | None = None,
        allowed_user_id: str | None = None,
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
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot is required. Install with: pip install python-telegram-bot"
            )

        self.bot_token = bot_token
        self.allowed_user_id = allowed_user_id

        if not self.bot_token:
            # We don't raise here to allow instantiation for testing/conditional logic,
            # but start() will fail.
            logger.warning("TELEGRAM_BOT_TOKEN not set")

        self._application: Application | None = None
        self._message_handler: Callable[[IncomingMessage], Awaitable[None]] | None = (
            None
        )
        self._connected = False
        self._limiter: Any | None = None  # Will be MessagingRateLimiter

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

    async def _register_pending_voice(
        self, chat_id: str, voice_msg_id: str, status_msg_id: str
    ) -> None:
        """Register a voice note as pending transcription (for /clear reply during transcription)."""
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

    async def start(self) -> None:
        """Initialize and connect to Telegram."""
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        # Configure request with longer timeouts
        request = HTTPXRequest(
            connection_pool_size=8, connect_timeout=30.0, read_timeout=30.0
        )

        # Build Application
        builder = Application.builder().token(self.bot_token).request(request)
        self._application = builder.build()

        # Register Internal Handlers
        # We catch ALL text messages and commands to forward them
        self._application.add_handler(
            MessageHandler(filters.TEXT & (~filters.COMMAND), self._on_telegram_message)
        )
        self._application.add_handler(CommandHandler("start", self._on_start_command))
        # Catch-all for other commands if needed, or let them fall through
        self._application.add_handler(
            MessageHandler(filters.COMMAND, self._on_telegram_message)
        )
        # Voice note handler
        self._application.add_handler(
            MessageHandler(filters.VOICE, self._on_telegram_voice)
        )

        # Initialize internal components with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await self._application.initialize()
                await self._application.start()

                # Start polling (non-blocking way for integration)
                if self._application.updater:
                    await self._application.updater.start_polling(
                        drop_pending_updates=False
                    )

                self._connected = True
                break
            except (NetworkError, Exception) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1)
                    logger.warning(
                        f"Connection failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to connect after {max_retries} attempts")
                    raise

        # Initialize rate limiter
        from ..limiter import MessagingRateLimiter

        self._limiter = await MessagingRateLimiter.get_instance(
            rate_limit=self._messaging_rate_limit,
            rate_window=self._messaging_rate_window,
        )

        # Send startup notification
        try:
            target = self.allowed_user_id
            if target:
                startup_text = (
                    f"🚀 *{escape_md_v2('Claude Code Proxy is online!')}* "
                    f"{escape_md_v2('(Bot API)')}"
                )
                await self.send_message(
                    target,
                    startup_text,
                )
        except Exception as e:
            if self._log_api_error_tracebacks:
                logger.warning("Could not send startup message: {}", e)
            else:
                logger.warning(
                    "Could not send startup message: exc_type={}",
                    type(e).__name__,
                )

        logger.info("Telegram platform started (Bot API)")

    async def stop(self) -> None:
        """Stop the bot."""
        if self._application and self._application.updater:
            await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()

        self._connected = False
        logger.info("Telegram platform stopped")

    async def _with_retry(
        self, func: Callable[..., Awaitable[Any]], *args, **kwargs
    ) -> Any:
        """Helper to execute a function with exponential backoff on network errors."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except (TimeoutError, NetworkError) as e:
                if "Message is not modified" in str(e):
                    return None
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"Telegram API network error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Telegram API failed after {max_retries} attempts: {e}"
                    )
                    raise
            except RetryAfter as e:
                # Telegram explicitly tells us to wait (PTB_TIMEDELTA: retry_after is timedelta)
                from datetime import timedelta

                retry_after = e.retry_after
                if isinstance(retry_after, timedelta):
                    wait_secs = retry_after.total_seconds()
                else:
                    wait_secs = float(retry_after)

                logger.warning(f"Rate limited by Telegram, waiting {wait_secs}s...")
                await asyncio.sleep(wait_secs)
                # We don't increment attempt here, as this is a specific instruction
                return await func(*args, **kwargs)
            except TelegramError as e:
                # Non-network Telegram errors
                err_lower = str(e).lower()
                if "message is not modified" in err_lower:
                    return None
                # Best-effort no-op cases (common during chat cleanup / /clear).
                if any(
                    x in err_lower
                    for x in [
                        "message to edit not found",
                        "message to delete not found",
                        "message can't be deleted",
                        "message can't be edited",
                        "not enough rights to delete",
                    ]
                ):
                    return None
                if "Can't parse entities" in str(e) and kwargs.get("parse_mode"):
                    logger.warning("Markdown failed, retrying without parse_mode")
                    kwargs["parse_mode"] = None
                    return await func(*args, **kwargs)
                raise

    async def _send_message_raw(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = "MarkdownV2",
        message_thread_id: str | None = None,
    ) -> str:
        """Send a message to a chat."""
        app = self._application
        if not app or not app.bot:
            raise RuntimeError("Telegram application or bot not initialized")

        async def _do_send(parse_mode=parse_mode):
            bot = app.bot
            kwargs: dict[str, Any] = {
                "chat_id": chat_id,
                "text": text,
                "reply_to_message_id": int(reply_to) if reply_to else None,
                "parse_mode": parse_mode,
            }
            if message_thread_id is not None:
                kwargs["message_thread_id"] = int(message_thread_id)
            msg = await bot.send_message(**kwargs)
            return str(msg.message_id)

        return await self._with_retry(_do_send, parse_mode=parse_mode)

    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
        parse_mode: str | None = "MarkdownV2",
        message_thread_id: str | None = None,
    ) -> str:
        """Send a message to a chat."""
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
        parse_mode: str | None = "MarkdownV2",
    ) -> None:
        """Edit an existing message."""
        app = self._application
        if not app or not app.bot:
            raise RuntimeError("Telegram application or bot not initialized")

        async def _do_edit(parse_mode=parse_mode):
            bot = app.bot
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=int(message_id),
                text=text,
                parse_mode=parse_mode,
            )

        await self._with_retry(_do_edit, parse_mode=parse_mode)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: str | None = "MarkdownV2",
    ) -> None:
        """Edit an existing message."""
        await self._edit_message_raw(chat_id, message_id, text, parse_mode)

    async def _delete_message_raw(
        self,
        chat_id: str,
        message_id: str,
    ) -> None:
        """Delete a message from a chat."""
        app = self._application
        if not app or not app.bot:
            raise RuntimeError("Telegram application or bot not initialized")

        async def _do_delete():
            bot = app.bot
            await bot.delete_message(chat_id=chat_id, message_id=int(message_id))

        await self._with_retry(_do_delete)

    async def delete_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> None:
        """Delete a message from a chat."""
        await self._delete_message_raw(chat_id, message_id)

    async def _delete_messages_raw(self, chat_id: str, message_ids: list[str]) -> None:
        """Delete multiple messages (best-effort)."""
        if not message_ids:
            return
        app = self._application
        if not app or not app.bot:
            raise RuntimeError("Telegram application or bot not initialized")

        # PTB supports bulk deletion via delete_messages; fall back to per-message.
        bot = app.bot
        if hasattr(bot, "delete_messages"):

            async def _do_bulk():
                mids = []
                for mid in message_ids:
                    try:
                        mids.append(int(mid))
                    except Exception:
                        continue
                if not mids:
                    return None
                # delete_messages accepts a sequence of ints (up to 100).
                await bot.delete_messages(chat_id=chat_id, message_ids=mids)

            await self._with_retry(_do_bulk)
            return

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
        parse_mode: str | None = "MarkdownV2",
        fire_and_forget: bool = True,
        message_thread_id: str | None = None,
    ) -> str | None:
        """Enqueue a message to be sent (using limiter)."""
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
        parse_mode: str | None = "MarkdownV2",
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
        """Enqueue a bulk delete (if supported) or a sequence of deletes."""
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

    async def _on_start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /start command."""
        if update.message:
            await update.message.reply_text("👋 Hello! I am the Claude Code Proxy Bot.")
        # We can also treat this as a message if we want it to trigger something
        await self._on_telegram_message(update, context)

    async def _on_telegram_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming updates."""
        if (
            not update.message
            or not update.message.text
            or not update.effective_user
            or not update.effective_chat
        ):
            return

        user_id = str(update.effective_user.id)
        chat_id = str(update.effective_chat.id)

        # Security check
        if self.allowed_user_id and user_id != str(self.allowed_user_id).strip():
            logger.warning(f"Unauthorized access attempt from {user_id}")
            return

        message_id = str(update.message.message_id)
        reply_to = (
            str(update.message.reply_to_message.message_id)
            if update.message.reply_to_message
            else None
        )
        thread_id = (
            str(update.message.message_thread_id)
            if getattr(update.message, "message_thread_id", None) is not None
            else None
        )
        raw_text = update.message.text or ""
        if self._log_raw_messaging_content:
            text_preview = raw_text[:80]
            if len(raw_text) > 80:
                text_preview += "..."
            logger.info(
                "TELEGRAM_MSG: chat_id={} message_id={} reply_to={} text_preview={!r}",
                chat_id,
                message_id,
                reply_to,
                text_preview,
            )
        else:
            logger.info(
                "TELEGRAM_MSG: chat_id={} message_id={} reply_to={} text_len={}",
                chat_id,
                message_id,
                reply_to,
                len(raw_text),
            )

        if not self._message_handler:
            return

        incoming = IncomingMessage(
            text=update.message.text,
            chat_id=chat_id,
            user_id=user_id,
            message_id=message_id,
            platform="telegram",
            reply_to_message_id=reply_to,
            message_thread_id=thread_id,
            raw_event=update,
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
                    chat_id,
                    f"❌ *{escape_md_v2('Error:')}* {escape_md_v2(format_user_error_preview(e))}",
                    reply_to=incoming.message_id,
                    message_thread_id=thread_id,
                    parse_mode="MarkdownV2",
                )

    async def _on_telegram_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming voice messages."""
        message = update.message
        effective_user = update.effective_user
        effective_chat = update.effective_chat
        if (
            message is None
            or message.voice is None
            or effective_user is None
            or effective_chat is None
        ):
            return
        voice = message.voice

        async def _reply_text(text: str) -> None:
            await message.reply_text(text)

        if await self._voice_flow.reply_if_disabled(_reply_text):
            return

        user_id = str(effective_user.id)
        chat_id = str(effective_chat.id)

        if self.allowed_user_id and user_id != str(self.allowed_user_id).strip():
            logger.warning(f"Unauthorized voice access attempt from {user_id}")
            return

        thread_id = (
            str(message.message_thread_id)
            if getattr(message, "message_thread_id", None) is not None
            else None
        )
        message_id = str(message.message_id)
        reply_to = (
            str(message.reply_to_message.message_id)
            if message.reply_to_message
            else None
        )

        async def _download_to(tmp_path) -> None:
            tg_file = await context.bot.get_file(voice.file_id)
            await tg_file.download_to_drive(custom_path=str(tmp_path))

        await self._voice_flow.handle(
            VoiceNoteRequest(
                platform="telegram",
                chat_id=chat_id,
                user_id=user_id,
                message_id=message_id,
                raw_event=update,
                content_type=voice.mime_type or "audio/ogg",
                temp_suffix=audio_suffix_from_metadata(content_type=voice.mime_type),
                status_text=format_status("⏳", "Transcribing voice note..."),
                status_parse_mode="MarkdownV2",
                message_thread_id=thread_id,
                reply_to_message_id=reply_to,
                download_to=_download_to,
                reply_text=_reply_text,
            ),
            message_handler=self._message_handler,
            queue_send_message=self.queue_send_message,
            queue_delete_message=self.queue_delete_message,
        )
