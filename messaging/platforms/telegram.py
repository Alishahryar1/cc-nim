"""Telegram messaging runtime."""

import asyncio
import contextlib
import os
from collections.abc import Awaitable, Callable
from typing import Any

# Opt-in to future behavior for python-telegram-bot (retry_after as timedelta).
os.environ["PTB_TIMEDELTA"] = "1"

from loguru import logger

from core.anthropic import format_user_error_preview

from ..models import IncomingMessage
from ..rendering.telegram_markdown import escape_md_v2
from .ports import InboundMessageHandler
from .telegram_inbound import (
    telegram_text_message_from_update,
    telegram_voice_request_from_update,
)
from .telegram_io import TelegramMessenger
from .voice_flow import VoiceNoteFlow

try:
    from telegram import Update
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


class TelegramRuntime:
    """Owns Telegram SDK lifecycle and inbound event handoff."""

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
    ) -> None:
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot is required. Install with: pip install python-telegram-bot"
            )

        self.bot_token = bot_token
        self.allowed_user_id = allowed_user_id
        if not self.bot_token:
            logger.warning("TELEGRAM_BOT_TOKEN not set")

        self._application: Application | None = None
        self._message_handler: InboundMessageHandler | None = None
        self._connected = False
        self._limiter: Any | None = None
        self.outbound = TelegramMessenger(
            get_application=lambda: self._application,
            get_limiter=lambda: self._limiter,
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

    async def cancel_pending_voice(
        self, chat_id: str, reply_id: str
    ) -> tuple[str, str] | None:
        """Cancel a pending voice transcription."""
        return await self._voice_flow.cancel_pending_voice(chat_id, reply_id)

    async def start(self) -> None:
        """Initialize and connect to Telegram."""
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        request = HTTPXRequest(
            connection_pool_size=8, connect_timeout=30.0, read_timeout=30.0
        )
        builder = Application.builder().token(self.bot_token).request(request)
        self._application = builder.build()

        self._application.add_handler(
            MessageHandler(filters.TEXT & (~filters.COMMAND), self._on_telegram_message)
        )
        self._application.add_handler(CommandHandler("start", self._on_start_command))
        self._application.add_handler(
            MessageHandler(filters.COMMAND, self._on_telegram_message)
        )
        self._application.add_handler(
            MessageHandler(filters.VOICE, self._on_telegram_voice)
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                await self._application.initialize()
                await self._application.start()
                if self._application.updater:
                    await self._application.updater.start_polling(
                        drop_pending_updates=False
                    )
                self._connected = True
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1)
                    logger.warning(
                        "Connection failed (attempt {}/{}): {}. Retrying in {}s...",
                        attempt + 1,
                        max_retries,
                        e,
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Failed to connect after {} attempts", max_retries)
                    raise

        from ..limiter import MessagingRateLimiter

        self._limiter = await MessagingRateLimiter.get_instance(
            rate_limit=self._messaging_rate_limit,
            rate_window=self._messaging_rate_window,
        )

        try:
            target = self.allowed_user_id
            if target:
                startup_text = (
                    f"🚀 *{escape_md_v2('Claude Code Proxy is online!')}* "
                    f"{escape_md_v2('(Bot API)')}"
                )
                await self.outbound.send_message(target, startup_text)
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
        """Stop Telegram polling and SDK resources."""
        if self._application and self._application.updater:
            await self._application.updater.stop()
            await self._application.stop()
            await self._application.shutdown()

        self._connected = False
        logger.info("Telegram platform stopped")

    def on_message(self, handler: Callable[[IncomingMessage], Awaitable[None]]) -> None:
        """Register the workflow callback for inbound messages."""
        self._message_handler = handler

    @property
    def is_connected(self) -> bool:
        """Return whether Telegram startup completed."""
        return self._connected

    async def _on_start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if update.message:
            await update.message.reply_text("👋 Hello! I am the Claude Code Proxy Bot.")
        await self._on_telegram_message(update, context)

    async def _on_telegram_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        incoming = telegram_text_message_from_update(
            update,
            allowed_user_id=self.allowed_user_id,
            log_raw_messaging_content=self._log_raw_messaging_content,
        )
        if incoming is None or self._message_handler is None:
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
                    incoming.chat_id,
                    f"❌ *{escape_md_v2('Error:')}* {escape_md_v2(format_user_error_preview(e))}",
                    reply_to=incoming.message_id,
                    message_thread_id=incoming.message_thread_id,
                    parse_mode="MarkdownV2",
                )

    async def _on_telegram_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        message = update.message

        async def _reply_text(text: str) -> None:
            if message is not None:
                await message.reply_text(text)

        if await self._voice_flow.reply_if_disabled(_reply_text):
            return

        request = telegram_voice_request_from_update(
            update,
            context,
            allowed_user_id=self.allowed_user_id,
        )
        if request is None:
            return

        await self._voice_flow.handle(
            request,
            message_handler=self._message_handler,
            queue_send_message=self.outbound.queue_send_message,
            queue_delete_message=self.outbound.queue_delete_message,
        )
