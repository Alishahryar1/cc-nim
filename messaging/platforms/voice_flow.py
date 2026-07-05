"""Shared voice-note flow for messaging platform adapters."""

import contextlib
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from core.anthropic import format_user_error_preview

from ..models import IncomingMessage
from ..voice import PendingVoiceRegistry, VoiceTranscriptionService

AUDIO_EXTENSIONS = (".ogg", ".mp4", ".mp3", ".wav", ".m4a")
VOICE_DISABLED_MESSAGE = "Voice notes are disabled."
VOICE_TRANSCRIPTION_ERROR_MESSAGE = (
    "Could not transcribe voice note. Please try again or send text."
)

MessageHandler = Callable[[IncomingMessage], Awaitable[None]]
QueueSend = Callable[..., Awaitable[str | None]]
QueueDelete = Callable[..., Awaitable[None]]


@dataclass(frozen=True)
class VoiceNoteRequest:
    """Platform-normalized voice-note input."""

    platform: str
    chat_id: str
    user_id: str
    message_id: str
    raw_event: Any
    content_type: str
    temp_suffix: str
    status_text: str
    download_to: Callable[[Path], Awaitable[None]]
    reply_text: Callable[[str], Awaitable[None]]
    reply_to_message_id: str | None = None
    status_parse_mode: str | None = None
    message_thread_id: str | None = None
    username: str | None = None


def is_audio_metadata(filename: str | None, content_type: str | None) -> bool:
    """Return whether attachment metadata describes an audio file."""
    normalized_content_type = (content_type or "").lower()
    normalized_filename = (filename or "").lower()
    return normalized_content_type.startswith("audio/") or any(
        normalized_filename.endswith(extension) for extension in AUDIO_EXTENSIONS
    )


def audio_suffix_from_metadata(
    *,
    filename: str | None = None,
    content_type: str | None = None,
    default: str = ".ogg",
) -> str:
    """Choose a temp-file suffix from platform attachment metadata."""
    normalized_filename = (filename or "").lower()
    normalized_content_type = (content_type or "").lower()

    if "m4a" in normalized_content_type:
        return ".m4a"
    if "mp4" in normalized_content_type:
        if normalized_filename.endswith(".m4a"):
            return ".m4a"
        return ".mp4"
    if "mpeg" in normalized_content_type or "mp3" in normalized_content_type:
        return ".mp3"
    if "wav" in normalized_content_type:
        return ".wav"

    for extension in AUDIO_EXTENSIONS:
        if normalized_filename.endswith(extension):
            return extension
    return default


class VoiceNoteFlow:
    """Own common voice transcription state and control flow."""

    def __init__(
        self,
        *,
        voice_note_enabled: bool,
        whisper_model: str,
        whisper_device: str,
        hf_token: str,
        nvidia_nim_api_key: str,
        log_raw_messaging_content: bool,
        log_api_error_tracebacks: bool,
    ) -> None:
        self._voice_note_enabled = voice_note_enabled
        self._whisper_model = whisper_model
        self._whisper_device = whisper_device
        self._log_raw_messaging_content = log_raw_messaging_content
        self._log_api_error_tracebacks = log_api_error_tracebacks
        self._pending_voice = PendingVoiceRegistry()
        self._voice_transcription = VoiceTranscriptionService(
            hf_token=hf_token,
            nvidia_nim_api_key=nvidia_nim_api_key,
        )

    @property
    def is_enabled(self) -> bool:
        """Return whether voice-note handling is enabled."""
        return self._voice_note_enabled

    async def reply_if_disabled(
        self, reply_text: Callable[[str], Awaitable[None]]
    ) -> bool:
        """Reply with the disabled message when voice-note handling is disabled."""
        if self._voice_note_enabled:
            return False
        await reply_text(VOICE_DISABLED_MESSAGE)
        return True

    async def register_pending_voice(
        self, chat_id: str, voice_msg_id: str, status_msg_id: str
    ) -> None:
        """Register a voice note as pending transcription."""
        await self._pending_voice.register(chat_id, voice_msg_id, status_msg_id)

    async def cancel_pending_voice(
        self, chat_id: str, reply_id: str
    ) -> tuple[str, str] | None:
        """Cancel a pending voice transcription."""
        return await self._pending_voice.cancel(chat_id, reply_id)

    async def is_voice_still_pending(self, chat_id: str, voice_msg_id: str) -> bool:
        """Return whether a voice note is still pending."""
        return await self._pending_voice.is_pending(chat_id, voice_msg_id)

    async def complete_pending_voice(
        self, chat_id: str, voice_msg_id: str, status_msg_id: str
    ) -> None:
        """Mark a voice note as no longer pending."""
        await self._pending_voice.complete(chat_id, voice_msg_id, status_msg_id)

    async def handle(
        self,
        request: VoiceNoteRequest,
        *,
        message_handler: MessageHandler | None,
        queue_send_message: QueueSend,
        queue_delete_message: QueueDelete,
    ) -> bool:
        """Transcribe a voice note and hand the resulting turn to messaging."""
        if await self.reply_if_disabled(request.reply_text):
            return True

        if message_handler is None:
            return False

        status_msg_id = await queue_send_message(
            request.chat_id,
            request.status_text,
            reply_to=request.message_id,
            parse_mode=request.status_parse_mode,
            fire_and_forget=False,
            message_thread_id=request.message_thread_id,
        )
        status_msg_id_text = str(status_msg_id)
        await self.register_pending_voice(
            request.chat_id,
            request.message_id,
            status_msg_id_text,
        )
        handed_off = False

        with tempfile.NamedTemporaryFile(
            suffix=request.temp_suffix, delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            await request.download_to(tmp_path)

            transcribed = await self._voice_transcription.transcribe(
                tmp_path,
                request.content_type,
                whisper_model=self._whisper_model,
                whisper_device=self._whisper_device,
            )

            if not await self.is_voice_still_pending(
                request.chat_id,
                request.message_id,
            ):
                await queue_delete_message(request.chat_id, status_msg_id_text)
                return True

            await self.complete_pending_voice(
                request.chat_id,
                request.message_id,
                status_msg_id_text,
            )
            handed_off = True

            incoming = IncomingMessage(
                text=transcribed,
                chat_id=request.chat_id,
                user_id=request.user_id,
                message_id=request.message_id,
                platform=request.platform,
                reply_to_message_id=request.reply_to_message_id,
                message_thread_id=request.message_thread_id,
                username=request.username,
                raw_event=request.raw_event,
                status_message_id=status_msg_id,
            )

            self._log_transcription(request, transcribed)
            await message_handler(incoming)
            return True
        except ValueError as e:
            await self._clear_failed_pending_voice(
                request,
                status_msg_id_text,
                queue_delete_message,
                handed_off=handed_off,
            )
            await request.reply_text(format_user_error_preview(e))
            return True
        except ImportError as e:
            await self._clear_failed_pending_voice(
                request,
                status_msg_id_text,
                queue_delete_message,
                handed_off=handed_off,
            )
            await request.reply_text(format_user_error_preview(e))
            return True
        except Exception as e:
            await self._clear_failed_pending_voice(
                request,
                status_msg_id_text,
                queue_delete_message,
                handed_off=handed_off,
            )
            if self._log_api_error_tracebacks:
                logger.error("Voice transcription failed: {}", e)
            else:
                logger.error(
                    "Voice transcription failed: exc_type={}",
                    type(e).__name__,
                )
            await request.reply_text(VOICE_TRANSCRIPTION_ERROR_MESSAGE)
            return True
        finally:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)

    async def _clear_failed_pending_voice(
        self,
        request: VoiceNoteRequest,
        status_msg_id: str,
        queue_delete_message: QueueDelete,
        *,
        handed_off: bool,
    ) -> None:
        await self.complete_pending_voice(
            request.chat_id,
            request.message_id,
            status_msg_id,
        )
        if not handed_off:
            with contextlib.suppress(Exception):
                await queue_delete_message(request.chat_id, status_msg_id)

    def _log_transcription(self, request: VoiceNoteRequest, transcribed: str) -> None:
        label = request.platform.upper()
        if self._log_raw_messaging_content:
            logger.info(
                "{}_VOICE: chat_id={} message_id={} transcribed={!r}",
                label,
                request.chat_id,
                request.message_id,
                (transcribed[:80] + "..." if len(transcribed) > 80 else transcribed),
            )
        else:
            logger.info(
                "{}_VOICE: chat_id={} message_id={} transcribed_len={}",
                label,
                request.chat_id,
                request.message_id,
                len(transcribed),
            )
