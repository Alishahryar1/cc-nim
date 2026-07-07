from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from messaging.platforms.voice_flow import (
    VOICE_DISABLED_MESSAGE,
    VOICE_TRANSCRIPTION_ERROR_MESSAGE,
    VoiceNoteFlow,
    VoiceNoteRequest,
    audio_suffix_from_metadata,
    is_audio_metadata,
)


def _flow(*, enabled: bool = True) -> VoiceNoteFlow:
    return VoiceNoteFlow(
        voice_note_enabled=enabled,
        whisper_model="base",
        whisper_device="cpu",
        huggingface_api_key="",
        nvidia_nim_api_key="",
        log_raw_messaging_content=False,
        log_api_error_tracebacks=False,
    )


def _request(
    *,
    download_to=None,
    reply_text=None,
    message_id: str = "voice",
) -> VoiceNoteRequest:
    async def default_download_to(path: Path) -> None:
        path.write_bytes(b"voice")

    return VoiceNoteRequest(
        platform="telegram",
        chat_id="chat",
        user_id="user",
        message_id=message_id,
        raw_event={"raw": True},
        content_type="audio/ogg",
        temp_suffix=".ogg",
        status_text="transcribing",
        status_parse_mode="MarkdownV2",
        message_thread_id="thread",
        reply_to_message_id="reply",
        download_to=download_to or default_download_to,
        reply_text=reply_text or AsyncMock(),
    )


@pytest.mark.asyncio
async def test_voice_flow_success_builds_incoming_message(monkeypatch) -> None:
    flow = _flow()
    transcribe = AsyncMock(return_value="hello from voice")
    monkeypatch.setattr(flow._voice_transcription, "transcribe", transcribe)
    handler = AsyncMock()
    queue_send = AsyncMock(return_value="status")
    queue_delete = AsyncMock()
    downloaded_paths: list[Path] = []

    async def download_to(path: Path) -> None:
        downloaded_paths.append(path)
        path.write_bytes(b"voice")

    handled = await flow.handle(
        _request(download_to=download_to),
        message_handler=handler,
        queue_send_message=queue_send,
        queue_delete_messages=queue_delete,
    )

    assert handled is True
    queue_send.assert_awaited_once_with(
        "chat",
        "transcribing",
        reply_to="voice",
        parse_mode="MarkdownV2",
        fire_and_forget=False,
        message_thread_id="thread",
    )
    queue_delete.assert_not_awaited()
    handler.assert_awaited_once()
    incoming = handler.call_args.args[0]
    assert incoming.text == "hello from voice"
    assert incoming.chat_id == "chat"
    assert incoming.user_id == "user"
    assert incoming.message_id == "voice"
    assert incoming.reply_to_message_id == "reply"
    assert incoming.message_thread_id == "thread"
    assert incoming.status_message_id == "status"
    assert downloaded_paths and not downloaded_paths[0].exists()


@pytest.mark.asyncio
async def test_voice_flow_disabled_replies_without_transcribing(monkeypatch) -> None:
    flow = _flow(enabled=False)
    transcribe = AsyncMock(return_value="should not run")
    monkeypatch.setattr(flow._voice_transcription, "transcribe", transcribe)
    reply_text = AsyncMock()

    handled = await flow.handle(
        _request(reply_text=reply_text),
        message_handler=AsyncMock(),
        queue_send_message=AsyncMock(),
        queue_delete_messages=AsyncMock(),
    )

    assert handled is True
    reply_text.assert_awaited_once_with(VOICE_DISABLED_MESSAGE)
    transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_voice_flow_cancelled_transcription_deletes_status(monkeypatch) -> None:
    flow = _flow()

    async def canceling_transcribe(*args, **kwargs) -> str:
        await flow.cancel_pending_voice("chat", "voice")
        return "ignored"

    monkeypatch.setattr(
        flow._voice_transcription,
        "transcribe",
        AsyncMock(side_effect=canceling_transcribe),
    )
    handler = AsyncMock()
    queue_send = AsyncMock(return_value="status")
    queue_delete = AsyncMock()

    handled = await flow.handle(
        _request(),
        message_handler=handler,
        queue_send_message=queue_send,
        queue_delete_messages=queue_delete,
    )

    assert handled is True
    handler.assert_not_awaited()
    queue_delete.assert_awaited_once_with("chat", ["status"])


@pytest.mark.asyncio
async def test_voice_flow_download_failure_cleans_pending_state(monkeypatch) -> None:
    flow = _flow()
    transcribe = AsyncMock(return_value="should not run")
    monkeypatch.setattr(flow._voice_transcription, "transcribe", transcribe)
    reply_text = AsyncMock()
    queue_delete = AsyncMock()

    async def failing_download(_path: Path) -> None:
        raise RuntimeError("download failed")

    handled = await flow.handle(
        _request(download_to=failing_download, reply_text=reply_text),
        message_handler=AsyncMock(),
        queue_send_message=AsyncMock(return_value="status"),
        queue_delete_messages=queue_delete,
    )

    assert handled is True
    transcribe.assert_not_awaited()
    queue_delete.assert_awaited_once_with("chat", ["status"])
    reply_text.assert_awaited_once_with(VOICE_TRANSCRIPTION_ERROR_MESSAGE)
    assert await flow.cancel_pending_voice("chat", "voice") is None


@pytest.mark.asyncio
async def test_voice_flow_transcription_failure_cleans_pending_state(
    monkeypatch,
) -> None:
    flow = _flow()
    monkeypatch.setattr(
        flow._voice_transcription,
        "transcribe",
        AsyncMock(side_effect=RuntimeError("transcription failed")),
    )
    reply_text = AsyncMock()
    queue_delete = AsyncMock()

    handled = await flow.handle(
        _request(reply_text=reply_text),
        message_handler=AsyncMock(),
        queue_send_message=AsyncMock(return_value="status"),
        queue_delete_messages=queue_delete,
    )

    assert handled is True
    queue_delete.assert_awaited_once_with("chat", ["status"])
    reply_text.assert_awaited_once_with(VOICE_TRANSCRIPTION_ERROR_MESSAGE)
    assert await flow.cancel_pending_voice("chat", "voice") is None


@pytest.mark.asyncio
async def test_voice_flow_handler_failure_cleans_pending_without_deleting_status(
    monkeypatch,
) -> None:
    flow = _flow()
    monkeypatch.setattr(
        flow._voice_transcription,
        "transcribe",
        AsyncMock(return_value="hello from voice"),
    )
    reply_text = AsyncMock()
    queue_delete = AsyncMock()

    async def failing_handler(_incoming) -> None:
        raise RuntimeError("handler failed")

    handled = await flow.handle(
        _request(reply_text=reply_text),
        message_handler=failing_handler,
        queue_send_message=AsyncMock(return_value="status"),
        queue_delete_messages=queue_delete,
    )

    assert handled is True
    queue_delete.assert_not_awaited()
    reply_text.assert_awaited_once_with(VOICE_TRANSCRIPTION_ERROR_MESSAGE)
    assert await flow.cancel_pending_voice("chat", "voice") is None


def test_audio_metadata_helpers() -> None:
    assert is_audio_metadata("voice.ogg", "application/octet-stream") is True
    assert is_audio_metadata("file.txt", "audio/ogg") is True
    assert is_audio_metadata("file.txt", "text/plain") is False
    assert (
        audio_suffix_from_metadata(filename="voice.ogg", content_type="audio/mp4")
        == ".mp4"
    )
    assert (
        audio_suffix_from_metadata(filename="clip.m4a", content_type="audio/mp4")
        == ".m4a"
    )
    assert (
        audio_suffix_from_metadata(filename="clip.m4a", content_type="audio/mpeg")
        == ".mp3"
    )
    assert audio_suffix_from_metadata(content_type="audio/mpeg") == ".mp3"
    assert audio_suffix_from_metadata(filename="clip.m4a") == ".m4a"
    assert audio_suffix_from_metadata(content_type="audio/mp4") == ".mp4"
    assert audio_suffix_from_metadata(content_type="audio/wav") == ".wav"
