"""Tests for NVIDIA NIM voice transcription wiring."""

from pathlib import Path
from unittest.mock import patch

from messaging.transcription import transcribe_audio


def test_transcribe_audio_nvidia_nim_forwards_api_key(tmp_path: Path) -> None:
    wav = tmp_path / "stub.wav"
    wav.write_bytes(b"\x00" * 128)
    with patch("messaging.transcription.transcribe_nvidia_nim_audio") as nim_fn:
        nim_fn.return_value = "ok"
        out = transcribe_audio(
            wav,
            "audio/wav",
            whisper_model="openai/whisper-large-v3",
            whisper_device="nvidia_nim",
            nvidia_nim_api_key="test-nim-key",
        )
    nim_fn.assert_called_once_with(
        wav, "openai/whisper-large-v3", api_key="test-nim-key"
    )
    assert out == "ok"


def test_nim_asr_model_map_entries_are_real_function_ids() -> None:
    from providers.nvidia_nim.voice import _NIM_ASR_MODEL_MAP

    for function_id, language_code in _NIM_ASR_MODEL_MAP.values():
        assert function_id
        assert function_id.strip().lower() != "none"
        # Hosted NIM function-id is a lowercase UUID string.
        parts = function_id.split("-")
        assert len(parts) == 5
        assert all(p for p in parts)
        assert language_code is not None
