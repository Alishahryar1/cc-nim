import json

import httpx
import pytest

from free_claude_code.api.speech import MiniMaxSpeechRequest
from free_claude_code.providers.minimax import (
    MINIMAX_TTS_ENDPOINTS,
    MiniMaxSpeechClient,
    MiniMaxSpeechError,
)


@pytest.mark.asyncio
async def test_speech_request_uses_china_endpoint_and_decodes_hex_audio() -> None:
    seen_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(
            200,
            json={
                "data": {"audio": "494433", "status": 2},
                "base_resp": {"status_code": 0},
            },
        )

    client = MiniMaxSpeechClient(
        api_key="test-key",
        region="china",
        transport=httpx.MockTransport(handler),
    )
    request = MiniMaxSpeechRequest(
        text="Hello",
        audio_setting={"format": "mp3", "sample_rate": 32000},
        voice_setting={"voice_id": "English_expressive_narrator"},
    )

    assert await client.synthesize(request.upstream_payload()) == b"ID3"
    assert seen_request is not None
    assert str(seen_request.url) == MINIMAX_TTS_ENDPOINTS["china"]
    assert seen_request.headers["authorization"] == "Bearer test-key"
    payload = json.loads(seen_request.content)
    assert payload["model"] == "speech-2.8-hd"
    assert payload["text"] == "Hello"
    assert payload["stream"] is False
    assert request.media_type == "audio/mpeg"


def test_speech_request_preserves_all_supported_optional_fields() -> None:
    request = MiniMaxSpeechRequest(
        text="Hello",
        stream=True,
        language_boost="English",
        output_format="hex",
        voice_setting={"voice_id": "English_expressive_narrator"},
        pronunciation_dict={"tone": ["FCC/F C C"]},
        audio_setting={"format": "flac"},
        voice_modify={"pitch": 1},
        subtitle_enable=True,
    )

    assert request.upstream_payload() == {
        "model": "speech-2.8-hd",
        "text": "Hello",
        "stream": True,
        "language_boost": "English",
        "output_format": "hex",
        "voice_setting": {"voice_id": "English_expressive_narrator"},
        "pronunciation_dict": {"tone": ["FCC/F C C"]},
        "audio_setting": {"format": "flac"},
        "voice_modify": {"pitch": 1},
        "subtitle_enable": True,
    }
    assert request.media_type == "audio/flac"


@pytest.mark.asyncio
async def test_speech_stream_collects_chunks_until_terminal_status() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=(
                'data: {"data":{"audio":"4944","status":1},'
                '"base_resp":{"status_code":0}}\n\n'
                'data: {"data":{"audio":"33","status":2},'
                '"base_resp":{"status_code":0}}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    client = MiniMaxSpeechClient(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )

    request = MiniMaxSpeechRequest(text="Hello", stream=True)
    audio = await client.synthesize(request.upstream_payload())

    assert audio == b"ID3"


@pytest.mark.asyncio
async def test_speech_response_rejects_upstream_status_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {"audio": "494433", "status": 2},
                "base_resp": {"status_code": 1004},
            },
        )

    client = MiniMaxSpeechClient(
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(MiniMaxSpeechError, match="rejected"):
        request = MiniMaxSpeechRequest(text="Hello")
        await client.synthesize(request.upstream_payload())
