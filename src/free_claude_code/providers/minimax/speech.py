"""MiniMax synchronous text-to-speech transport and response parsing."""

import json
from typing import Any, Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

MINIMAX_TTS_ENDPOINTS: Final = {
    "global": "https://api.minimax.io/v1/t2a_v2",
    "china": "https://api.minimaxi.com/v1/t2a_v2",
}
MINIMAX_SPEECH_MODELS: Final = (
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
    "speech-02-hd",
    "speech-02-turbo",
    "speech-01-hd",
    "speech-01-turbo",
)
_AUDIO_MEDIA_TYPES: Final = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "pcm": "audio/L16",
}


class MiniMaxSpeechError(RuntimeError):
    """A safe speech synthesis failure without upstream response contents."""


class MiniMaxSpeechRequest(BaseModel):
    """Supported request fields for the synchronous MiniMax T2A operation."""

    model_config = ConfigDict(extra="forbid")

    model: Literal[
        "speech-2.8-hd",
        "speech-2.8-turbo",
        "speech-2.6-hd",
        "speech-2.6-turbo",
        "speech-02-hd",
        "speech-02-turbo",
        "speech-01-hd",
        "speech-01-turbo",
    ] = "speech-2.8-hd"
    text: str = Field(min_length=1, max_length=10_000)
    stream: bool = False
    language_boost: str | None = None
    output_format: Literal["hex"] = "hex"
    voice_setting: dict[str, Any] | None = None
    pronunciation_dict: dict[str, Any] | None = None
    audio_setting: dict[str, Any] | None = None
    voice_modify: dict[str, Any] | None = None
    subtitle_enable: bool | None = None

    @property
    def media_type(self) -> str:
        """Return the response media type requested through ``audio_setting``."""
        audio_format = (self.audio_setting or {}).get("format", "mp3")
        return _AUDIO_MEDIA_TYPES.get(str(audio_format), "application/octet-stream")

    def upstream_payload(self) -> dict[str, Any]:
        """Serialize only documented request fields for the upstream API."""
        return self.model_dump(exclude_none=True)


class MiniMaxSpeechClient:
    """Call the regional MiniMax T2A endpoint and return decoded audio bytes."""

    def __init__(
        self,
        *,
        api_key: str,
        region: Literal["global", "china"] = "global",
        proxy: str = "",
        timeout: httpx.Timeout | float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("MiniMax speech synthesis requires MINIMAX_API_KEY.")
        self._api_key = api_key
        self._endpoint = MINIMAX_TTS_ENDPOINTS[region]
        self._proxy = proxy or None
        self._timeout = timeout
        self._transport = transport

    @property
    def endpoint(self) -> str:
        """Return the selected regional endpoint."""
        return self._endpoint

    async def synthesize(self, request: MiniMaxSpeechRequest) -> bytes:
        """Generate audio and decode non-streaming JSON or streaming SSE chunks."""
        try:
            async with httpx.AsyncClient(
                proxy=self._proxy,
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    self._endpoint,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request.upstream_payload(),
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise MiniMaxSpeechError("MiniMax speech request failed.") from exc

        payloads = _response_payloads(response)
        audio_parts: list[bytes] = []
        terminal = False
        for payload in payloads:
            _raise_for_upstream_error(payload)
            data = payload.get("data")
            if not isinstance(data, dict):
                continue
            audio = data.get("audio")
            if audio:
                if not isinstance(audio, str):
                    raise MiniMaxSpeechError("MiniMax returned invalid audio data.")
                try:
                    audio_parts.append(bytes.fromhex(audio))
                except ValueError as exc:
                    raise MiniMaxSpeechError(
                        "MiniMax returned invalid hex audio data."
                    ) from exc
            terminal = terminal or data.get("status") == 2

        if not terminal or not audio_parts:
            raise MiniMaxSpeechError("MiniMax returned an incomplete speech response.")
        return b"".join(audio_parts)


def _response_payloads(response: httpx.Response) -> list[dict[str, Any]]:
    content_type = response.headers.get("content-type", "").lower()
    try:
        if "text/event-stream" not in content_type:
            parsed = response.json()
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list) and all(
                isinstance(item, dict) for item in parsed
            ):
                return parsed
            raise MiniMaxSpeechError("MiniMax returned an invalid JSON response.")

        payloads: list[dict[str, Any]] = []
        for line in response.text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            raw = line.removeprefix("data:").strip()
            if not raw or raw == "[DONE]":
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise MiniMaxSpeechError("MiniMax returned an invalid stream event.")
            payloads.append(item)
        return payloads
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MiniMaxSpeechError("MiniMax returned malformed speech data.") from exc


def _raise_for_upstream_error(payload: dict[str, Any]) -> None:
    base_response = payload.get("base_resp")
    if not isinstance(base_response, dict) or base_response.get("status_code") != 0:
        raise MiniMaxSpeechError("MiniMax rejected the speech request.")
