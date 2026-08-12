"""HTTP request models for speech synthesis."""

from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

_AUDIO_MEDIA_TYPES: Final = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "pcm": "audio/L16",
}


class MiniMaxAudioSetting(BaseModel):
    """Audio settings with a constrained output format."""

    model_config = ConfigDict(extra="allow")

    format: Literal["mp3", "wav", "flac", "pcm"] = "mp3"


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
    audio_setting: MiniMaxAudioSetting | None = None
    voice_modify: dict[str, Any] | None = None
    subtitle_enable: bool | None = None

    @property
    def media_type(self) -> str:
        """Return the response media type requested through ``audio_setting``."""
        audio_format = self.audio_setting.format if self.audio_setting else "mp3"
        return _AUDIO_MEDIA_TYPES[audio_format]

    def upstream_payload(self) -> dict[str, Any]:
        """Serialize only documented request fields for the upstream API."""
        return self.model_dump(exclude_none=True)
