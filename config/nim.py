"""NVIDIA NIM settings (strict validation)."""

from typing import Optional, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class NimSettings(BaseSettings):
    """Strictly validated NVIDIA NIM settings."""

    temperature: float = Field(1.0, ge=0.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    top_k: int = -1
    max_tokens: int = Field(81920, ge=1)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)

    min_p: float = Field(0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.0, ge=0.0)

    seed: Optional[int] = None
    stop: Optional[str] = None

    parallel_tool_calls: bool = True
    return_tokens_as_token_ids: bool = False
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False

    min_tokens: int = Field(0, ge=0)
    chat_template: Optional[str] = None
    request_id: Optional[str] = None

    reasoning_effort: Literal["low", "medium", "high"] = "high"
    include_reasoning: bool = True

    model_config = SettingsConfigDict(
        env_prefix="NVIDIA_NIM_",
        # Rely on global load_dotenv in config.settings to avoid
        # reading unrelated .env keys into this settings model.
        extra="forbid",
    )

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v):
        if v < -1:
            raise ValueError("top_k must be -1 or >= 0")
        return v

    @field_validator("seed", mode="before")
    @classmethod
    def parse_optional_int(cls, v):
        if v == "" or v is None:
            return None
        return int(v)

    @field_validator("stop", "chat_template", "request_id", mode="before")
    @classmethod
    def parse_optional_str(cls, v):
        if v == "":
            return None
        return v
