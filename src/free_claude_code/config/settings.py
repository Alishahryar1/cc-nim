"""Flat application settings schema loaded by Pydantic Settings."""

from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import HTTP_CONNECT_TIMEOUT_DEFAULT
from .env_files import env_file_override, settings_env_files
from .nim import NimSettings
from .provider_catalog import BEDROCK_DEFAULT_BASE, SUPPORTED_PROVIDER_IDS
from .reasoning import ReasoningPreference


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    open_router_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    open_router_credential_strategy: str = Field(
        default="sequential", validation_alias="OPENROUTER_CREDENTIAL_STRATEGY"
    )

    mistral_api_key: str = Field(default="", validation_alias="MISTRAL_API_KEY")
    mistral_credential_strategy: str = Field(
        default="sequential", validation_alias="MISTRAL_CREDENTIAL_STRATEGY"
    )

    codestral_api_key: str = Field(default="", validation_alias="CODESTRAL_API_KEY")
    codestral_credential_strategy: str = Field(
        default="sequential", validation_alias="CODESTRAL_CREDENTIAL_STRATEGY"
    )

    deepseek_api_key: str = Field(default="", validation_alias="DEEPSEEK_API_KEY")
    deepseek_credential_strategy: str = Field(
        default="sequential", validation_alias="DEEPSEEK_CREDENTIAL_STRATEGY"
    )

    kimi_api_key: str = Field(default="", validation_alias="KIMI_API_KEY")
    kimi_credential_strategy: str = Field(
        default="sequential", validation_alias="KIMI_CREDENTIAL_STRATEGY"
    )

    kimi_code_api_key: str = Field(default="", validation_alias="KIMI_CODE_API_KEY")
    kimi_code_credential_strategy: str = Field(
        default="sequential", validation_alias="KIMI_CODE_CREDENTIAL_STRATEGY"
    )

    wafer_api_key: str = Field(default="", validation_alias="WAFER_API_KEY")
    wafer_credential_strategy: str = Field(
        default="sequential", validation_alias="WAFER_CREDENTIAL_STRATEGY"
    )

    minimax_api_key: str = Field(default="", validation_alias="MINIMAX_API_KEY")
    minimax_credential_strategy: str = Field(
        default="sequential", validation_alias="MINIMAX_CREDENTIAL_STRATEGY"
    )

    opencode_api_key: str = Field(default="", validation_alias="OPENCODE_API_KEY")
    opencode_credential_strategy: str = Field(
        default="sequential", validation_alias="OPENCODE_CREDENTIAL_STRATEGY"
    )

    vercel_ai_gateway_api_key: str = Field(
        default="", validation_alias="AI_GATEWAY_API_KEY"
    )
    vercel_ai_gateway_credential_strategy: str = Field(
        default="sequential", validation_alias="VERCEL_AI_GATEWAY_CREDENTIAL_STRATEGY"
    )

    bedrock_api_key: str = Field(
        default="", validation_alias="AWS_BEARER_TOKEN_BEDROCK"
    )
    bedrock_base_url: str = Field(
        default=BEDROCK_DEFAULT_BASE,
        validation_alias="BEDROCK_BASE_URL",
    )
    bedrock_credential_strategy: str = Field(
        default="sequential", validation_alias="BEDROCK_CREDENTIAL_STRATEGY"
    )

    huggingface_api_key: str = Field(default="", validation_alias="HUGGINGFACE_API_KEY")
    huggingface_credential_strategy: str = Field(
        default="sequential", validation_alias="HUGGINGFACE_CREDENTIAL_STRATEGY"
    )

    cohere_api_key: str = Field(default="", validation_alias="COHERE_API_KEY")
    cohere_credential_strategy: str = Field(
        default="sequential", validation_alias="COHERE_CREDENTIAL_STRATEGY"
    )

    github_models_token: str = Field(default="", validation_alias="GITHUB_MODELS_TOKEN")
    github_models_credential_strategy: str = Field(
        default="sequential", validation_alias="GITHUB_MODELS_CREDENTIAL_STRATEGY"
    )

    sambanova_api_key: str = Field(default="", validation_alias="SAMBANOVA_API_KEY")
    sambanova_credential_strategy: str = Field(
        default="sequential", validation_alias="SAMBANOVA_CREDENTIAL_STRATEGY"
    )

    zai_api_key: str = Field(default="", validation_alias="ZAI_API_KEY")
    zai_credential_strategy: str = Field(
        default="sequential", validation_alias="ZAI_CREDENTIAL_STRATEGY"
    )

    fireworks_api_key: str = Field(default="", validation_alias="FIREWORKS_API_KEY")
    fireworks_credential_strategy: str = Field(
        default="sequential", validation_alias="FIREWORKS_CREDENTIAL_STRATEGY"
    )

    cloudflare_api_token: str = Field(
        default="", validation_alias="CLOUDFLARE_API_TOKEN"
    )
    cloudflare_account_id: str = Field(
        default="", validation_alias="CLOUDFLARE_ACCOUNT_ID"
    )
    cloudflare_credential_strategy: str = Field(
        default="sequential", validation_alias="CLOUDFLARE_CREDENTIAL_STRATEGY"
    )

    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    gemini_credential_strategy: str = Field(
        default="sequential", validation_alias="GEMINI_CREDENTIAL_STRATEGY"
    )

    vertex_project_id: str = Field(default="", validation_alias="VERTEX_PROJECT_ID")
    vertex_location: str = Field(default="global", validation_alias="VERTEX_LOCATION")
    vertex_credential_strategy: str = Field(
        default="sequential", validation_alias="VERTEX_CREDENTIAL_STRATEGY"
    )

    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_credential_strategy: str = Field(
        default="sequential", validation_alias="GROQ_CREDENTIAL_STRATEGY"
    )

    cerebras_api_key: str = Field(default="", validation_alias="CEREBRAS_API_KEY")
    cerebras_credential_strategy: str = Field(
        default="sequential", validation_alias="CEREBRAS_CREDENTIAL_STRATEGY"
    )

    ollama_api_key: str = Field(default="", validation_alias="OLLAMA_API_KEY")
    ollama_credential_strategy: str = Field(
        default="sequential", validation_alias="OLLAMA_CREDENTIAL_STRATEGY"
    )

    siliconflow_api_key: str = Field(default="", validation_alias="SILICONFLOW_API_KEY")
    siliconflow_credential_strategy: str = Field(
        default="sequential", validation_alias="SILICONFLOW_CREDENTIAL_STRATEGY"
    )

    messaging_platform: str = Field(
        default="discord", validation_alias="MESSAGING_PLATFORM"
    )
    messaging_rate_limit: int = Field(
        default=1, validation_alias="MESSAGING_RATE_LIMIT"
    )
    messaging_rate_window: float = Field(
        default=1.0, validation_alias="MESSAGING_RATE_WINDOW"
    )

    nvidia_nim_api_key: str = ""
    nvidia_nim_credential_strategy: str = Field(
        default="sequential", validation_alias="NVIDIA_NIM_CREDENTIAL_STRATEGY"
    )

    lm_studio_base_url: str = Field(
        default="http://localhost:1234/v1",
        validation_alias="LM_STUDIO_BASE_URL",
    )
    llamacpp_base_url: str = Field(
        default="http://localhost:8080/v1",
        validation_alias="LLAMACPP_BASE_URL",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )

    model: str = "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"
    model_fable: str | None = Field(default=None, validation_alias="MODEL_FABLE")
    model_opus: str | None = Field(default=None, validation_alias="MODEL_OPUS")
    model_sonnet: str | None = Field(default=None, validation_alias="MODEL_SONNET")
    model_haiku: str | None = Field(default=None, validation_alias="MODEL_HAIKU")

    fallback_providers: str = Field(default="", validation_alias="FALLBACK_PROVIDERS")

    nvidia_nim_proxy: str = Field(default="", validation_alias="NVIDIA_NIM_PROXY")
    open_router_proxy: str = Field(default="", validation_alias="OPENROUTER_PROXY")
    mistral_proxy: str = Field(default="", validation_alias="MISTRAL_PROXY")
    codestral_proxy: str = Field(default="", validation_alias="CODESTRAL_PROXY")
    lmstudio_proxy: str = Field(default="", validation_alias="LMSTUDIO_PROXY")
    llamacpp_proxy: str = Field(default="", validation_alias="LLAMACPP_PROXY")
    kimi_proxy: str = Field(default="", validation_alias="KIMI_PROXY")
    kimi_code_proxy: str = Field(default="", validation_alias="KIMI_CODE_PROXY")
    wafer_proxy: str = Field(default="", validation_alias="WAFER_PROXY")
    minimax_proxy: str = Field(default="", validation_alias="MINIMAX_PROXY")
    opencode_proxy: str = Field(default="", validation_alias="OPENCODE_PROXY")
    opencode_go_proxy: str = Field(default="", validation_alias="OPENCODE_GO_PROXY")
    vercel_ai_gateway_proxy: str = Field(
        default="", validation_alias="VERCEL_AI_GATEWAY_PROXY"
    )
    bedrock_proxy: str = Field(default="", validation_alias="BEDROCK_PROXY")
    huggingface_proxy: str = Field(default="", validation_alias="HUGGINGFACE_PROXY")
    cohere_proxy: str = Field(default="", validation_alias="COHERE_PROXY")
    github_models_proxy: str = Field(default="", validation_alias="GITHUB_MODELS_PROXY")
    sambanova_proxy: str = Field(default="", validation_alias="SAMBANOVA_PROXY")
    zai_proxy: str = Field(default="", validation_alias="ZAI_PROXY")
    fireworks_proxy: str = Field(default="", validation_alias="FIREWORKS_PROXY")
    cloudflare_proxy: str = Field(default="", validation_alias="CLOUDFLARE_PROXY")
    gemini_proxy: str = Field(default="", validation_alias="GEMINI_PROXY")
    vertex_proxy: str = Field(default="", validation_alias="VERTEX_PROXY")
    groq_proxy: str = Field(default="", validation_alias="GROQ_PROXY")
    cerebras_proxy: str = Field(default="", validation_alias="CEREBRAS_PROXY")
    ollama_cloud_proxy: str = Field(default="", validation_alias="OLLAMA_CLOUD_PROXY")
    siliconflow_proxy: str = Field(default="", validation_alias="SILICONFLOW_PROXY")

    provider_rate_limit: int = Field(default=40, validation_alias="PROVIDER_RATE_LIMIT")
    provider_rate_window: int = Field(
        default=60, validation_alias="PROVIDER_RATE_WINDOW"
    )
    provider_max_concurrency: int = Field(
        default=5, validation_alias="PROVIDER_MAX_CONCURRENCY"
    )
    reasoning_policy: ReasoningPreference = Field(
        default=ReasoningPreference.CLIENT,
        validation_alias="REASONING_POLICY",
    )
    reasoning_fable: ReasoningPreference = Field(
        default=ReasoningPreference.INHERIT,
        validation_alias="REASONING_FABLE",
    )
    reasoning_opus: ReasoningPreference = Field(
        default=ReasoningPreference.INHERIT,
        validation_alias="REASONING_OPUS",
    )
    reasoning_sonnet: ReasoningPreference = Field(
        default=ReasoningPreference.INHERIT,
        validation_alias="REASONING_SONNET",
    )
    reasoning_haiku: ReasoningPreference = Field(
        default=ReasoningPreference.INHERIT,
        validation_alias="REASONING_HAIKU",
    )

    http_read_timeout: float = Field(
        default=120.0, validation_alias="HTTP_READ_TIMEOUT"
    )
    http_write_timeout: float = Field(
        default=10.0, validation_alias="HTTP_WRITE_TIMEOUT"
    )
    http_connect_timeout: float = Field(
        default=HTTP_CONNECT_TIMEOUT_DEFAULT,
        validation_alias="HTTP_CONNECT_TIMEOUT",
    )

    fast_prefix_detection: bool = True

    enable_network_probe_mock: bool = True
    enable_title_generation_skip: bool = True
    enable_suggestion_mode_skip: bool = True
    enable_filepath_extraction_mock: bool = True

    enable_web_server_tools: bool = Field(
        default=False, validation_alias="ENABLE_WEB_SERVER_TOOLS"
    )
    web_fetch_allowed_schemes: str = Field(
        default="http,https", validation_alias="WEB_FETCH_ALLOWED_SCHEMES"
    )
    web_fetch_allow_private_networks: bool = Field(
        default=False, validation_alias="WEB_FETCH_ALLOW_PRIVATE_NETWORKS"
    )

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_raw_api_payloads: bool = Field(
        default=False, validation_alias="LOG_RAW_API_PAYLOADS"
    )
    log_raw_sse_events: bool = Field(
        default=False, validation_alias="LOG_RAW_SSE_EVENTS"
    )
    log_api_error_tracebacks: bool = Field(
        default=False, validation_alias="LOG_API_ERROR_TRACEBACKS"
    )
    log_raw_messaging_content: bool = Field(
        default=False, validation_alias="LOG_RAW_MESSAGING_CONTENT"
    )
    log_raw_cli_diagnostics: bool = Field(
        default=False, validation_alias="LOG_RAW_CLI_DIAGNOSTICS"
    )
    log_messaging_error_details: bool = Field(
        default=False, validation_alias="LOG_MESSAGING_ERROR_DETAILS"
    )
    debug_platform_edits: bool = Field(
        default=False, validation_alias="DEBUG_PLATFORM_EDITS"
    )
    debug_subagent_stack: bool = Field(
        default=False, validation_alias="DEBUG_SUBAGENT_STACK"
    )

    nim: NimSettings = Field(default_factory=NimSettings)

    voice_note_enabled: bool = Field(
        default=True, validation_alias="VOICE_NOTE_ENABLED"
    )
    whisper_device: str = Field(default="cpu", validation_alias="WHISPER_DEVICE")
    whisper_model: str = Field(default="base", validation_alias="WHISPER_MODEL")

    telegram_bot_token: str | None = None
    allowed_telegram_user_id: str | None = None
    telegram_proxy_url: str = Field(default="", validation_alias="TELEGRAM_PROXY_URL")
    discord_bot_token: str | None = Field(
        default=None, validation_alias="DISCORD_BOT_TOKEN"
    )
    allowed_discord_channels: str | None = Field(
        default=None, validation_alias="ALLOWED_DISCORD_CHANNELS"
    )
    allowed_dir: str = ""
    max_message_log_entries_per_chat: int | None = Field(
        default=None, validation_alias="MAX_MESSAGE_LOG_ENTRIES_PER_CHAT"
    )

    token_saver_mode: str = Field(default="none", validation_alias="TOKEN_SAVER_MODE")

    host: str = "0.0.0.0"
    port: int = 8082
    open_admin_browser: bool = Field(default=True, validation_alias="FCC_OPEN_BROWSER")
    anthropic_auth_token: str = Field(
        default="", validation_alias="ANTHROPIC_AUTH_TOKEN"
    )

    @field_validator(
        "telegram_bot_token",
        "allowed_telegram_user_id",
        "discord_bot_token",
        "allowed_discord_channels",
        "model_fable",
        "model_opus",
        "model_sonnet",
        "model_haiku",
        mode="before",
    )
    @classmethod
    def parse_optional_str(cls, v: Any) -> Any:
        if v == "":
            return None
        return v

    @field_validator("max_message_log_entries_per_chat", mode="before")
    @classmethod
    def parse_optional_log_cap(cls, v: Any) -> Any:
        if v == "" or v is None:
            return None
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(valid)}, got {v!r}")
        return upper

    @field_validator("reasoning_policy")
    @classmethod
    def validate_root_reasoning_policy(
        cls, value: ReasoningPreference
    ) -> ReasoningPreference:
        if value is ReasoningPreference.INHERIT:
            raise ValueError("REASONING_POLICY cannot inherit")
        return value

    @field_validator("whisper_device")
    @classmethod
    def validate_whisper_device(cls, v: str) -> str:
        if v not in ("cpu", "cuda", "nvidia_nim"):
            raise ValueError(
                f"whisper_device must be 'cpu', 'cuda', or 'nvidia_nim', got {v!r}"
            )
        return v

    @field_validator("messaging_platform")
    @classmethod
    def validate_messaging_platform(cls, v: str) -> str:
        if v not in ("telegram", "discord", "none"):
            raise ValueError(
                f"messaging_platform must be 'telegram', 'discord', or 'none', got {v!r}"
            )
        return v

    @field_validator("messaging_rate_limit")
    @classmethod
    def validate_messaging_rate_limit(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("messaging_rate_limit must be > 0")
        return v

    @field_validator("messaging_rate_window")
    @classmethod
    def validate_messaging_rate_window(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("messaging_rate_window must be > 0")
        return float(v)

    @field_validator("web_fetch_allowed_schemes")
    @classmethod
    def validate_web_fetch_allowed_schemes(cls, v: str) -> str:
        schemes = [part.strip().lower() for part in v.split(",") if part.strip()]
        if not schemes:
            raise ValueError("web_fetch_allowed_schemes must list at least one scheme")
        for scheme in schemes:
            if not scheme.isascii() or not scheme.isalpha():
                raise ValueError(
                    f"Invalid URL scheme in web_fetch_allowed_schemes: {scheme!r}"
                )
        return ",".join(schemes)

    @field_validator(
        "model", "model_fable", "model_opus", "model_sonnet", "model_haiku"
    )
    @classmethod
    def validate_model_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if "/" not in v:
            raise ValueError(
                f"Model must be prefixed with provider type. Valid providers: {', '.join(SUPPORTED_PROVIDER_IDS)}. Format: provider_type/model/name"
            )
        provider = v.split("/", 1)[0]
        if provider not in SUPPORTED_PROVIDER_IDS:
            supported = ", ".join(f"'{p}'" for p in SUPPORTED_PROVIDER_IDS)
            raise ValueError(f"Invalid provider: '{provider}'. Supported: {supported}")
        return v

    @model_validator(mode="after")
    def check_nvidia_nim_api_key(self) -> Settings:
        if (
            self.voice_note_enabled
            and self.whisper_device == "nvidia_nim"
            and not self.nvidia_nim_api_key.strip()
        ):
            raise ValueError(
                "NVIDIA_NIM_API_KEY is required when WHISPER_DEVICE is 'nvidia_nim'. Set it in your .env file."
            )
        return self

    @model_validator(mode="after")
    def prefer_dotenv_anthropic_auth_token(self) -> Settings:
        """Let explicit .env auth config override stale shell/client tokens."""
        dotenv_value = env_file_override(self.model_config, "ANTHROPIC_AUTH_TOKEN")
        if dotenv_value is not None:
            self.anthropic_auth_token = dotenv_value
        return self

    model_config = SettingsConfigDict(
        env_file=settings_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
