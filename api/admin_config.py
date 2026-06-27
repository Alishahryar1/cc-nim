"""Admin UI configuration manifest and managed env persistence."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values
from pydantic import ValidationError

from config.paths import managed_env_path
from config.provider_catalog import PROVIDER_CATALOG
from config.settings import Settings

FieldType = Literal[
    "text",
    "secret",
    "number",
    "boolean",
    "tri_boolean",
    "select",
    "textarea",
]
SourceType = Literal[
    "default",
    "template",
    "repo_env",
    "managed_env",
    "explicit_env_file",
    "process",
]

MASKED_SECRET = "********"


@dataclass(frozen=True, slots=True)
class ConfigSectionSpec:
    """A group of config fields rendered together in the admin UI."""

    section_id: str
    label: str
    description: str
    advanced: bool = False


@dataclass(frozen=True, slots=True)
class ConfigFieldSpec:
    """Typed metadata for one env-backed admin setting."""

    key: str
    label: str
    section_id: str
    field_type: FieldType = "text"
    settings_attr: str | None = None
    default: str = ""
    options: tuple[str, ...] = ()
    secret: bool = False
    advanced: bool = False
    restart_required: bool = False
    session_sensitive: bool = False
    description: str = ""


SECTIONS: tuple[ConfigSectionSpec, ...] = (
    ConfigSectionSpec(
        "providers",
        "Providers",
        "Provider keys, local endpoints, and proxy settings.",
    ),
    ConfigSectionSpec(
        "models",
        "Model Routing",
        "Provider-prefixed models used for Claude model tiers.",
    ),
    ConfigSectionSpec(
        "thinking",
        "Thinking",
        "Global and tier-specific thinking behavior.",
    ),
    ConfigSectionSpec(
        "runtime",
        "Runtime",
        "Server API token, rate limits, timeouts, and process settings.",
    ),
    ConfigSectionSpec(
        "messaging",
        "Messaging",
        "Discord, Telegram, CLI workspace, and session settings.",
    ),
    ConfigSectionSpec(
        "voice",
        "Voice",
        "Voice note transcription settings.",
    ),
    ConfigSectionSpec(
        "web_tools",
        "Web Tools",
        "Local Anthropic web_search and web_fetch behavior.",
    ),
    ConfigSectionSpec(
        "diagnostics",
        "Diagnostics",
        "Logging and debugging flags.",
        advanced=True,
    ),
    ConfigSectionSpec(
        "smoke",
        "Smoke Tests",
        "Optional live smoke-test model overrides.",
        advanced=True,
    ),
)


FIELDS: tuple[ConfigFieldSpec, ...] = (
    ConfigFieldSpec(
        "NVIDIA_NIM_API_KEY",
        "NVIDIA NIM API Key",
        "providers",
        "secret",
        settings_attr="nvidia_nim_api_key",
        secret=True,
        description="Used by NVIDIA NIM chat and optional NIM voice transcription.",
    ),
    ConfigFieldSpec(
        "OPENROUTER_API_KEY",
        "OpenRouter API Key",
        "providers",
        "secret",
        settings_attr="open_router_api_key",
        secret=True,
    ),
    ConfigFieldSpec(
        "MISTRAL_API_KEY",
        "Mistral API Key",
        "providers",
        "secret",
        settings_attr="mistral_api_key",
        secret=True,
        description=(
            "Mistral La Plateforme (api.mistral.ai); Experiment plan is free tier with rate limits."
        ),
    ),
    ConfigFieldSpec(
        "CODESTRAL_API_KEY",
        "Codestral API Key",
        "providers",
        "secret",
        settings_attr="codestral_api_key",
        secret=True,
        description=(
            "Mistral Codestral endpoint (codestral.mistral.ai); distinct from Mistral "
            "La Plateforme ``MISTRAL_API_KEY``. See Mistral docs for coding/FIM domains."
        ),
    ),
    ConfigFieldSpec(
        "DEEPSEEK_API_KEY",
        "DeepSeek API Key",
        "providers",
        "secret",
        settings_attr="deepseek_api_key",
        secret=True,
    ),
    ConfigFieldSpec(
        "KIMI_API_KEY",
        "Kimi API Key",
        "providers",
        "secret",
        settings_attr="kimi_api_key",
        secret=True,
    ),
    ConfigFieldSpec(
        "WAFER_API_KEY",
        "Wafer API Key",
        "providers",
        "secret",
        settings_attr="wafer_api_key",
        secret=True,
    ),
    ConfigFieldSpec(
        "OPENCODE_API_KEY",
        "OpenCode API Key",
        "providers",
        "secret",
        settings_attr="opencode_api_key",
        secret=True,
        description=(
            "OpenCode Zen curated gateway (opencode.ai/zen/v1) and OpenCode Go subscription "
            "gateway (opencode.ai/zen/go/v1); single key from opencode.ai/auth."
        ),
    ),
    ConfigFieldSpec(
        "ZAI_API_KEY",
        "Z.ai API Key",
        "providers",
        "secret",
        settings_attr="zai_api_key",
        secret=True,
        description="Z.ai Coding Plan API key.",
    ),
    ConfigFieldSpec(
        "FIREWORKS_API_KEY",
        "Fireworks API Key",
        "providers",
        "secret",
        settings_attr="fireworks_api_key",
        secret=True,
        description="Fireworks AI inference API key.",
    ),
    ConfigFieldSpec(
        "GEMINI_API_KEY",
        "Gemini API Key",
        "providers",
        "secret",
        settings_attr="gemini_api_key",
        secret=True,
        description=(
            "Google AI Studio Gemini API key (Google AI Studio / Gemini API "
            "[OpenAI-compatible](https://ai.google.dev/gemini-api/docs/openai)); "
            "free tier has per-model rate limits and data may be used for improvement "
            "outside the UK/CH/EEA/EU."
        ),
    ),
    ConfigFieldSpec(
        "GROQ_API_KEY",
        "Groq API Key",
        "providers",
        "secret",
        settings_attr="groq_api_key",
        secret=True,
        description=(
            "GroqCloud OpenAI-compatible API key ([console.groq.com/keys]("
            "https://console.groq.com/keys)); see Groq "
            "[OpenAI compatibility docs](https://console.groq.com/docs/openai)."
        ),
    ),
    ConfigFieldSpec(
        "CEREBRAS_API_KEY",
        "Cerebras API Key",
        "providers",
        "secret",
        settings_attr="cerebras_api_key",
        secret=True,
        description=(
            "Cerebras Inference API key (create in [Cloud Console](https://cloud.cerebras.ai)); "
            "see [Quickstart](https://inference-docs.cerebras.ai/quickstart) and "
            "[OpenAI compatibility](https://inference-docs.cerebras.ai/resources/openai)."
        ),
    ),
    ConfigFieldSpec(
        "AGENTROUTER_API_KEY",
        "AgentRouter API Key",
        "providers",
        "secret",
        settings_attr="agentrouter_api_key",
        secret=True,
        description=(
            "AgentRouter API key for Anthropic-compatible Messages "
            "at agentrouter.org/v1."
        ),
    ),
    ConfigFieldSpec(
        "LM_STUDIO_BASE_URL",
        "LM Studio Base URL",
        "providers",
        settings_attr="lm_studio_base_url",
        default="http://localhost:1234/v1",
    ),
    ConfigFieldSpec(
        "LLAMACPP_BASE_URL",
        "llama.cpp Base URL",
        "providers",
        settings_attr="llamacpp_base_url",
        default="http://localhost:8080/v1",
    ),
    ConfigFieldSpec(
        "OLLAMA_BASE_URL",
        "Ollama Base URL",
        "providers",
        settings_attr="ollama_base_url",
        default="http://localhost:11434",
    ),
    ConfigFieldSpec(
        "NVIDIA_NIM_PROXY",
        "NVIDIA NIM Proxy",
        "providers",
        "secret",
        settings_attr="nvidia_nim_proxy",
        secret=True,
        advanced=True,
    ),
    ConfigFieldSpec(
        "OPENROUTER_PROXY",
        "OpenRouter Proxy",
        "providers",
        "secret",
        settings_attr="open_router_proxy",
        secret=True,
        advanced=True,
    ),
    ConfigFieldSpec(
        "MISTRAL_PROXY",
        "Mistral Proxy",
        "providers",
        "secret",
        settings_attr="mistral_proxy",
        secret=True,
        advanced=True,
    ),
    ConfigFieldSpec(
        "CODESTRAL_PROXY",
        "Codestral Proxy",
        "providers",
        "secret",
        settings_attr="codestral_proxy",
        secret=True,
        advanced=True,
    ),
    ConfigFieldSpec(
        "LMSTUDIO_PROXY",
        "LM Studio Proxy",
        "providers",
        "secret",
        settings_attr="lmstudio_proxy",
        secret=True,
        advanced=True,
    ),
    ConfigFieldSpec(
        "LLAMACPP_PROXY",
        "llama.cpp Proxy",
        "providers",
        "secret",
        settings_attr="llamacpp_proxy",
        secret=True,
        advanced=True,
    ),
    ConfigFieldSpec(
        "KIMI_PROXY",
        "Kimi Proxy",
        "providers",
        "secret",
        settings_attr="kimi_proxy",
        secret=True,
        advanced=True,
    ),
    ConfigFieldSpec(
        "WAFER_PROXY",
        "Wafer Proxy",
        "providers",
        "secret",
        settings_attr="wafer_proxy",
        secret=True,
        advanced=True,
    ),
    ConfigFieldSpec(
        "OPENCODE_PROXY",
        "OpenCode Zen Proxy",
        "providers",
        "secret",
        settings_attr="opencode_proxy",
        secret=True,
        advanced=True,
    ),
    ConfigFieldSpec(
        "OPENCODE_GO_PROXY",
        "OpenCode Go Proxy",
        "providers",
        "secret",
        settings_attr="opencode_go_proxy",
        secret=True,
        advanced=True,
    ),
    ConfigFieldSpec(
        "ZAI_PROXY",
        "Z.ai Proxy",
        "providers",
        "secret",
        settings_attr="zai_proxy",
        secret=True,
        advanced=True,
    ),
    ConfigFieldSpec(
        "FIREWORKS_PROXY",
        "Fireworks Proxy",
        "providers",
        "secret",
        settings_attr="fireworks_proxy",
        secret=True,
        advanced=True,
    ),
    ConfigFieldSpec(
        "GEMINI_PROXY",
        "Gemini Proxy",
        "providers",
        "secret",
        settings_attr="gemini_proxy",
        secret=True,
        advanced=True,
    ),
    ConfigFieldSpec(
        "GROQ_PROXY",
        "Groq Proxy",
        "providers",
        "secret",
        settings_attr="groq_proxy",
        secret=True,
        advanced=True,
    ),
    ConfigFieldSpec(
        "CEREBRAS_PROXY",
        "Cerebras Proxy",
        "providers",
        "secret",
        settings_attr="cerebras_proxy",
        secret=True,
        advanced=True,
    ),
    ConfigFieldSpec(
        "AGENTROUTER_PROXY",
        "AgentRouter Proxy",
        "providers",
        "secret",
        settings_attr="agentrouter_proxy",
        secret=True,
        advanced=True,
    ),
    ConfigFieldSpec(
        "MODEL",
        "Default Model",
        "models",
        settings_attr="model",
        default="nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
        description="Fallback provider/model route for all Claude model names.",
    ),
    ConfigFieldSpec(
        "MODEL_OPUS",
        "Opus Override",
        "models",
        settings_attr="model_opus",
        description="Optional provider/model route for Opus requests.",
    ),
    ConfigFieldSpec(
        "MODEL_SONNET",
        "Sonnet Override",
        "models",
        settings_attr="model_sonnet",
        description="Optional provider/model route for Sonnet requests.",
    ),
    ConfigFieldSpec(
        "MODEL_HAIKU",
        "Haiku Override",
        "models",
        settings_attr="model_haiku",
        description="Optional provider/model route for Haiku requests.",
    ),
    ConfigFieldSpec(
        "ENABLE_MODEL_THINKING",
        "Enable Thinking",
        "thinking",
        "boolean",
        settings_attr="enable_model_thinking",
        default="true",
    ),
    ConfigFieldSpec(
        "ENABLE_OPUS_THINKING",
        "Opus Thinking",
        "thinking",
        "tri_boolean",
        settings_attr="enable_opus_thinking",
        description="Blank inherits Enable Thinking.",
    ),
    ConfigFieldSpec(
        "ENABLE_SONNET_THINKING",
        "Sonnet Thinking",
        "thinking",
        "tri_boolean",
        settings_attr="enable_sonnet_thinking",
        description="Blank inherits Enable Thinking.",
    ),
    ConfigFieldSpec(
        "ENABLE_HAIKU_THINKING",
        "Haiku Thinking",
        "thinking",
        "tri_boolean",
        settings_attr="enable_haiku_thinking",
        description="Blank inherits Enable Thinking.",
    ),
    ConfigFieldSpec(
        "ANTHROPIC_AUTH_TOKEN",
        "API/CLI Auth Token",
        "runtime",
        "secret",
        settings_attr="anthropic_auth_token",
        default="freecc",
        secret=True,
        description="Protects Claude/API access. It is not admin-page login.",
    ),
    ConfigFieldSpec(
        "PROVIDER_RATE_LIMIT",
        "Provider Rate Limit",
        "runtime",
        "number",
        settings_attr="provider_rate_limit",
        default="1",
    ),
    ConfigFieldSpec(
        "PROVIDER_RATE_WINDOW",
        "Provider Rate Window",
        "runtime",
        "number",
        settings_attr="provider_rate_window",
        default="3",
    ),
    ConfigFieldSpec(
        "PROVIDER_MAX_CONCURRENCY",
        "Provider Max Concurrency",
        "runtime",
        "number",
        settings_attr="provider_max_concurrency",
        default="5",
    ),
    ConfigFieldSpec(
        "HTTP_READ_TIMEOUT",
        "HTTP Read Timeout",
        "runtime",
        "number",
        settings_attr="http_read_timeout",
        default="300",
    ),
    ConfigFieldSpec(
        "HTTP_WRITE_TIMEOUT",
        "HTTP Write Timeout",
        "runtime",
        "number",
        settings_attr="http_write_timeout",
        default="60",
    ),
    ConfigFieldSpec(
        "HTTP_CONNECT_TIMEOUT",
        "HTTP Connect Timeout",
        "runtime",
        "number",
        settings_attr="http_connect_timeout",
        default="60",
    ),
    ConfigFieldSpec(
        "HOST",
        "Server Host",
        "runtime",
        settings_attr="host",
        default="0.0.0.0",
        restart_required=True,
    ),
    ConfigFieldSpec(
        "PORT",
        "Server Port",
        "runtime",
        "number",
        settings_attr="port",
        default="8082",
        restart_required=True,
    ),
    ConfigFieldSpec(
        "MESSAGING_PLATFORM",
        "Messaging Platform",
        "messaging",
        "select",
        settings_attr="messaging_platform",
        default="discord",
        options=("telegram", "discord", "none"),
        session_sensitive=True,
    ),
    ConfigFieldSpec(
        "MESSAGING_RATE_LIMIT",
        "Messaging Rate Limit",
        "messaging",
        "number",
        settings_attr="messaging_rate_limit",
        default="1",
        session_sensitive=True,
    ),
    ConfigFieldSpec(
        "MESSAGING_RATE_WINDOW",
        "Messaging Rate Window",
        "messaging",
        "number",
        settings_attr="messaging_rate_window",
        default="1",
        session_sensitive=True,
    ),
    ConfigFieldSpec(
        "TELEGRAM_BOT_TOKEN",
        "Telegram Bot Token",
        "messaging",
        "secret",
        settings_attr="telegram_bot_token",
        secret=True,
        session_sensitive=True,
    ),
    ConfigFieldSpec(
        "ALLOWED_TELEGRAM_USER_ID",
        "Allowed Telegram User ID",
        "messaging",
        settings_attr="allowed_telegram_user_id",
        session_sensitive=True,
    ),
    ConfigFieldSpec(
        "DISCORD_BOT_TOKEN",
        "Discord Bot Token",
        "messaging",
        "secret",
        settings_attr="discord_bot_token",
        secret=True,
        session_sensitive=True,
    ),
    ConfigFieldSpec(
        "ALLOWED_DISCORD_CHANNELS",
        "Allowed Discord Channels",
        "messaging",
        settings_attr="allowed_discord_channels",
        session_sensitive=True,
    ),
    ConfigFieldSpec(
        "ALLOWED_DIR",
        "Allowed Directory",
        "messaging",
        settings_attr="allowed_dir",
        session_sensitive=True,
    ),
    ConfigFieldSpec(
        "MAX_MESSAGE_LOG_ENTRIES_PER_CHAT",
        "Max Message Log Entries",
        "messaging",
        "number",
        settings_attr="max_message_log_entries_per_chat",
        advanced=True,
        session_sensitive=True,
    ),
    ConfigFieldSpec(
        "VOICE_NOTE_ENABLED",
        "Voice Notes",
        "voice",
        "boolean",
        settings_attr="voice_note_enabled",
        default="false",
        session_sensitive=True,
    ),
    ConfigFieldSpec(
        "WHISPER_DEVICE",
        "Whisper Device",
        "voice",
        "select",
        settings_attr="whisper_device",
        default="nvidia_nim",
        options=("cpu", "cuda", "nvidia_nim"),
        session_sensitive=True,
    ),
    ConfigFieldSpec(
        "WHISPER_MODEL",
        "Whisper Model",
        "voice",
        settings_attr="whisper_model",
        default="openai/whisper-large-v3",
        session_sensitive=True,
    ),
    ConfigFieldSpec(
        "HF_TOKEN",
        "Hugging Face Token",
        "voice",
        "secret",
        settings_attr="hf_token",
        secret=True,
        session_sensitive=True,
    ),
    ConfigFieldSpec(
        "FAST_PREFIX_DETECTION",
        "Fast Prefix Detection",
        "runtime",
        "boolean",
        settings_attr="fast_prefix_detection",
        default="true",
        advanced=True,
    ),
    ConfigFieldSpec(
        "ENABLE_NETWORK_PROBE_MOCK",
        "Network Probe Mock",
        "runtime",
        "boolean",
        settings_attr="enable_network_probe_mock",
        default="true",
        advanced=True,
    ),
    ConfigFieldSpec(
        "ENABLE_TITLE_GENERATION_SKIP",
        "Title Generation Skip",
        "runtime",
        "boolean",
        settings_attr="enable_title_generation_skip",
        default="true",
        advanced=True,
    ),
    ConfigFieldSpec(
        "ENABLE_SUGGESTION_MODE_SKIP",
        "Suggestion Mode Skip",
        "runtime",
        "boolean",
        settings_attr="enable_suggestion_mode_skip",
        default="true",
        advanced=True,
    ),
    ConfigFieldSpec(
        "ENABLE_FILEPATH_EXTRACTION_MOCK",
        "Filepath Extraction Mock",
        "runtime",
        "boolean",
        settings_attr="enable_filepath_extraction_mock",
        default="true",
        advanced=True,
    ),
    ConfigFieldSpec(
        "ENABLE_WEB_SERVER_TOOLS",
        "Web Server Tools",
        "web_tools",
        "boolean",
        settings_attr="enable_web_server_tools",
        default="true",
    ),
    ConfigFieldSpec(
        "WEB_FETCH_ALLOWED_SCHEMES",
        "Allowed Web Fetch Schemes",
        "web_tools",
        settings_attr="web_fetch_allowed_schemes",
        default="http,https",
    ),
    ConfigFieldSpec(
        "WEB_FETCH_ALLOW_PRIVATE_NETWORKS",
        "Allow Private Networks",
        "web_tools",
        "boolean",
        settings_attr="web_fetch_allow_private_networks",
        default="false",
    ),
    ConfigFieldSpec(
        "DEBUG_PLATFORM_EDITS",
        "Debug Platform Edits",
        "diagnostics",
        "boolean",
        settings_attr="debug_platform_edits",
        default="false",
        advanced=True,
    ),
    ConfigFieldSpec(
        "DEBUG_SUBAGENT_STACK",
        "Debug Subagent Stack",
        "diagnostics",
        "boolean",
        settings_attr="debug_subagent_stack",
        default="false",
        advanced=True,
    ),
    ConfigFieldSpec(
        "LOG_RAW_API_PAYLOADS",
        "Log Raw API Payloads",
        "diagnostics",
        "boolean",
        settings_attr="log_raw_api_payloads",
        default="false",
        advanced=True,
    ),
    ConfigFieldSpec(
        "LOG_RAW_SSE_EVENTS",
        "Log Raw SSE Events",
        "diagnostics",
        "boolean",
        settings_attr="log_raw_sse_events",
        default="false",
        advanced=True,
    ),
    ConfigFieldSpec(
        "LOG_API_ERROR_TRACEBACKS",
        "Log API Error Tracebacks",
        "diagnostics",
        "boolean",
        settings_attr="log_api_error_tracebacks",
        default="false",
        advanced=True,
    ),
    ConfigFieldSpec(
        "LOG_RAW_MESSAGING_CONTENT",
        "Log Raw Messaging Content",
        "diagnostics",
        "boolean",
        settings_attr="log_raw_messaging_content",
        default="false",
        advanced=True,
    ),
    ConfigFieldSpec(
        "LOG_RAW_CLI_DIAGNOSTICS",
        "Log Raw CLI Diagnostics",
        "diagnostics",
        "boolean",
        settings_attr="log_raw_cli_diagnostics",
        default="false",
        advanced=True,
    ),
    ConfigFieldSpec(
        "LOG_MESSAGING_ERROR_DETAILS",
        "Log Messaging Error Details",
        "diagnostics",
        "boolean",
        settings_attr="log_messaging_error_details",
        default="false",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_NVIDIA_NIM",
        "Smoke NVIDIA NIM Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_OPEN_ROUTER",
        "Smoke OpenRouter Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_MISTRAL",
        "Smoke Mistral Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_MISTRAL_CODESTRAL",
        "Smoke Mistral Codestral Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_DEEPSEEK",
        "Smoke DeepSeek Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_LMSTUDIO",
        "Smoke LM Studio Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_LLAMACPP",
        "Smoke llama.cpp Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_OLLAMA",
        "Smoke Ollama Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_KIMI",
        "Smoke Kimi Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_WAFER",
        "Smoke Wafer Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_OPENCODE",
        "Smoke OpenCode Zen Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_OPENCODE_GO",
        "Smoke OpenCode Go Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_ZAI",
        "Smoke Z.ai Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_FIREWORKS",
        "Smoke Fireworks Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_GEMINI",
        "Smoke Gemini Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_GROQ",
        "Smoke Groq Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_CEREBRAS",
        "Smoke Cerebras Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_NIM_MODELS",
        "Smoke NIM Models",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_NIM_EXTRA_MODELS",
        "Smoke NIM Extra Models",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_OPENROUTER_FREE_MODELS",
        "Smoke OpenRouter Free Models",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_OPENROUTER_FREE_EXTRA_MODELS",
        "Smoke OpenRouter Free Extra Models",
        "smoke",
        advanced=True,
    ),
)

FIELD_BY_KEY = {field.key: field for field in FIELDS}


def repo_env_path() -> Path:
    """Return the repo-local env path."""

    return Path(".env")


def explicit_env_path() -> Path | None:
    """Return the explicit FCC_ENV_FILE path, when configured."""

    if explicit := os.environ.get("FCC_ENV_FILE"):
        return Path(explicit)
    return None


def configured_env_files() -> tuple[tuple[SourceType, Path], ...]:
    """Return dotenv files in low-to-high precedence order."""

    files: list[tuple[SourceType, Path]] = [
        ("repo_env", repo_env_path()),
        ("managed_env", managed_env_path()),
    ]
    if explicit := explicit_env_path():
        files.append(("explicit_env_file", explicit))
    return tuple(files)


def _template_text() -> str:
    import importlib.resources

    packaged = importlib.resources.files("cli").joinpath("env.example")
    if packaged.is_file():
        return packaged.read_text("utf-8")

    source_template = Path(__file__).resolve().parents[1] / ".env.example"
    if source_template.is_file():
        return source_template.read_text(encoding="utf-8")

    return ""


def _dotenv_values_from_text(text: str) -> dict[str, str]:
    values = dotenv_values(stream=StringIO(text))
    return {key: "" if value is None else value for key, value in values.items()}


def template_values() -> dict[str, str]:
    """Return .env.example values plus manifest defaults for newer fields."""

    values = _dotenv_values_from_text(_template_text())
    for field in FIELDS:
        values.setdefault(field.key, field.default)
    return values


def _dotenv_values_from_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = dotenv_values(path)
    return {key: "" if value is None else value for key, value in values.items()}


def _field_input_key(field: ConfigFieldSpec) -> str | None:
    if field.settings_attr is None:
        return None
    model_field = Settings.model_fields[field.settings_attr]
    alias = model_field.validation_alias
    if alias is None:
        return field.settings_attr
    return str(alias)


def _is_locked_source(source: SourceType) -> bool:
    return source in {"process", "explicit_env_file"}


def _normalize_for_env(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _display_value(field: ConfigFieldSpec, value: str) -> str:
    if field.secret and value:
        return MASKED_SECRET
    return value


def _load_value_state() -> dict[str, dict[str, Any]]:
    values = template_values()
    sources: dict[str, SourceType] = {
        key: "template" if key in values else "default" for key in FIELD_BY_KEY
    }

    for source, path in configured_env_files():
        file_values = _dotenv_values_from_file(path)
        for key, value in file_values.items():
            if key in FIELD_BY_KEY:
                values[key] = value
                sources[key] = source

    for key in FIELD_BY_KEY:
        if key in os.environ:
            values[key] = os.environ[key]
            sources[key] = "process"

    return {
        key: {
            "value": values.get(key, ""),
            "source": sources.get(key, "default"),
        }
        for key in FIELD_BY_KEY
    }


def load_config_response() -> dict[str, Any]:
    """Return manifest and current config values for the admin UI."""

    state = _load_value_state()
    fields: list[dict[str, Any]] = []
    for field in FIELDS:
        entry = state[field.key]
        source = entry["source"]
        raw_value = entry["value"]
        fields.append(
            {
                "key": field.key,
                "label": field.label,
                "section": field.section_id,
                "type": field.field_type,
                "value": _display_value(field, raw_value),
                "configured": bool(str(raw_value).strip()),
                "source": source,
                "locked": _is_locked_source(source),
                "secret": field.secret,
                "advanced": field.advanced,
                "restart_required": field.restart_required,
                "session_sensitive": field.session_sensitive,
                "options": list(field.options),
                "description": field.description,
            }
        )

    return {
        "sections": [
            {
                "id": section.section_id,
                "label": section.label,
                "description": section.description,
                "advanced": section.advanced,
            }
            for section in SECTIONS
        ],
        "fields": fields,
        "paths": {
            "managed": str(managed_env_path()),
            "repo": str(repo_env_path()),
            "explicit": str(explicit_env_path()) if explicit_env_path() else None,
        },
        "provider_status": provider_config_status(state),
    }


def _target_values_with_updates(updates: Mapping[str, Any]) -> dict[str, str]:
    state = _load_value_state()
    values = template_values()

    # Preserve existing managed values when present. If no managed config exists,
    # seed the first write from effective repo values to migrate legacy setups.
    managed_values = _dotenv_values_from_file(managed_env_path())
    if managed_values:
        values.update(
            {key: val for key, val in managed_values.items() if key in values}
        )
    else:
        for key, entry in state.items():
            if entry["source"] in {"repo_env", "template", "default"}:
                values[key] = str(entry["value"])

    for key, value in updates.items():
        field = FIELD_BY_KEY.get(key)
        if field is None:
            continue
        if _is_locked_source(state[key]["source"]):
            continue
        if field.secret and value == MASKED_SECRET:
            continue
        values[key] = _normalize_for_env(value)

    for field in FIELDS:
        values.setdefault(field.key, field.default)
    return values


def _effective_values_for_validation(
    target_values: Mapping[str, str],
) -> dict[str, str]:
    values = dict(target_values)
    for key, entry in _load_value_state().items():
        if _is_locked_source(entry["source"]):
            values[key] = str(entry["value"])
    return values


def validate_values(values: Mapping[str, str]) -> tuple[bool, list[str]]:
    """Validate proposed env values against the Settings model."""

    kwargs: dict[str, Any] = {"_env_file": None}
    for field in FIELDS:
        input_key = _field_input_key(field)
        if input_key is None:
            continue
        kwargs[input_key] = values.get(field.key, "")

    try:
        Settings(**kwargs)
    except ValidationError as exc:
        return False, _format_validation_errors(exc)
    return True, []


def _format_validation_errors(exc: ValidationError) -> list[str]:
    errors: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ()))
        message = str(error.get("msg", "Invalid value"))
        errors.append(f"{loc}: {message}" if loc else message)
    return errors


def validate_updates(updates: Mapping[str, Any]) -> dict[str, Any]:
    """Validate partial admin updates and return a masked generated env preview."""

    target_values = _target_values_with_updates(updates)
    effective_values = _effective_values_for_validation(target_values)
    valid, errors = validate_values(effective_values)
    return {
        "valid": valid,
        "errors": errors,
        "env_preview": render_env_file(target_values, mask_secrets=True),
    }


def changed_pending_fields(updates: Mapping[str, Any]) -> list[str]:
    """Return changed fields that require manual runtime action."""

    state = _load_value_state()
    pending: list[str] = []
    for key, value in updates.items():
        field = FIELD_BY_KEY.get(key)
        if field is None or not (field.restart_required or field.session_sensitive):
            continue
        if _normalize_for_env(value) == str(state[key]["value"]):
            continue
        pending.append(key)
    return pending


def write_managed_env(updates: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and atomically write the admin-managed env file."""

    validation = validate_updates(updates)
    if not validation["valid"]:
        return validation | {"applied": False, "pending_fields": []}

    target_values = _target_values_with_updates(updates)
    pending_fields = changed_pending_fields(updates)
    path = managed_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(render_env_file(target_values), encoding="utf-8")
    os.replace(temp_path, path)
    return {
        "applied": True,
        "valid": True,
        "errors": [],
        "env_preview": render_env_file(target_values, mask_secrets=True),
        "path": str(path),
        "pending_fields": pending_fields,
    }


def _quote_env_value(value: str) -> str:
    if value == "":
        return ""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    if any(char.isspace() for char in value) or any(
        char in value for char in ('"', "#", "=", "$")
    ):
        return f'"{escaped}"'
    return value


def render_env_file(values: Mapping[str, str], *, mask_secrets: bool = False) -> str:
    """Render a complete grouped env file."""

    lines: list[str] = [
        "# Managed by Free Claude Code /admin.",
        "# Edit in the server UI when possible.",
        "",
    ]
    fields_by_section: dict[str, list[ConfigFieldSpec]] = {
        section.section_id: [] for section in SECTIONS
    }
    for field in FIELDS:
        fields_by_section.setdefault(field.section_id, []).append(field)

    for section in SECTIONS:
        lines.append(f"# {section.label}")
        for field in fields_by_section.get(section.section_id, []):
            value = values.get(field.key, field.default)
            if mask_secrets and field.secret and value:
                value = MASKED_SECRET
            lines.append(f"{field.key}={_quote_env_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def provider_config_status(
    state: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return provider configuration status without making network calls."""

    state = state or _load_value_state()
    statuses: list[dict[str, Any]] = []
    for provider_id, descriptor in PROVIDER_CATALOG.items():
        if descriptor.credential_env is None:
            base_url = ""
            if descriptor.base_url_attr is not None:
                base_url = _value_for_settings_attr(state, descriptor.base_url_attr)
            statuses.append(
                {
                    "provider_id": provider_id,
                    "kind": "local",
                    "status": "missing_url" if not base_url.strip() else "unknown",
                    "label": "Missing URL" if not base_url.strip() else "Not checked",
                    "base_url": base_url or descriptor.default_base_url or "",
                }
            )
            continue

        value = str(state.get(descriptor.credential_env, {}).get("value", ""))
        configured = bool(value.strip())
        statuses.append(
            {
                "provider_id": provider_id,
                "kind": "remote",
                "status": "configured" if configured else "missing_key",
                "label": "Configured" if configured else "Missing key",
                "credential_env": descriptor.credential_env,
            }
        )
    return statuses


def _value_for_settings_attr(
    state: Mapping[str, Mapping[str, Any]], settings_attr: str
) -> str:
    for field in FIELDS:
        if field.settings_attr == settings_attr:
            return str(state.get(field.key, {}).get("value", field.default))
    return ""


def env_keys() -> frozenset[str]:
    """Return env keys owned by the admin manifest."""

    return frozenset(field.key for field in FIELDS)


def fields_with_attrs() -> Iterable[ConfigFieldSpec]:
    """Yield fields that validate through Settings."""

    return (field for field in FIELDS if field.settings_attr is not None)
