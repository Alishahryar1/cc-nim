"""Admin configuration manifest."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from free_claude_code.config.settings import Settings

from .provider_manifest import provider_field_specs

FieldType = Literal[
    "text",
    "secret",
    "number",
    "boolean",
    "tri_boolean",
    "select",
    "textarea",
]


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


_NON_PROVIDER_FIELDS: tuple[ConfigFieldSpec, ...] = (
    ConfigFieldSpec(
        "MODEL",
        "Default Model",
        "models",
        settings_attr="model",
        default="nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
        description="Fallback provider/model route for all Claude model names.",
    ),
    ConfigFieldSpec(
        "MODEL_FABLE",
        "Fable Override",
        "models",
        settings_attr="model_fable",
        description="Optional provider/model route for Fable requests.",
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
        "ENABLE_FABLE_THINKING",
        "Fable Thinking",
        "thinking",
        "tri_boolean",
        settings_attr="enable_fable_thinking",
        description="Blank inherits Enable Thinking.",
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
        restart_required=True,
        description="Bearer token protecting Claude/API access. It is not admin-page login.",
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
        "FCC_OPEN_BROWSER",
        "Open Admin on Startup",
        "runtime",
        "boolean",
        settings_attr="open_admin_browser",
        default="true",
        description="Open the Admin UI after the next fcc-server launch becomes healthy.",
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
        "TELEGRAM_PROXY_URL",
        "Telegram Proxy URL",
        "messaging",
        "secret",
        settings_attr="telegram_proxy_url",
        secret=True,
        session_sensitive=True,
        description="Optional Telegram-only proxy, e.g. socks5://127.0.0.1:1080.",
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
        "Max Tracked Messages Per Chat",
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
        restart_required=True,
    ),
    ConfigFieldSpec(
        "DEBUG_SUBAGENT_STACK",
        "Debug Subagent Stack",
        "diagnostics",
        "boolean",
        settings_attr="debug_subagent_stack",
        default="false",
        advanced=True,
        restart_required=True,
    ),
    ConfigFieldSpec(
        "LOG_RAW_API_PAYLOADS",
        "Log Raw API Payloads",
        "diagnostics",
        "boolean",
        settings_attr="log_raw_api_payloads",
        default="false",
        advanced=True,
        restart_required=True,
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
        restart_required=True,
    ),
    ConfigFieldSpec(
        "LOG_RAW_MESSAGING_CONTENT",
        "Log Raw Messaging Content",
        "diagnostics",
        "boolean",
        settings_attr="log_raw_messaging_content",
        default="false",
        advanced=True,
        restart_required=True,
    ),
    ConfigFieldSpec(
        "LOG_RAW_CLI_DIAGNOSTICS",
        "Log Raw CLI Diagnostics",
        "diagnostics",
        "boolean",
        settings_attr="log_raw_cli_diagnostics",
        default="false",
        advanced=True,
        restart_required=True,
    ),
    ConfigFieldSpec(
        "LOG_MESSAGING_ERROR_DETAILS",
        "Log Messaging Error Details",
        "diagnostics",
        "boolean",
        settings_attr="log_messaging_error_details",
        default="false",
        advanced=True,
        restart_required=True,
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
        "FCC_SMOKE_MODEL_OLLAMA_CLOUD",
        "Smoke Ollama Cloud Model",
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
        "FCC_SMOKE_MODEL_MINIMAX",
        "Smoke MiniMax Model",
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
        "FCC_SMOKE_MODEL_VERCEL",
        "Smoke Vercel AI Gateway Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_HUGGINGFACE",
        "Smoke Hugging Face Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_COHERE",
        "Smoke Cohere Model",
        "smoke",
        advanced=True,
    ),
    ConfigFieldSpec(
        "FCC_SMOKE_MODEL_GITHUB_MODELS",
        "Smoke GitHub Models Model",
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
        "FCC_SMOKE_MODEL_CLOUDFLARE",
        "Smoke Cloudflare Model",
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
        "FCC_SMOKE_MODEL_SAMBANOVA",
        "Smoke SambaNova Model",
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


FIELDS: tuple[ConfigFieldSpec, ...] = (
    *(ConfigFieldSpec(**spec) for spec in provider_field_specs()),
    *_NON_PROVIDER_FIELDS,
)
FIELD_BY_KEY = {field.key: field for field in FIELDS}


def field_input_key(field: ConfigFieldSpec) -> str | None:
    """Return the Settings input key used for a manifest field."""

    if field.settings_attr is None:
        return None
    model_field = Settings.model_fields[field.settings_attr]
    alias = model_field.validation_alias
    if alias is None:
        return field.settings_attr
    return str(alias)


def env_keys() -> frozenset[str]:
    """Return env keys owned by the admin manifest."""

    return frozenset(field.key for field in FIELDS)


def fields_with_attrs() -> Iterable[ConfigFieldSpec]:
    """Yield fields that validate through Settings."""

    return (field for field in FIELDS if field.settings_attr is not None)
