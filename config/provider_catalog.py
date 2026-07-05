"""Neutral provider catalog: IDs, credentials, defaults, proxy and capability metadata.

Adapter factories live in :mod:`providers.runtime.factory`; this module stays free of
provider implementation imports (see contract tests).
"""

from dataclasses import dataclass
from typing import Literal

TransportType = Literal["openai_chat", "anthropic_messages"]

# Default upstream base URLs (also re-exported via :mod:`providers.defaults`)
NVIDIA_NIM_DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"
# Moonshot Kimi Anthropic-compatible Messages API (POST …/messages).
KIMI_DEFAULT_BASE = "https://api.moonshot.ai/anthropic/v1"
WAFER_DEFAULT_BASE = "https://pass.wafer.ai/v1"
MINIMAX_DEFAULT_BASE = "https://api.minimax.io/anthropic/v1"
# DeepSeek Chat Completions API; cache usage is reported on this endpoint.
DEEPSEEK_DEFAULT_BASE = "https://api.deepseek.com"
FIREWORKS_DEFAULT_BASE = "https://api.fireworks.ai/inference/v1"
# Cloudflare account-scoped AI REST root; provider appends /accounts/{id}/ai/v1.
CLOUDFLARE_AI_REST_ROOT = "https://api.cloudflare.com/client/v4"
OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api/v1"
MISTRAL_DEFAULT_BASE = "https://api.mistral.ai/v1"
# Codestral IDE/personal endpoint (distinct from La Plateforme ``api.mistral.ai`` keys).
CODESTRAL_DEFAULT_BASE = "https://codestral.mistral.ai/v1"
LMSTUDIO_DEFAULT_BASE = "http://localhost:1234/v1"
LLAMACPP_DEFAULT_BASE = "http://localhost:8080/v1"
OLLAMA_DEFAULT_BASE = "http://localhost:11434"
OPENCODE_DEFAULT_BASE = "https://opencode.ai/zen/v1"
OPENCODE_GO_DEFAULT_BASE = "https://opencode.ai/zen/go/v1"
VERCEL_AI_GATEWAY_DEFAULT_BASE = "https://ai-gateway.vercel.sh/v1"
HUGGINGFACE_DEFAULT_BASE = "https://router.huggingface.co/v1"
COHERE_DEFAULT_BASE = "https://api.cohere.ai/compatibility/v1"
# Z.ai Anthropic-compatible Messages API (not OpenAI Coding Plan chat completions).
ZAI_DEFAULT_BASE = "https://api.z.ai/api/anthropic/v1"
# Google AI Studio Gemini API OpenAI-compat layer (not Vertex AI).
GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_DEFAULT_BASE = "https://api.groq.com/openai/v1"
CEREBRAS_DEFAULT_BASE = "https://api.cerebras.ai/v1"


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    """Metadata for building :class:`~providers.base.ProviderConfig` and factory wiring."""

    provider_id: str
    display_name: str
    transport_type: TransportType
    capabilities: tuple[str, ...]
    credential_env: str | None = None
    credential_url: str | None = None
    credential_attr: str | None = None
    static_credential: str | None = None
    default_base_url: str | None = None
    base_url_attr: str | None = None
    proxy_attr: str | None = None


PROVIDER_CATALOG: dict[str, ProviderDescriptor] = {
    "nvidia_nim": ProviderDescriptor(
        provider_id="nvidia_nim",
        display_name="NVIDIA NIM",
        transport_type="openai_chat",
        credential_env="NVIDIA_NIM_API_KEY",
        credential_url="https://build.nvidia.com/settings/api-keys",
        credential_attr="nvidia_nim_api_key",
        default_base_url=NVIDIA_NIM_DEFAULT_BASE,
        proxy_attr="nvidia_nim_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "open_router": ProviderDescriptor(
        provider_id="open_router",
        display_name="OpenRouter",
        transport_type="anthropic_messages",
        credential_env="OPENROUTER_API_KEY",
        credential_url="https://openrouter.ai/keys",
        credential_attr="open_router_api_key",
        default_base_url=OPENROUTER_DEFAULT_BASE,
        proxy_attr="open_router_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "native_anthropic"),
    ),
    "gemini": ProviderDescriptor(
        provider_id="gemini",
        display_name="Gemini",
        transport_type="openai_chat",
        credential_env="GEMINI_API_KEY",
        credential_url="https://aistudio.google.com/apikey",
        credential_attr="gemini_api_key",
        default_base_url=GEMINI_DEFAULT_BASE,
        proxy_attr="gemini_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "deepseek": ProviderDescriptor(
        provider_id="deepseek",
        display_name="DeepSeek",
        transport_type="openai_chat",
        credential_env="DEEPSEEK_API_KEY",
        credential_url="https://platform.deepseek.com/api_keys",
        credential_attr="deepseek_api_key",
        default_base_url=DEEPSEEK_DEFAULT_BASE,
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "mistral": ProviderDescriptor(
        provider_id="mistral",
        display_name="Mistral",
        transport_type="openai_chat",
        credential_env="MISTRAL_API_KEY",
        credential_url="https://console.mistral.ai/",
        credential_attr="mistral_api_key",
        default_base_url=MISTRAL_DEFAULT_BASE,
        proxy_attr="mistral_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "mistral_codestral": ProviderDescriptor(
        provider_id="mistral_codestral",
        display_name="Mistral Codestral",
        transport_type="openai_chat",
        credential_env="CODESTRAL_API_KEY",
        credential_url="https://console.mistral.ai/",
        credential_attr="codestral_api_key",
        default_base_url=CODESTRAL_DEFAULT_BASE,
        proxy_attr="codestral_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "opencode": ProviderDescriptor(
        provider_id="opencode",
        display_name="OpenCode Zen",
        transport_type="openai_chat",
        credential_env="OPENCODE_API_KEY",
        credential_url="https://opencode.ai/auth",
        credential_attr="opencode_api_key",
        default_base_url=OPENCODE_DEFAULT_BASE,
        proxy_attr="opencode_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "opencode_go": ProviderDescriptor(
        provider_id="opencode_go",
        display_name="OpenCode Go",
        transport_type="openai_chat",
        credential_env="OPENCODE_API_KEY",
        credential_url="https://opencode.ai/auth",
        credential_attr="opencode_api_key",
        default_base_url=OPENCODE_GO_DEFAULT_BASE,
        proxy_attr="opencode_go_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "vercel": ProviderDescriptor(
        provider_id="vercel",
        display_name="Vercel AI Gateway",
        transport_type="openai_chat",
        credential_env="AI_GATEWAY_API_KEY",
        credential_url="https://vercel.com/docs/ai-gateway",
        credential_attr="vercel_ai_gateway_api_key",
        default_base_url=VERCEL_AI_GATEWAY_DEFAULT_BASE,
        proxy_attr="vercel_ai_gateway_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "huggingface": ProviderDescriptor(
        provider_id="huggingface",
        display_name="Hugging Face",
        transport_type="openai_chat",
        credential_env="HUGGINGFACE_API_KEY",
        credential_url="https://huggingface.co/settings/tokens",
        credential_attr="huggingface_api_key",
        default_base_url=HUGGINGFACE_DEFAULT_BASE,
        proxy_attr="huggingface_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "cohere": ProviderDescriptor(
        provider_id="cohere",
        display_name="Cohere",
        transport_type="openai_chat",
        credential_env="COHERE_API_KEY",
        credential_url="https://dashboard.cohere.com/api-keys",
        credential_attr="cohere_api_key",
        default_base_url=COHERE_DEFAULT_BASE,
        proxy_attr="cohere_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "wafer": ProviderDescriptor(
        provider_id="wafer",
        display_name="Wafer",
        transport_type="anthropic_messages",
        credential_env="WAFER_API_KEY",
        credential_url="https://www.wafer.ai/pass",
        credential_attr="wafer_api_key",
        default_base_url=WAFER_DEFAULT_BASE,
        proxy_attr="wafer_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "native_anthropic"),
    ),
    "kimi": ProviderDescriptor(
        provider_id="kimi",
        display_name="Kimi",
        transport_type="anthropic_messages",
        credential_env="KIMI_API_KEY",
        credential_url="https://platform.moonshot.cn/console/api-keys",
        credential_attr="kimi_api_key",
        default_base_url=KIMI_DEFAULT_BASE,
        proxy_attr="kimi_proxy",
        capabilities=(
            "chat",
            "streaming",
            "tools",
            "thinking",
            "native_anthropic",
        ),
    ),
    "minimax": ProviderDescriptor(
        provider_id="minimax",
        display_name="MiniMax",
        transport_type="anthropic_messages",
        credential_env="MINIMAX_API_KEY",
        credential_url="https://platform.minimax.io/user-center/basic-information/interface-key",
        credential_attr="minimax_api_key",
        default_base_url=MINIMAX_DEFAULT_BASE,
        proxy_attr="minimax_proxy",
        capabilities=(
            "chat",
            "streaming",
            "tools",
            "thinking",
            "native_anthropic",
            "rate_limit",
        ),
    ),
    "cerebras": ProviderDescriptor(
        provider_id="cerebras",
        display_name="Cerebras",
        transport_type="openai_chat",
        credential_env="CEREBRAS_API_KEY",
        credential_url="https://cloud.cerebras.ai",
        credential_attr="cerebras_api_key",
        default_base_url=CEREBRAS_DEFAULT_BASE,
        proxy_attr="cerebras_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "groq": ProviderDescriptor(
        provider_id="groq",
        display_name="Groq",
        transport_type="openai_chat",
        credential_env="GROQ_API_KEY",
        credential_url="https://console.groq.com/keys",
        credential_attr="groq_api_key",
        default_base_url=GROQ_DEFAULT_BASE,
        proxy_attr="groq_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "rate_limit"),
    ),
    "fireworks": ProviderDescriptor(
        provider_id="fireworks",
        display_name="Fireworks",
        transport_type="anthropic_messages",
        credential_env="FIREWORKS_API_KEY",
        credential_url="https://fireworks.ai/account/api-keys",
        credential_attr="fireworks_api_key",
        default_base_url=FIREWORKS_DEFAULT_BASE,
        proxy_attr="fireworks_proxy",
        capabilities=(
            "chat",
            "streaming",
            "tools",
            "thinking",
            "native_anthropic",
            "rate_limit",
        ),
    ),
    "cloudflare": ProviderDescriptor(
        provider_id="cloudflare",
        display_name="Cloudflare",
        transport_type="openai_chat",
        credential_env="CLOUDFLARE_API_TOKEN",
        credential_url="https://dash.cloudflare.com/profile/api-tokens",
        credential_attr="cloudflare_api_token",
        default_base_url=CLOUDFLARE_AI_REST_ROOT,
        proxy_attr="cloudflare_proxy",
        capabilities=(
            "chat",
            "streaming",
            "tools",
            "thinking",
            "rate_limit",
        ),
    ),
    "zai": ProviderDescriptor(
        provider_id="zai",
        display_name="Z.ai",
        transport_type="anthropic_messages",
        credential_env="ZAI_API_KEY",
        credential_attr="zai_api_key",
        default_base_url=ZAI_DEFAULT_BASE,
        proxy_attr="zai_proxy",
        capabilities=(
            "chat",
            "streaming",
            "tools",
            "thinking",
            "native_anthropic",
            "rate_limit",
        ),
    ),
    "lmstudio": ProviderDescriptor(
        provider_id="lmstudio",
        display_name="LM Studio",
        transport_type="openai_chat",
        static_credential="lm-studio",
        default_base_url=LMSTUDIO_DEFAULT_BASE,
        base_url_attr="lm_studio_base_url",
        proxy_attr="lmstudio_proxy",
        capabilities=("chat", "streaming", "tools", "local"),
    ),
    "llamacpp": ProviderDescriptor(
        provider_id="llamacpp",
        display_name="llama.cpp",
        transport_type="anthropic_messages",
        static_credential="llamacpp",
        default_base_url=LLAMACPP_DEFAULT_BASE,
        base_url_attr="llamacpp_base_url",
        proxy_attr="llamacpp_proxy",
        capabilities=("chat", "streaming", "tools", "native_anthropic", "local"),
    ),
    "ollama": ProviderDescriptor(
        provider_id="ollama",
        display_name="Ollama",
        transport_type="anthropic_messages",
        static_credential="ollama",
        default_base_url=OLLAMA_DEFAULT_BASE,
        base_url_attr="ollama_base_url",
        capabilities=(
            "chat",
            "streaming",
            "tools",
            "thinking",
            "native_anthropic",
            "local",
        ),
    ),
}

# Key order:
# NVIDIA NIM first (README default), DeepSeek fourth, OpenCode gateways adjacent,
# Vercel / Hugging Face / Cohere follow gateway-style remotes, then native Anthropic
# remotes and locals per project plan (github.com/cheahjs/free-llm-api-resources
# Free Providers TOC as rough guide beyond fixed slots).
# ``SUPPORTED_PROVIDER_IDS`` inherits this insertion order for UI and error-message listing.
SUPPORTED_PROVIDER_IDS: tuple[str, ...] = tuple(PROVIDER_CATALOG.keys())

if len(set(SUPPORTED_PROVIDER_IDS)) != len(SUPPORTED_PROVIDER_IDS):
    raise AssertionError("Duplicate provider ids in PROVIDER_CATALOG key order")
