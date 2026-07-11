"""Neutral provider catalog: IDs, credentials, defaults, proxy and capability metadata.

Adapter factories live in :mod:`providers.runtime.factory`; this module stays free of
provider implementation imports (see contract tests).
"""

from dataclasses import dataclass

# Default upstream base URLs (also re-exported via :mod:`providers.defaults`)
NVIDIA_NIM_DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"
# Moonshot Kimi OpenAI-compatible Chat Completions API.
KIMI_DEFAULT_BASE = "https://api.moonshot.ai/v1"
WAFER_DEFAULT_BASE = "https://pass.wafer.ai/v1"
MINIMAX_DEFAULT_BASE = "https://api.minimax.io/v1"
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
GITHUB_MODELS_DEFAULT_BASE = "https://models.github.ai/inference"
# Z.ai GLM Coding Plan OpenAI-compatible Chat Completions API.
ZAI_DEFAULT_BASE = "https://api.z.ai/api/coding/paas/v4"
# Google AI Studio Gemini API OpenAI-compat layer (not Vertex AI).
GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_DEFAULT_BASE = "https://api.groq.com/openai/v1"
CEREBRAS_DEFAULT_BASE = "https://api.cerebras.ai/v1"
SAMBANOVA_DEFAULT_BASE = "https://api.sambanova.ai/v1"


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Static provider semantics used by application policy."""

    local: bool = False
    server_tool_passthrough: bool = False


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    """Metadata for building :class:`~providers.base.ProviderConfig` and factory wiring."""

    provider_id: str
    display_name: str
    capabilities: ProviderCapabilities = ProviderCapabilities()
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
        credential_env="NVIDIA_NIM_API_KEY",
        credential_url="https://build.nvidia.com/settings/api-keys",
        credential_attr="nvidia_nim_api_key",
        default_base_url=NVIDIA_NIM_DEFAULT_BASE,
        proxy_attr="nvidia_nim_proxy",
    ),
    "open_router": ProviderDescriptor(
        provider_id="open_router",
        display_name="OpenRouter",
        credential_env="OPENROUTER_API_KEY",
        credential_url="https://openrouter.ai/keys",
        credential_attr="open_router_api_key",
        default_base_url=OPENROUTER_DEFAULT_BASE,
        proxy_attr="open_router_proxy",
    ),
    "gemini": ProviderDescriptor(
        provider_id="gemini",
        display_name="Gemini",
        credential_env="GEMINI_API_KEY",
        credential_url="https://aistudio.google.com/apikey",
        credential_attr="gemini_api_key",
        default_base_url=GEMINI_DEFAULT_BASE,
        proxy_attr="gemini_proxy",
    ),
    "deepseek": ProviderDescriptor(
        provider_id="deepseek",
        display_name="DeepSeek",
        credential_env="DEEPSEEK_API_KEY",
        credential_url="https://platform.deepseek.com/api_keys",
        credential_attr="deepseek_api_key",
        default_base_url=DEEPSEEK_DEFAULT_BASE,
    ),
    "mistral": ProviderDescriptor(
        provider_id="mistral",
        display_name="Mistral",
        credential_env="MISTRAL_API_KEY",
        credential_url="https://console.mistral.ai/",
        credential_attr="mistral_api_key",
        default_base_url=MISTRAL_DEFAULT_BASE,
        proxy_attr="mistral_proxy",
    ),
    "mistral_codestral": ProviderDescriptor(
        provider_id="mistral_codestral",
        display_name="Mistral Codestral",
        credential_env="CODESTRAL_API_KEY",
        credential_url="https://console.mistral.ai/",
        credential_attr="codestral_api_key",
        default_base_url=CODESTRAL_DEFAULT_BASE,
        proxy_attr="codestral_proxy",
    ),
    "opencode": ProviderDescriptor(
        provider_id="opencode",
        display_name="OpenCode Zen",
        credential_env="OPENCODE_API_KEY",
        credential_url="https://opencode.ai/auth",
        credential_attr="opencode_api_key",
        default_base_url=OPENCODE_DEFAULT_BASE,
        proxy_attr="opencode_proxy",
    ),
    "opencode_go": ProviderDescriptor(
        provider_id="opencode_go",
        display_name="OpenCode Go",
        credential_env="OPENCODE_API_KEY",
        credential_url="https://opencode.ai/auth",
        credential_attr="opencode_api_key",
        default_base_url=OPENCODE_GO_DEFAULT_BASE,
        proxy_attr="opencode_go_proxy",
    ),
    "vercel": ProviderDescriptor(
        provider_id="vercel",
        display_name="Vercel AI Gateway",
        credential_env="AI_GATEWAY_API_KEY",
        credential_url="https://vercel.com/docs/ai-gateway",
        credential_attr="vercel_ai_gateway_api_key",
        default_base_url=VERCEL_AI_GATEWAY_DEFAULT_BASE,
        proxy_attr="vercel_ai_gateway_proxy",
    ),
    "huggingface": ProviderDescriptor(
        provider_id="huggingface",
        display_name="Hugging Face",
        credential_env="HUGGINGFACE_API_KEY",
        credential_url="https://huggingface.co/settings/tokens",
        credential_attr="huggingface_api_key",
        default_base_url=HUGGINGFACE_DEFAULT_BASE,
        proxy_attr="huggingface_proxy",
    ),
    "cohere": ProviderDescriptor(
        provider_id="cohere",
        display_name="Cohere",
        credential_env="COHERE_API_KEY",
        credential_url="https://dashboard.cohere.com/api-keys",
        credential_attr="cohere_api_key",
        default_base_url=COHERE_DEFAULT_BASE,
        proxy_attr="cohere_proxy",
    ),
    "github_models": ProviderDescriptor(
        provider_id="github_models",
        display_name="GitHub Models",
        credential_env="GITHUB_MODELS_TOKEN",
        credential_url="https://github.com/settings/tokens",
        credential_attr="github_models_token",
        default_base_url=GITHUB_MODELS_DEFAULT_BASE,
        proxy_attr="github_models_proxy",
    ),
    "wafer": ProviderDescriptor(
        provider_id="wafer",
        display_name="Wafer",
        credential_env="WAFER_API_KEY",
        credential_url="https://www.wafer.ai/pass",
        credential_attr="wafer_api_key",
        default_base_url=WAFER_DEFAULT_BASE,
        proxy_attr="wafer_proxy",
    ),
    "kimi": ProviderDescriptor(
        provider_id="kimi",
        display_name="Kimi",
        credential_env="KIMI_API_KEY",
        credential_url="https://platform.moonshot.cn/console/api-keys",
        credential_attr="kimi_api_key",
        default_base_url=KIMI_DEFAULT_BASE,
        proxy_attr="kimi_proxy",
    ),
    "minimax": ProviderDescriptor(
        provider_id="minimax",
        display_name="MiniMax",
        credential_env="MINIMAX_API_KEY",
        credential_url="https://platform.minimax.io/user-center/basic-information/interface-key",
        credential_attr="minimax_api_key",
        default_base_url=MINIMAX_DEFAULT_BASE,
        proxy_attr="minimax_proxy",
    ),
    "cerebras": ProviderDescriptor(
        provider_id="cerebras",
        display_name="Cerebras",
        credential_env="CEREBRAS_API_KEY",
        credential_url="https://cloud.cerebras.ai",
        credential_attr="cerebras_api_key",
        default_base_url=CEREBRAS_DEFAULT_BASE,
        proxy_attr="cerebras_proxy",
    ),
    "groq": ProviderDescriptor(
        provider_id="groq",
        display_name="Groq",
        credential_env="GROQ_API_KEY",
        credential_url="https://console.groq.com/keys",
        credential_attr="groq_api_key",
        default_base_url=GROQ_DEFAULT_BASE,
        proxy_attr="groq_proxy",
    ),
    "sambanova": ProviderDescriptor(
        provider_id="sambanova",
        display_name="SambaNova",
        credential_env="SAMBANOVA_API_KEY",
        credential_url="https://cloud.sambanova.ai/apis",
        credential_attr="sambanova_api_key",
        default_base_url=SAMBANOVA_DEFAULT_BASE,
        proxy_attr="sambanova_proxy",
    ),
    "fireworks": ProviderDescriptor(
        provider_id="fireworks",
        display_name="Fireworks",
        credential_env="FIREWORKS_API_KEY",
        credential_url="https://fireworks.ai/account/api-keys",
        credential_attr="fireworks_api_key",
        default_base_url=FIREWORKS_DEFAULT_BASE,
        proxy_attr="fireworks_proxy",
    ),
    "cloudflare": ProviderDescriptor(
        provider_id="cloudflare",
        display_name="Cloudflare",
        credential_env="CLOUDFLARE_API_TOKEN",
        credential_url="https://dash.cloudflare.com/profile/api-tokens",
        credential_attr="cloudflare_api_token",
        default_base_url=CLOUDFLARE_AI_REST_ROOT,
        proxy_attr="cloudflare_proxy",
    ),
    "zai": ProviderDescriptor(
        provider_id="zai",
        display_name="Z.ai",
        credential_env="ZAI_API_KEY",
        credential_attr="zai_api_key",
        default_base_url=ZAI_DEFAULT_BASE,
        proxy_attr="zai_proxy",
    ),
    "lmstudio": ProviderDescriptor(
        provider_id="lmstudio",
        display_name="LM Studio",
        static_credential="lm-studio",
        default_base_url=LMSTUDIO_DEFAULT_BASE,
        base_url_attr="lm_studio_base_url",
        proxy_attr="lmstudio_proxy",
        capabilities=ProviderCapabilities(local=True),
    ),
    "llamacpp": ProviderDescriptor(
        provider_id="llamacpp",
        display_name="llama.cpp",
        static_credential="llamacpp",
        default_base_url=LLAMACPP_DEFAULT_BASE,
        base_url_attr="llamacpp_base_url",
        proxy_attr="llamacpp_proxy",
        capabilities=ProviderCapabilities(
            local=True,
            server_tool_passthrough=True,
        ),
    ),
    "ollama": ProviderDescriptor(
        provider_id="ollama",
        display_name="Ollama",
        static_credential="ollama",
        default_base_url=OLLAMA_DEFAULT_BASE,
        base_url_attr="ollama_base_url",
        capabilities=ProviderCapabilities(
            local=True,
            server_tool_passthrough=True,
        ),
    ),
}

# Key order:
# NVIDIA NIM first (README default), DeepSeek fourth, OpenCode gateways adjacent,
# Vercel / Hugging Face / Cohere / GitHub Models follow gateway-style remotes,
# then cloud gateways and local providers per project plan
# (github.com/cheahjs/free-llm-api-resources Free Providers TOC as rough guide
# beyond fixed slots).
# ``SUPPORTED_PROVIDER_IDS`` inherits this insertion order for UI and error-message listing.
SUPPORTED_PROVIDER_IDS: tuple[str, ...] = tuple(PROVIDER_CATALOG.keys())

if len(set(SUPPORTED_PROVIDER_IDS)) != len(SUPPORTED_PROVIDER_IDS):
    raise AssertionError("Duplicate provider ids in PROVIDER_CATALOG key order")
