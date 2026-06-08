"""Neutral provider catalog: IDs, credentials, defaults, proxy and capability metadata.

Adapter factories live in :mod:`providers.registry`; this module stays free of
provider implementation imports (see contract tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TransportType = Literal["openai_chat", "anthropic_messages"]

# Default upstream base URLs (also re-exported via :mod:`providers.defaults`)
NVIDIA_NIM_DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"
KIMI_DEFAULT_BASE = "https://api.moonshot.ai/v1"
WAFER_DEFAULT_BASE = "https://pass.wafer.ai/v1"
# DeepSeek Anthropic-compatible Messages API (not OpenAI ``/v1`` chat completions).
DEEPSEEK_ANTHROPIC_DEFAULT_BASE = "https://api.deepseek.com/anthropic"
# Historical export name: DeepSeek upstream is the native Anthropic path above.
DEEPSEEK_DEFAULT_BASE = DEEPSEEK_ANTHROPIC_DEFAULT_BASE
OPENROUTER_DEFAULT_BASE = "https://openrouter.ai/api/v1"
LMSTUDIO_DEFAULT_BASE = "http://localhost:1234/v1"
LLAMACPP_DEFAULT_BASE = "http://localhost:8080/v1"
OLLAMA_DEFAULT_BASE = "http://localhost:11434"


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    """Metadata for building :class:`~providers.base.ProviderConfig` and factory wiring."""

    provider_id: str
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
        transport_type="anthropic_messages",
        credential_env="OPENROUTER_API_KEY",
        credential_url="https://openrouter.ai/keys",
        credential_attr="open_router_api_key",
        default_base_url=OPENROUTER_DEFAULT_BASE,
        proxy_attr="open_router_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "native_anthropic"),
    ),
    "deepseek": ProviderDescriptor(
        provider_id="deepseek",
        transport_type="anthropic_messages",
        credential_env="DEEPSEEK_API_KEY",
        credential_url="https://platform.deepseek.com/api_keys",
        credential_attr="deepseek_api_key",
        default_base_url=DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
        capabilities=("chat", "streaming", "tools", "thinking", "native_anthropic"),
    ),
    "lmstudio": ProviderDescriptor(
        provider_id="lmstudio",
        transport_type="openai_chat",
        static_credential="lm-studio",
        default_base_url=LMSTUDIO_DEFAULT_BASE,
        base_url_attr="lm_studio_base_url",
        proxy_attr="lmstudio_proxy",
        capabilities=("chat", "streaming", "tools", "local"),
    ),
    "llamacpp": ProviderDescriptor(
        provider_id="llamacpp",
        transport_type="openai_chat",
        static_credential="llamacpp",
        default_base_url=LLAMACPP_DEFAULT_BASE,
        base_url_attr="llamacpp_base_url",
        proxy_attr="llamacpp_proxy",
        capabilities=("chat", "streaming", "tools", "local"),
    ),
    "ollama": ProviderDescriptor(
        provider_id="ollama",
        transport_type="openai_chat",
        static_credential="ollama",
        default_base_url=OLLAMA_DEFAULT_BASE,
        base_url_attr="ollama_base_url",
        capabilities=(
            "chat",
            "streaming",
            "tools",
            "thinking",
            "local",
        ),
    ),
    "kimi": ProviderDescriptor(
        provider_id="kimi",
        transport_type="openai_chat",
        credential_env="KIMI_API_KEY",
        credential_url="https://platform.moonshot.cn/console/api-keys",
        credential_attr="kimi_api_key",
        default_base_url=KIMI_DEFAULT_BASE,
        proxy_attr="kimi_proxy",
        capabilities=("chat", "streaming", "tools"),
    ),
    "wafer": ProviderDescriptor(
        provider_id="wafer",
        transport_type="anthropic_messages",
        credential_env="WAFER_API_KEY",
        credential_url="https://www.wafer.ai/pass",
        credential_attr="wafer_api_key",
        default_base_url=WAFER_DEFAULT_BASE,
        proxy_attr="wafer_proxy",
        capabilities=("chat", "streaming", "tools", "thinking", "native_anthropic"),
    ),
}

# Order matches docs / historical error text; must match PROVIDER_CATALOG keys.
SUPPORTED_PROVIDER_IDS: tuple[str, ...] = tuple(PROVIDER_CATALOG.keys())

if len(set(SUPPORTED_PROVIDER_IDS)) != len(SUPPORTED_PROVIDER_IDS):
    raise AssertionError("Duplicate provider ids in PROVIDER_CATALOG key order")
