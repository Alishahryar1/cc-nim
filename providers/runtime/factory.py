"""Provider factory wiring and lazy adapter construction."""

from collections.abc import Callable

from config.provider_catalog import (
    PROVIDER_CATALOG,
    SUPPORTED_PROVIDER_IDS,
)
from config.settings import Settings
from providers.base import BaseProvider, ProviderConfig
from providers.exceptions import UnknownProviderTypeError

from .config import build_provider_config

ProviderFactory = Callable[[ProviderConfig, Settings], BaseProvider]


def _create_nvidia_nim(config: ProviderConfig, settings: Settings) -> BaseProvider:
    from providers.nvidia_nim import NvidiaNimProvider

    return NvidiaNimProvider(config, nim_settings=settings.nim)


def _create_open_router(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.open_router import OpenRouterProvider

    return OpenRouterProvider(config)


def _create_mistral(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.mistral import MistralProvider

    return MistralProvider(config)


def _create_mistral_codestral(
    config: ProviderConfig, _settings: Settings
) -> BaseProvider:
    from providers.codestral import CodestralProvider

    return CodestralProvider(config)


def _create_deepseek(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.deepseek import DeepSeekProvider

    return DeepSeekProvider(config)


def _create_lmstudio(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.lmstudio import LMStudioProvider

    return LMStudioProvider(config)


def _create_llamacpp(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.llamacpp import LlamaCppProvider

    return LlamaCppProvider(config)


def _create_ollama(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.ollama import OllamaProvider

    return OllamaProvider(config)


def _create_kimi(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.kimi import KimiProvider

    return KimiProvider(config)


def _create_wafer(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.wafer import WaferProvider

    return WaferProvider(config)


def _create_minimax(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.minimax import MiniMaxProvider

    return MiniMaxProvider(config)


def _create_opencode(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.opencode import OpenCodeProvider

    return OpenCodeProvider(config)


def _create_opencode_go(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.opencode import OpenCodeProvider

    return OpenCodeProvider(config, provider_name="OPENCODE_GO")


def _create_vercel(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.vercel import VercelProvider

    return VercelProvider(config)


def _create_huggingface(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.huggingface import HuggingFaceProvider

    return HuggingFaceProvider(config)


def _create_cohere(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.cohere import CohereProvider

    return CohereProvider(config)


def _create_github_models(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.github_models import GitHubModelsProvider

    return GitHubModelsProvider(config)


def _create_zai(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.zai import ZaiProvider

    return ZaiProvider(config)


def _create_fireworks(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.fireworks import FireworksProvider

    return FireworksProvider(config)


def _create_cloudflare(config: ProviderConfig, settings: Settings) -> BaseProvider:
    from providers.cloudflare import CloudflareProvider

    return CloudflareProvider(config, account_id=settings.cloudflare_account_id)


def _create_gemini(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.gemini import GeminiProvider

    return GeminiProvider(config)


def _create_groq(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.groq import GroqProvider

    return GroqProvider(config)


def _create_sambanova(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.sambanova import SambaNovaProvider

    return SambaNovaProvider(config)


def _create_cerebras(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.cerebras import CerebrasProvider

    return CerebrasProvider(config)


PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "nvidia_nim": _create_nvidia_nim,
    "open_router": _create_open_router,
    "gemini": _create_gemini,
    "deepseek": _create_deepseek,
    "mistral": _create_mistral,
    "mistral_codestral": _create_mistral_codestral,
    "opencode": _create_opencode,
    "opencode_go": _create_opencode_go,
    "vercel": _create_vercel,
    "huggingface": _create_huggingface,
    "cohere": _create_cohere,
    "github_models": _create_github_models,
    "wafer": _create_wafer,
    "kimi": _create_kimi,
    "minimax": _create_minimax,
    "cerebras": _create_cerebras,
    "groq": _create_groq,
    "sambanova": _create_sambanova,
    "fireworks": _create_fireworks,
    "cloudflare": _create_cloudflare,
    "zai": _create_zai,
    "lmstudio": _create_lmstudio,
    "llamacpp": _create_llamacpp,
    "ollama": _create_ollama,
}

if set(PROVIDER_CATALOG) != set(SUPPORTED_PROVIDER_IDS) or set(
    PROVIDER_FACTORIES
) != set(SUPPORTED_PROVIDER_IDS):
    raise AssertionError(
        "PROVIDER_CATALOG, PROVIDER_FACTORIES, and SUPPORTED_PROVIDER_IDS are out of sync: "
        f"catalog={set(PROVIDER_CATALOG)!r} factories={set(PROVIDER_FACTORIES)!r} "
        f"ids={set(SUPPORTED_PROVIDER_IDS)!r}"
    )


def create_provider(provider_id: str, settings: Settings) -> BaseProvider:
    """Create a provider instance for a supported provider id."""
    descriptor = PROVIDER_CATALOG.get(provider_id)
    if descriptor is None:
        supported = "', '".join(PROVIDER_CATALOG)
        raise UnknownProviderTypeError(
            f"Unknown provider_type: '{provider_id}'. Supported: '{supported}'"
        )

    factory = PROVIDER_FACTORIES.get(provider_id)
    if factory is None:
        raise AssertionError(f"Unhandled provider descriptor: {provider_id}")
    return factory(build_provider_config(descriptor, settings), settings)
