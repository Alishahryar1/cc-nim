"""Provider factory wiring and lazy adapter construction."""

from collections.abc import Callable

from free_claude_code.application.errors import UnknownProviderError
from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    SUPPORTED_PROVIDER_IDS,
)
from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.rate_limit import ProviderRateLimiter

from .config import build_provider_config

ProviderFactory = Callable[
    [ProviderConfig, Settings, ProviderRateLimiter], BaseProvider
]


def _create_nvidia_nim(
    config: ProviderConfig,
    settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.nvidia_nim import NvidiaNimProvider

    return NvidiaNimProvider(
        config,
        nim_settings=settings.nim,
        rate_limiter=rate_limiter,
    )


def _create_open_router(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.open_router import OpenRouterProvider

    return OpenRouterProvider(config, rate_limiter=rate_limiter)


def _create_mistral(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.mistral import MistralProvider

    return MistralProvider(config, rate_limiter=rate_limiter)


def _create_mistral_codestral(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.codestral import CodestralProvider

    return CodestralProvider(config, rate_limiter=rate_limiter)


def _create_deepseek(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.deepseek import DeepSeekProvider

    return DeepSeekProvider(config, rate_limiter=rate_limiter)


def _create_lmstudio(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.lmstudio import LMStudioProvider

    return LMStudioProvider(config, rate_limiter=rate_limiter)


def _create_llamacpp(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.llamacpp import LlamaCppProvider

    return LlamaCppProvider(config, rate_limiter=rate_limiter)


def _create_ollama(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.ollama import OllamaProvider

    return OllamaProvider(config, rate_limiter=rate_limiter)


def _create_kimi(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.kimi import KimiProvider

    return KimiProvider(config, rate_limiter=rate_limiter)


def _create_wafer(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.wafer import WaferProvider

    return WaferProvider(config, rate_limiter=rate_limiter)


def _create_minimax(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.minimax import MiniMaxProvider

    return MiniMaxProvider(config, rate_limiter=rate_limiter)


def _create_opencode(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.opencode import OpenCodeProvider

    return OpenCodeProvider(config, rate_limiter=rate_limiter)


def _create_opencode_go(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.opencode import OpenCodeProvider

    return OpenCodeProvider(
        config,
        provider_name="OPENCODE_GO",
        rate_limiter=rate_limiter,
    )


def _create_vercel(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.vercel import VercelProvider

    return VercelProvider(config, rate_limiter=rate_limiter)


def _create_huggingface(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.huggingface import HuggingFaceProvider

    return HuggingFaceProvider(config, rate_limiter=rate_limiter)


def _create_cohere(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.cohere import CohereProvider

    return CohereProvider(config, rate_limiter=rate_limiter)


def _create_github_models(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.github_models import GitHubModelsProvider

    return GitHubModelsProvider(config, rate_limiter=rate_limiter)


def _create_zai(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.zai import ZaiProvider

    return ZaiProvider(config, rate_limiter=rate_limiter)


def _create_fireworks(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.fireworks import FireworksProvider

    return FireworksProvider(config, rate_limiter=rate_limiter)


def _create_cloudflare(
    config: ProviderConfig,
    settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.cloudflare import CloudflareProvider

    return CloudflareProvider(
        config,
        account_id=settings.cloudflare_account_id,
        rate_limiter=rate_limiter,
    )


def _create_gemini(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.gemini import GeminiProvider

    return GeminiProvider(config, rate_limiter=rate_limiter)


def _create_groq(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.groq import GroqProvider

    return GroqProvider(config, rate_limiter=rate_limiter)


def _create_sambanova(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.sambanova import SambaNovaProvider

    return SambaNovaProvider(config, rate_limiter=rate_limiter)


def _create_cerebras(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.cerebras import CerebrasProvider

    return CerebrasProvider(config, rate_limiter=rate_limiter)


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
        raise UnknownProviderError.for_provider(provider_id, PROVIDER_CATALOG)

    factory = PROVIDER_FACTORIES.get(provider_id)
    if factory is None:
        raise AssertionError(f"Unhandled provider descriptor: {provider_id}")
    config = build_provider_config(descriptor, settings)
    rate_limiter = ProviderRateLimiter(
        rate_limit=config.rate_limit or 40,
        rate_window=config.rate_window or 60.0,
        max_concurrency=config.max_concurrency,
    )
    return factory(config, settings, rate_limiter)
