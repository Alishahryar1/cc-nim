"""Provider construction from declarative profiles and exceptional adapters."""

from collections.abc import Callable

from free_claude_code.application.errors import UnknownProviderError
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.openai_chat import (
    OPENAI_CHAT_PROFILES,
    create_openai_chat_provider,
)
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


def _create_vertex(
    config: ProviderConfig,
    settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.vertex import VertexProvider

    return VertexProvider(
        config,
        project_id=settings.vertex_project_id,
        location=settings.vertex_location,
        rate_limiter=rate_limiter,
    )


def _create_github_models(
    config: ProviderConfig,
    _settings: Settings,
    rate_limiter: ProviderRateLimiter,
) -> BaseProvider:
    from free_claude_code.providers.github_models import GitHubModelsProvider

    return GitHubModelsProvider(config, rate_limiter=rate_limiter)


_SPECIAL_PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "nvidia_nim": _create_nvidia_nim,
    "open_router": _create_open_router,
    "mistral": _create_mistral,
    "deepseek": _create_deepseek,
    "lmstudio": _create_lmstudio,
    "cloudflare": _create_cloudflare,
    "gemini": _create_gemini,
    "vertex": _create_vertex,
    "github_models": _create_github_models,
}

_profiled_ids = set(OPENAI_CHAT_PROFILES)
_special_ids = set(_SPECIAL_PROVIDER_FACTORIES)
if _profiled_ids & _special_ids or _profiled_ids | _special_ids != set(
    PROVIDER_CATALOG
):
    raise AssertionError(
        "Every provider must have exactly one construction owner: "
        f"profiles={_profiled_ids!r} special={_special_ids!r} "
        f"catalog={set(PROVIDER_CATALOG)!r}"
    )


def create_provider(provider_id: str, settings: Settings) -> BaseProvider:
    """Create a provider instance for a supported provider id."""
    descriptor = PROVIDER_CATALOG.get(provider_id)
    if descriptor is None:
        raise UnknownProviderError.for_provider(provider_id, PROVIDER_CATALOG)

    config = build_provider_config(descriptor, settings)
    rate_limiter = ProviderRateLimiter(
        rate_limit=config.rate_limit or 40,
        rate_window=config.rate_window or 60.0,
        max_concurrency=config.max_concurrency,
    )
    factory = _SPECIAL_PROVIDER_FACTORIES.get(provider_id)
    if factory is not None:
        return factory(config, settings, rate_limiter)
    return create_openai_chat_provider(provider_id, config, rate_limiter)
