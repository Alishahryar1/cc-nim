"""Provider construction from declarative profiles and exceptional adapters."""

from collections.abc import AsyncIterator, Callable

from free_claude_code.application.errors import UnknownProviderError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.openai_chat import (
    OPENAI_CHAT_PROFILES,
    create_openai_chat_provider,
)

from .config import build_provider_config

ProviderFactory = Callable[
    [ProviderConfig, Settings, ProviderAdmissionController], BaseProvider
]


def _create_nvidia_nim(
    config: ProviderConfig,
    settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.nvidia_nim import NvidiaNimProvider

    return NvidiaNimProvider(
        config,
        nim_settings=settings.nim,
        admission=admission,
    )


def _create_open_router(
    config: ProviderConfig,
    _settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.open_router import OpenRouterProvider

    return OpenRouterProvider(config, admission=admission)


def _create_mistral(
    config: ProviderConfig,
    _settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.mistral import MistralProvider

    return MistralProvider(config, admission=admission)


def _create_deepseek(
    config: ProviderConfig,
    _settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.deepseek import DeepSeekProvider

    return DeepSeekProvider(config, admission=admission)


def _create_lmstudio(
    config: ProviderConfig,
    _settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.lmstudio import LMStudioProvider

    return LMStudioProvider(config, admission=admission)


def _create_ollama(
    config: ProviderConfig,
    _settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:

    return _OllamaWrapper(config, admission)


def _create_llamacpp(
    config: ProviderConfig,
    _settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:

    return _LlamaCPPWrapper(config, admission)


class _OllamaWrapper(BaseProvider):
    def __init__(self, config: ProviderConfig, admission: ProviderAdmissionController):
        super().__init__(config)
        from free_claude_code.providers.ollama import OllamaProvider

        self._provider = OllamaProvider(base_url=config.base_url)

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        return frozenset([ProviderModelInfo(model_id="llama3.1")])

    def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:

        # Convert request to format expected by OllamaProvider
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        model = request.model or "llama3.1"

        async def _gen():
            # execute is async, returns an async generator function when stream=True
            generator_fn = await self._provider.execute(
                messages, model=model, stream=True
            )
            if generator_fn is not None:
                async for chunk in generator_fn():
                    yield chunk

        return _gen()


class _LlamaCPPWrapper(BaseProvider):
    def __init__(self, config: ProviderConfig, admission: ProviderAdmissionController):
        super().__init__(config)
        from free_claude_code.providers.llamacpp import LlamaCPPProvider

        self._provider = LlamaCPPProvider(base_url=config.base_url)

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        return frozenset([ProviderModelInfo(model_id="default")])

    def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        model = request.model or "default"

        async def _gen():
            generator_fn = await self._provider.execute(
                messages, model=model, stream=True
            )
            if generator_fn is not None:
                async for chunk in generator_fn():
                    yield chunk

        return _gen()


def _create_cloudflare(
    config: ProviderConfig,
    settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.cloudflare import CloudflareProvider

    return CloudflareProvider(
        config,
        account_id=settings.cloudflare_account_id,
        admission=admission,
    )


def _create_gemini(
    config: ProviderConfig,
    _settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.gemini import GeminiProvider

    return GeminiProvider(config, admission=admission)


def _create_vertex(
    config: ProviderConfig,
    settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.vertex import VertexProvider

    return VertexProvider(
        config,
        project_id=settings.vertex_project_id,
        location=settings.vertex_location,
        admission=admission,
    )


def _create_github_models(
    config: ProviderConfig,
    _settings: Settings,
    admission: ProviderAdmissionController,
) -> BaseProvider:
    from free_claude_code.providers.github_models import GitHubModelsProvider

    return GitHubModelsProvider(config, admission=admission)


_SPECIAL_PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "nvidia_nim": _create_nvidia_nim,
    "open_router": _create_open_router,
    "mistral": _create_mistral,
    "deepseek": _create_deepseek,
    "lmstudio": _create_lmstudio,
    "ollama": _create_ollama,
    "llamacpp": _create_llamacpp,
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
    admission = ProviderAdmissionController(
        provider_name=provider_id,
        rate_limit=config.rate_limit or 40,
        rate_window=config.rate_window or 60.0,
        max_concurrency=config.max_concurrency,
    )
    factory = _SPECIAL_PROVIDER_FACTORIES.get(provider_id)
    if factory is not None:
        return factory(config, settings, admission)
    return create_openai_chat_provider(provider_id, config, admission)
