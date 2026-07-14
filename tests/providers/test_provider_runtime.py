import asyncio
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.application.errors import UnknownProviderError
from free_claude_code.config.nim import NimSettings
from free_claude_code.config.provider_catalog import (
    COHERE_DEFAULT_BASE,
    GITHUB_MODELS_DEFAULT_BASE,
    HUGGINGFACE_DEFAULT_BASE,
    MINIMAX_DEFAULT_BASE,
    OLLAMA_CLOUD_DEFAULT_BASE,
    PROVIDER_CATALOG,
    SUPPORTED_PROVIDER_IDS,
    VERCEL_AI_GATEWAY_DEFAULT_BASE,
    ZAI_DEFAULT_BASE,
)
from free_claude_code.providers.cloudflare import CloudflareProvider
from free_claude_code.providers.deepseek import DeepSeekProvider
from free_claude_code.providers.gemini import GeminiProvider
from free_claude_code.providers.github_models import GitHubModelsProvider
from free_claude_code.providers.lmstudio import LMStudioProvider
from free_claude_code.providers.mistral import MistralProvider
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.open_router import OpenRouterProvider
from free_claude_code.providers.openai_chat import (
    OPENAI_CHAT_PROFILES,
    OpenAIChatProvider,
)
from free_claude_code.providers.rate_limit import ProviderRateLimiter
from free_claude_code.providers.runtime import (
    ProviderRuntime,
    build_provider_config,
    create_provider,
)


def _make_settings(**overrides):
    mock = MagicMock()
    mock.model = "nvidia_nim/meta/llama3"
    mock.model_fable = None
    mock.model_opus = None
    mock.model_sonnet = None
    mock.model_haiku = None
    mock.nvidia_nim_api_key = "test_key"
    mock.open_router_api_key = "test_openrouter_key"
    mock.mistral_api_key = "test_mistral_key"
    mock.codestral_api_key = "test_codestral_key"
    mock.deepseek_api_key = "test_deepseek_key"
    mock.wafer_api_key = "test_wafer_key"
    mock.minimax_api_key = "test_minimax_key"
    mock.opencode_api_key = "test_opencode_key"
    mock.vercel_ai_gateway_api_key = "test_vercel_key"
    mock.huggingface_api_key = "test_huggingface_key"
    mock.cohere_api_key = "test_cohere_key"
    mock.github_models_token = "test_github_models_token"
    mock.zai_api_key = "test_zai_key"
    mock.lm_studio_base_url = "http://localhost:1234/v1"
    mock.llamacpp_base_url = "http://localhost:8080/v1"
    mock.ollama_base_url = "http://localhost:11434"
    mock.ollama_api_key = "test_ollama_cloud_key"
    mock.nvidia_nim_proxy = ""
    mock.open_router_proxy = ""
    mock.lmstudio_proxy = ""
    mock.llamacpp_proxy = ""
    mock.mistral_proxy = ""
    mock.codestral_proxy = ""
    mock.kimi_proxy = ""
    mock.kimi_api_key = "test_kimi_key"
    mock.wafer_proxy = ""
    mock.minimax_proxy = ""
    mock.opencode_proxy = ""
    mock.opencode_go_proxy = ""
    mock.vercel_ai_gateway_proxy = ""
    mock.huggingface_proxy = ""
    mock.cohere_proxy = ""
    mock.github_models_proxy = ""
    mock.zai_proxy = ""
    mock.fireworks_proxy = ""
    mock.fireworks_api_key = "test_fireworks_key"
    mock.cloudflare_api_token = "test_cloudflare_token"
    mock.cloudflare_account_id = "test_cloudflare_account"
    mock.cloudflare_proxy = ""
    mock.gemini_api_key = ""
    mock.gemini_proxy = ""
    mock.groq_api_key = ""
    mock.groq_proxy = ""
    mock.cerebras_api_key = ""
    mock.cerebras_proxy = ""
    mock.ollama_cloud_proxy = ""
    mock.provider_rate_limit = 40
    mock.provider_rate_window = 60
    mock.provider_max_concurrency = 5
    mock.http_read_timeout = 300.0
    mock.http_write_timeout = 10.0
    mock.http_connect_timeout = 10.0
    mock.enable_model_thinking = True
    mock.log_raw_sse_events = False
    mock.log_api_error_tracebacks = False
    mock.nim = NimSettings()
    for key, value in overrides.items():
        setattr(mock, key, value)
    return mock


def test_importing_runtime_does_not_eager_load_other_adapters() -> None:
    """Runtime metadata must not import every provider adapter up front."""
    code = (
        "import sys\n"
        "import free_claude_code.providers.runtime\n"
        "assert 'free_claude_code.providers.open_router' not in sys.modules\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_provider_catalog_covers_advertised_provider_ids():
    assert set(PROVIDER_CATALOG) == set(SUPPORTED_PROVIDER_IDS)
    assert set(OPENAI_CHAT_PROFILES) < set(PROVIDER_CATALOG)
    for descriptor in PROVIDER_CATALOG.values():
        assert descriptor.provider_id


def test_ollama_descriptor_uses_local_openai_endpoint_semantics():
    descriptor = PROVIDER_CATALOG["ollama"]

    assert descriptor.default_base_url == "http://localhost:11434"
    assert descriptor.local is True


def test_ollama_cloud_descriptor_uses_direct_authenticated_endpoint():
    descriptor = PROVIDER_CATALOG["ollama_cloud"]

    assert descriptor.default_base_url == OLLAMA_CLOUD_DEFAULT_BASE
    assert descriptor.credential_env == "OLLAMA_API_KEY"
    assert descriptor.credential_attr == "ollama_api_key"
    assert descriptor.base_url_attr is None
    assert descriptor.local is False


def test_ollama_cloud_provider_config_uses_key_and_proxy():
    descriptor = PROVIDER_CATALOG["ollama_cloud"]
    settings = _make_settings(
        ollama_api_key="ollama-cloud-token",
        ollama_cloud_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)

    assert config.api_key == "ollama-cloud-token"
    assert config.base_url == OLLAMA_CLOUD_DEFAULT_BASE
    assert config.proxy == "http://proxy.test:8080"


@pytest.mark.parametrize(
    ("provider_id", "expected_api_key"),
    [
        ("lmstudio", "lm-studio"),
        ("llamacpp", "llamacpp"),
        ("ollama", "ollama"),
    ],
)
def test_local_provider_factory_resolves_catalog_static_credential(
    provider_id: str,
    expected_api_key: str,
) -> None:
    descriptor = PROVIDER_CATALOG[provider_id]
    settings = _make_settings()

    config = build_provider_config(descriptor, settings)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider(provider_id, settings)

    assert config.api_key == expected_api_key
    assert isinstance(provider, OpenAIChatProvider)
    assert provider._api_key == expected_api_key


def test_zai_descriptor_uses_fixed_cloud_base_url():
    descriptor = PROVIDER_CATALOG["zai"]

    assert descriptor.default_base_url == ZAI_DEFAULT_BASE
    assert descriptor.base_url_attr is None


def test_zai_provider_config_ignores_stale_base_url_setting():
    descriptor = PROVIDER_CATALOG["zai"]

    config = build_provider_config(
        descriptor,
        _make_settings(zai_base_url="https://custom.zai.invalid/v1"),
    )

    assert config.base_url == ZAI_DEFAULT_BASE


def test_minimax_descriptor_uses_expected_endpoint_and_credential():
    descriptor = PROVIDER_CATALOG["minimax"]

    assert descriptor.default_base_url == MINIMAX_DEFAULT_BASE
    assert descriptor.credential_env == "MINIMAX_API_KEY"


def test_cloudflare_descriptor_uses_api_root_not_account_url():
    descriptor = PROVIDER_CATALOG["cloudflare"]

    assert descriptor.default_base_url == "https://api.cloudflare.com/client/v4"
    assert descriptor.base_url_attr is None


def test_create_cloudflare_provider_uses_account_scoped_base_url():
    settings = _make_settings(
        cloudflare_api_token="test_cloudflare_token",
        cloudflare_account_id="test-account",
    )

    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("cloudflare", settings)

    assert isinstance(provider, CloudflareProvider)
    assert provider._base_url == (
        "https://api.cloudflare.com/client/v4/accounts/test-account/ai/v1"
    )


def test_opencode_go_provider_config_uses_correct_base_url_and_name():
    with patch("httpx.AsyncClient"):
        provider = create_provider("opencode_go", _make_settings())

    assert isinstance(provider, OpenAIChatProvider)
    assert provider._base_url == "https://opencode.ai/zen/go/v1"
    assert provider._provider_name == "OPENCODE_GO"
    assert provider._api_key == "test_opencode_key"


def test_opencode_go_catalog_uses_opencode_api_key() -> None:
    desc = PROVIDER_CATALOG["opencode_go"]

    assert desc.credential_env == "OPENCODE_API_KEY"
    assert desc.credential_attr == "opencode_api_key"


def test_build_provider_config_opencode_go_uses_opencode_api_key() -> None:
    descriptor = PROVIDER_CATALOG["opencode_go"]
    settings = _make_settings(opencode_api_key="shared-opencode-token")

    config = build_provider_config(descriptor, settings)

    assert config.api_key == "shared-opencode-token"


def test_vercel_descriptor_uses_openai_chat_gateway() -> None:
    descriptor = PROVIDER_CATALOG["vercel"]

    assert descriptor.default_base_url == VERCEL_AI_GATEWAY_DEFAULT_BASE
    assert descriptor.credential_env == "AI_GATEWAY_API_KEY"
    assert descriptor.proxy_attr == "vercel_ai_gateway_proxy"


def test_huggingface_descriptor_uses_openai_chat_router() -> None:
    descriptor = PROVIDER_CATALOG["huggingface"]

    assert descriptor.default_base_url == HUGGINGFACE_DEFAULT_BASE
    assert descriptor.credential_env == "HUGGINGFACE_API_KEY"
    assert descriptor.proxy_attr == "huggingface_proxy"


def test_cohere_descriptor_uses_openai_chat_compatibility_api() -> None:
    descriptor = PROVIDER_CATALOG["cohere"]

    assert descriptor.default_base_url == COHERE_DEFAULT_BASE
    assert descriptor.credential_env == "COHERE_API_KEY"
    assert descriptor.proxy_attr == "cohere_proxy"


def test_github_models_descriptor_uses_openai_chat_inference_api() -> None:
    descriptor = PROVIDER_CATALOG["github_models"]

    assert descriptor.default_base_url == GITHUB_MODELS_DEFAULT_BASE
    assert descriptor.credential_env == "GITHUB_MODELS_TOKEN"
    assert descriptor.proxy_attr == "github_models_proxy"


def test_build_provider_config_vercel_uses_gateway_key_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["vercel"]
    settings = _make_settings(
        vercel_ai_gateway_api_key="vercel-token",
        vercel_ai_gateway_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)

    assert config.api_key == "vercel-token"
    assert config.proxy == "http://proxy.test:8080"


def test_build_provider_config_huggingface_uses_api_key_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["huggingface"]
    settings = _make_settings(
        huggingface_api_key="hf-token",
        huggingface_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)

    assert config.api_key == "hf-token"
    assert config.proxy == "http://proxy.test:8080"


def test_build_provider_config_cohere_uses_api_key_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["cohere"]
    settings = _make_settings(
        cohere_api_key="cohere-token",
        cohere_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)

    assert config.api_key == "cohere-token"
    assert config.proxy == "http://proxy.test:8080"


def test_build_provider_config_github_models_uses_token_and_proxy() -> None:
    descriptor = PROVIDER_CATALOG["github_models"]
    settings = _make_settings(
        github_models_token="github-token",
        github_models_proxy="http://proxy.test:8080",
    )

    config = build_provider_config(descriptor, settings)

    assert config.api_key == "github-token"
    assert config.proxy == "http://proxy.test:8080"


def test_create_provider_uses_openai_chat_openrouter_by_default():
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = create_provider("open_router", _make_settings())

    assert isinstance(provider, OpenRouterProvider)


def test_create_provider_instantiates_each_builtin():
    settings = _make_settings(
        gemini_api_key="test_gemini_key",
        groq_api_key="test_groq_key",
        cerebras_api_key="test_cerebras_key",
        fireworks_api_key="test_fireworks_key",
        cloudflare_api_token="test_cloudflare_token",
        cloudflare_account_id="test_cloudflare_account",
        vercel_ai_gateway_api_key="test_vercel_key",
        huggingface_api_key="test_huggingface_key",
        cohere_api_key="test_cohere_key",
        github_models_token="test_github_models_token",
        kimi_api_key="test_kimi_key",
        provider_rate_limit=7,
        provider_rate_window=11,
        provider_max_concurrency=3,
        sambanova_api_key="test_sambanova_key",
    )
    cases = {
        "nvidia_nim": NvidiaNimProvider,
        "open_router": OpenRouterProvider,
        "mistral": MistralProvider,
        "mistral_codestral": OpenAIChatProvider,
        "deepseek": DeepSeekProvider,
        "kimi": OpenAIChatProvider,
        "minimax": OpenAIChatProvider,
        "fireworks": OpenAIChatProvider,
        "cloudflare": CloudflareProvider,
        "lmstudio": LMStudioProvider,
        "llamacpp": OpenAIChatProvider,
        "ollama": OpenAIChatProvider,
        "ollama_cloud": OpenAIChatProvider,
        "wafer": OpenAIChatProvider,
        "opencode": OpenAIChatProvider,
        "opencode_go": OpenAIChatProvider,
        "vercel": OpenAIChatProvider,
        "huggingface": OpenAIChatProvider,
        "cohere": OpenAIChatProvider,
        "github_models": GitHubModelsProvider,
        "zai": OpenAIChatProvider,
        "gemini": GeminiProvider,
        "groq": OpenAIChatProvider,
        "sambanova": OpenAIChatProvider,
        "cerebras": OpenAIChatProvider,
    }
    sentinel_limiter = MagicMock(spec=ProviderRateLimiter)

    with (
        patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"),
        patch("httpx.AsyncClient"),
        patch(
            "free_claude_code.providers.runtime.factory.ProviderRateLimiter",
            return_value=sentinel_limiter,
        ) as limiter_factory,
    ):
        for provider_id, provider_cls in cases.items():
            provider = create_provider(provider_id, settings)

            assert isinstance(provider, provider_cls)
            assert provider._rate_limiter is sentinel_limiter
            limiter_factory.assert_called_once_with(
                rate_limit=7,
                rate_window=11,
                max_concurrency=3,
            )
            limiter_factory.reset_mock()

    assert set(cases) == set(PROVIDER_CATALOG)


def test_provider_runtime_caches_by_provider_id():
    runtime = ProviderRuntime(_make_settings())

    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        first = runtime.resolve_provider("nvidia_nim")
        second = runtime.resolve_provider("nvidia_nim")

    assert first is second


def test_provider_runtime_provider_owns_one_limiter() -> None:
    runtime = ProviderRuntime(_make_settings())

    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        first = runtime.resolve_provider("nvidia_nim")
        second = runtime.resolve_provider("nvidia_nim")

    assert isinstance(first, NvidiaNimProvider)
    assert isinstance(second, NvidiaNimProvider)
    assert first._rate_limiter is second._rate_limiter


def test_separate_provider_runtimes_never_share_limiters() -> None:
    first_runtime = ProviderRuntime(_make_settings())
    second_runtime = ProviderRuntime(_make_settings())

    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        first = first_runtime.resolve_provider("nvidia_nim")
        second = second_runtime.resolve_provider("nvidia_nim")

    assert isinstance(first, NvidiaNimProvider)
    assert isinstance(second, NvidiaNimProvider)
    assert first is not second
    assert first._rate_limiter is not second._rate_limiter


def test_different_providers_in_one_runtime_have_independent_limiters() -> None:
    runtime = ProviderRuntime(_make_settings())

    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        nim = runtime.resolve_provider("nvidia_nim")
        open_router = runtime.resolve_provider("open_router")

    assert isinstance(nim, NvidiaNimProvider)
    assert isinstance(open_router, OpenRouterProvider)
    assert nim._rate_limiter is not open_router._rate_limiter


def test_unknown_provider_raises_unknown_provider_type_error():
    with pytest.raises(UnknownProviderError, match="Unknown provider_type"):
        create_provider("unknown", _make_settings())


@pytest.mark.asyncio
async def test_provider_runtime_cleanup_runs_all_even_if_one_fails() -> None:
    """Successful providers leave the cache while failed providers remain retryable."""
    p1 = MagicMock()
    p1.cleanup = AsyncMock(side_effect=RuntimeError("first"))
    p2 = MagicMock()
    p2.cleanup = AsyncMock()
    runtime = ProviderRuntime(_make_settings(), {"a": p1, "b": p2})

    with pytest.raises(RuntimeError, match="first"):
        await runtime.cleanup()

    p1.cleanup.assert_awaited_once()
    p2.cleanup.assert_awaited_once()
    assert runtime.is_cached("a")
    assert not runtime.is_cached("b")

    p1.cleanup = AsyncMock()
    await runtime.cleanup()

    p1.cleanup.assert_awaited_once()
    assert not runtime.is_cached("a")


@pytest.mark.asyncio
async def test_cancelled_cleanup_retains_current_and_unvisited_providers() -> None:
    first = MagicMock()
    second = MagicMock()
    third = MagicMock()
    second_started = asyncio.Event()
    second_attempts = 0

    async def cleanup_second() -> None:
        nonlocal second_attempts
        second_attempts += 1
        if second_attempts == 1:
            second_started.set()
            await asyncio.Event().wait()

    first.cleanup = AsyncMock()
    second.cleanup = AsyncMock(side_effect=cleanup_second)
    third.cleanup = AsyncMock()
    runtime = ProviderRuntime(
        _make_settings(),
        {"first": first, "second": second, "third": third},
    )
    cleanup_task = asyncio.create_task(runtime.cleanup())
    await second_started.wait()

    cleanup_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cleanup_task

    assert runtime.is_cached("first") is False
    assert runtime.is_cached("second") is True
    assert runtime.is_cached("third") is True
    first.cleanup.assert_awaited_once_with()
    third.cleanup.assert_not_awaited()

    await runtime.cleanup()

    first.cleanup.assert_awaited_once_with()
    assert second.cleanup.await_count == 2
    third.cleanup.assert_awaited_once_with()
    assert runtime.is_cached("first") is False
    assert runtime.is_cached("second") is False
    assert runtime.is_cached("third") is False


@pytest.mark.asyncio
async def test_provider_runtime_cleanup_exceptiongroup_on_multiple_failures() -> None:
    p1 = MagicMock()
    p1.cleanup = AsyncMock(side_effect=RuntimeError("a"))
    p2 = MagicMock()
    p2.cleanup = AsyncMock(side_effect=RuntimeError("b"))
    runtime = ProviderRuntime(_make_settings(), {"x": p1, "y": p2})

    with pytest.raises(ExceptionGroup) as exc_info:
        await runtime.cleanup()

    assert len(exc_info.value.exceptions) == 2
    assert runtime.is_cached("x")
    assert runtime.is_cached("y")

    p1.cleanup = AsyncMock()
    p2.cleanup = AsyncMock()
    await runtime.cleanup()

    assert not runtime.is_cached("x")
    assert not runtime.is_cached("y")
