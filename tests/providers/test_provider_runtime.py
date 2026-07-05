import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.nim import NimSettings
from config.provider_catalog import (
    COHERE_DEFAULT_BASE,
    MINIMAX_DEFAULT_BASE,
    PROVIDER_CATALOG,
    ZAI_DEFAULT_BASE,
)
from config.provider_ids import SUPPORTED_PROVIDER_IDS
from providers.cerebras import CerebrasProvider
from providers.cloudflare import CloudflareProvider
from providers.codestral import CodestralProvider
from providers.cohere import CohereProvider
from providers.deepseek import DeepSeekProvider
from providers.exceptions import UnknownProviderTypeError
from providers.fireworks import FireworksProvider
from providers.gemini import GeminiProvider
from providers.groq import GroqProvider
from providers.huggingface import HUGGINGFACE_DEFAULT_BASE, HuggingFaceProvider
from providers.kimi import KimiProvider
from providers.llamacpp import LlamaCppProvider
from providers.lmstudio import LMStudioProvider
from providers.minimax import MiniMaxProvider
from providers.mistral import MistralProvider
from providers.nvidia_nim import NvidiaNimProvider
from providers.ollama import OllamaProvider
from providers.open_router import OpenRouterProvider
from providers.opencode import OpenCodeProvider
from providers.runtime import ProviderRuntime, build_provider_config, create_provider
from providers.vercel import VERCEL_AI_GATEWAY_DEFAULT_BASE, VercelProvider
from providers.wafer import WaferProvider
from providers.zai import ZaiProvider


def _make_settings(**overrides):
    mock = MagicMock()
    mock.model = "nvidia_nim/meta/llama3"
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
    mock.zai_api_key = "test_zai_key"
    mock.lm_studio_base_url = "http://localhost:1234/v1"
    mock.llamacpp_base_url = "http://localhost:8080/v1"
    mock.ollama_base_url = "http://localhost:11434"
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
        "import providers.runtime\n"
        "assert 'providers.open_router' not in sys.modules\n"
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
    for descriptor in PROVIDER_CATALOG.values():
        assert descriptor.provider_id
        assert descriptor.transport_type in {"openai_chat", "anthropic_messages"}
        assert descriptor.capabilities


def test_ollama_descriptor_uses_native_anthropic_transport():
    descriptor = PROVIDER_CATALOG["ollama"]

    assert descriptor.transport_type == "anthropic_messages"
    assert descriptor.default_base_url == "http://localhost:11434"
    assert "native_anthropic" in descriptor.capabilities


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


def test_minimax_descriptor_uses_native_anthropic_transport():
    descriptor = PROVIDER_CATALOG["minimax"]

    assert descriptor.transport_type == "anthropic_messages"
    assert descriptor.default_base_url == MINIMAX_DEFAULT_BASE
    assert descriptor.credential_env == "MINIMAX_API_KEY"
    assert "native_anthropic" in descriptor.capabilities


def test_cloudflare_descriptor_uses_api_root_not_account_url():
    descriptor = PROVIDER_CATALOG["cloudflare"]

    assert descriptor.transport_type == "openai_chat"
    assert descriptor.default_base_url == "https://api.cloudflare.com/client/v4"
    assert descriptor.base_url_attr is None
    assert "native_anthropic" not in descriptor.capabilities
    assert "thinking" in descriptor.capabilities


def test_create_cloudflare_provider_uses_account_scoped_base_url():
    settings = _make_settings(
        cloudflare_api_token="test_cloudflare_token",
        cloudflare_account_id="test-account",
    )

    with patch("providers.transports.openai_chat.transport.AsyncOpenAI"):
        provider = create_provider("cloudflare", settings)

    assert isinstance(provider, CloudflareProvider)
    assert provider._base_url == (
        "https://api.cloudflare.com/client/v4/accounts/test-account/ai/v1"
    )


def test_opencode_go_provider_config_uses_correct_base_url_and_name():
    with patch("httpx.AsyncClient"):
        provider = create_provider("opencode_go", _make_settings())

    assert isinstance(provider, OpenCodeProvider)
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

    assert descriptor.transport_type == "openai_chat"
    assert descriptor.default_base_url == VERCEL_AI_GATEWAY_DEFAULT_BASE
    assert descriptor.credential_env == "AI_GATEWAY_API_KEY"
    assert descriptor.proxy_attr == "vercel_ai_gateway_proxy"
    assert "thinking" in descriptor.capabilities


def test_huggingface_descriptor_uses_openai_chat_router() -> None:
    descriptor = PROVIDER_CATALOG["huggingface"]

    assert descriptor.transport_type == "openai_chat"
    assert descriptor.default_base_url == HUGGINGFACE_DEFAULT_BASE
    assert descriptor.credential_env == "HUGGINGFACE_API_KEY"
    assert descriptor.proxy_attr == "huggingface_proxy"
    assert "thinking" in descriptor.capabilities


def test_cohere_descriptor_uses_openai_chat_compatibility_api() -> None:
    descriptor = PROVIDER_CATALOG["cohere"]

    assert descriptor.transport_type == "openai_chat"
    assert descriptor.default_base_url == COHERE_DEFAULT_BASE
    assert descriptor.credential_env == "COHERE_API_KEY"
    assert descriptor.proxy_attr == "cohere_proxy"
    assert "thinking" in descriptor.capabilities


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


def test_create_provider_uses_native_openrouter_by_default():
    with patch("httpx.AsyncClient"):
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
        kimi_api_key="test_kimi_key",
    )
    cases = {
        "nvidia_nim": NvidiaNimProvider,
        "mistral": MistralProvider,
        "mistral_codestral": CodestralProvider,
        "deepseek": DeepSeekProvider,
        "kimi": KimiProvider,
        "minimax": MiniMaxProvider,
        "fireworks": FireworksProvider,
        "cloudflare": CloudflareProvider,
        "lmstudio": LMStudioProvider,
        "llamacpp": LlamaCppProvider,
        "ollama": OllamaProvider,
        "wafer": WaferProvider,
        "opencode": OpenCodeProvider,
        "opencode_go": OpenCodeProvider,
        "vercel": VercelProvider,
        "huggingface": HuggingFaceProvider,
        "cohere": CohereProvider,
        "zai": ZaiProvider,
        "gemini": GeminiProvider,
        "groq": GroqProvider,
        "cerebras": CerebrasProvider,
    }

    with (
        patch("providers.transports.openai_chat.transport.AsyncOpenAI"),
        patch("httpx.AsyncClient"),
    ):
        for provider_id, provider_cls in cases.items():
            assert isinstance(create_provider(provider_id, settings), provider_cls)


def test_provider_runtime_caches_by_provider_id():
    runtime = ProviderRuntime(_make_settings())

    with patch("providers.transports.openai_chat.transport.AsyncOpenAI"):
        first = runtime.resolve_provider("nvidia_nim")
        second = runtime.resolve_provider("nvidia_nim")

    assert first is second


def test_unknown_provider_raises_unknown_provider_type_error():
    with pytest.raises(UnknownProviderTypeError, match="Unknown provider_type"):
        create_provider("unknown", _make_settings())


@pytest.mark.asyncio
async def test_provider_runtime_cleanup_runs_all_even_if_one_fails() -> None:
    """Every provider gets cleanup; cache is cleared even when one raises."""
    p1 = MagicMock()
    p1.cleanup = AsyncMock(side_effect=RuntimeError("first"))
    p2 = MagicMock()
    p2.cleanup = AsyncMock()
    runtime = ProviderRuntime(_make_settings(), {"a": p1, "b": p2})

    with pytest.raises(RuntimeError, match="first"):
        await runtime.cleanup()

    p1.cleanup.assert_awaited_once()
    p2.cleanup.assert_awaited_once()
    assert not runtime.is_cached("a")
    assert not runtime.is_cached("b")


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
    assert not runtime.is_cached("x")
    assert not runtime.is_cached("y")
