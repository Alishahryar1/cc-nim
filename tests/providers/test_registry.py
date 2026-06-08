import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.nim import NimSettings
from config.provider_ids import SUPPORTED_PROVIDER_IDS
from providers.deepseek import DeepSeekProvider
from providers.exceptions import UnknownProviderTypeError
from providers.llamacpp import LlamaCppProvider
from providers.lmstudio import LMStudioProvider
from providers.nvidia_nim import NvidiaNimProvider
from providers.ollama import OllamaProvider
from providers.open_router import OpenRouterProvider
from providers.registry import (
    PROVIDER_DESCRIPTORS,
    ProviderRegistry,
    create_provider,
)
from providers.wafer import WaferProvider


def _make_settings(**overrides):
    mock = MagicMock()
    mock.model = "nvidia_nim/meta/llama3"
    mock.provider_type = "nvidia_nim"
    mock.nvidia_nim_api_key = "test_key"
    mock.open_router_api_key = "test_openrouter_key"
    mock.deepseek_api_key = "test_deepseek_key"
    mock.wafer_api_key = "test_wafer_key"
    mock.lm_studio_base_url = "http://localhost:1234/v1"
    mock.llamacpp_base_url = "http://localhost:8080/v1"
    mock.ollama_base_url = "http://localhost:11434"
    mock.nvidia_nim_proxy = ""
    mock.open_router_proxy = ""
    mock.lmstudio_proxy = ""
    mock.llamacpp_proxy = ""
    mock.kimi_proxy = ""
    mock.wafer_proxy = ""
    mock.provider_rate_limit = 40
    mock.provider_rate_window = 60
    mock.provider_max_concurrency = 5
    mock.http_read_timeout = 300.0
    mock.http_write_timeout = 10.0
    mock.http_connect_timeout = 10.0
    mock.enable_model_thinking = True
    mock.nim = NimSettings()
    for key, value in overrides.items():
        setattr(mock, key, value)
    return mock


def test_importing_registry_does_not_eager_load_other_adapters() -> None:
    """Registry metadata must not import every provider adapter up front."""
    code = (
        "import sys\n"
        "import providers.registry\n"
        "assert 'providers.open_router' not in sys.modules\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_descriptors_cover_advertised_provider_ids():
    assert set(PROVIDER_DESCRIPTORS) == set(SUPPORTED_PROVIDER_IDS)
    for descriptor in PROVIDER_DESCRIPTORS.values():
        assert descriptor.provider_id
        assert descriptor.transport_type in {"openai_chat", "anthropic_messages"}
        assert descriptor.capabilities


def test_ollama_descriptor_uses_openai_chat_transport():
    descriptor = PROVIDER_DESCRIPTORS["ollama"]

    assert descriptor.transport_type == "openai_chat"
    assert descriptor.default_base_url == "http://localhost:11434"
    assert "local" in descriptor.capabilities


def test_create_provider_uses_native_openrouter_by_default():
    with patch("httpx.AsyncClient"):
        provider = create_provider("open_router", _make_settings())

    assert isinstance(provider, OpenRouterProvider)


def test_create_provider_instantiates_each_builtin():
    settings = _make_settings()
    cases = {
        "nvidia_nim": NvidiaNimProvider,
        "deepseek": DeepSeekProvider,
        "lmstudio": LMStudioProvider,
        "llamacpp": LlamaCppProvider,
        "ollama": OllamaProvider,
        "wafer": WaferProvider,
    }

    with (
        patch("providers.openai_compat.AsyncOpenAI"),
        patch("httpx.AsyncClient"),
    ):
        for provider_id, provider_cls in cases.items():
            assert isinstance(create_provider(provider_id, settings), provider_cls)


def test_provider_registry_caches_by_provider_id():
    registry = ProviderRegistry()
    settings = _make_settings()

    with patch("providers.openai_compat.AsyncOpenAI"):
        first = registry.get("nvidia_nim", settings)
        second = registry.get("nvidia_nim", settings)

    assert first is second


def test_unknown_provider_raises_unknown_provider_type_error():
    with pytest.raises(UnknownProviderTypeError, match="Unknown provider_type"):
        create_provider("unknown", _make_settings())


@pytest.mark.asyncio
async def test_provider_registry_cleanup_runs_all_even_if_one_fails() -> None:
    """Every provider gets cleanup; cache is cleared even when one raises."""
    reg = ProviderRegistry()
    p1 = MagicMock()
    p1.cleanup = AsyncMock(side_effect=RuntimeError("first"))
    p2 = MagicMock()
    p2.cleanup = AsyncMock()
    reg._providers["a"] = p1
    reg._providers["b"] = p2
    with pytest.raises(RuntimeError, match="first"):
        await reg.cleanup()
    p1.cleanup.assert_awaited_once()
    p2.cleanup.assert_awaited_once()
    assert reg._providers == {}


@pytest.mark.asyncio
async def test_provider_registry_cleanup_exceptiongroup_on_multiple_failures() -> None:
    reg = ProviderRegistry()
    p1 = MagicMock()
    p1.cleanup = AsyncMock(side_effect=RuntimeError("a"))
    p2 = MagicMock()
    p2.cleanup = AsyncMock(side_effect=RuntimeError("b"))
    reg._providers["x"] = p1
    reg._providers["y"] = p2
    with pytest.raises(ExceptionGroup) as exc_info:
        await reg.cleanup()
    assert len(exc_info.value.exceptions) == 2
    assert reg._providers == {}
