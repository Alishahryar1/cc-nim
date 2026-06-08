from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.applications import Starlette
from starlette.datastructures import State

from api.dependencies import (
    cleanup_provider,
    get_provider,
    get_provider_for_type,
    get_settings,
    resolve_provider,
)
from config.nim import NimSettings
from providers.deepseek import DeepSeekProvider
from providers.exceptions import ServiceUnavailableError, UnknownProviderTypeError
from providers.lmstudio import LMStudioProvider
from providers.nvidia_nim import NvidiaNimProvider
from providers.ollama import OllamaProvider
from providers.open_router import OpenRouterProvider
from providers.registry import ProviderRegistry
from providers.wafer import WaferProvider


def _make_mock_settings(**overrides):
    """Create a mock settings object with all required fields for get_provider()."""
    mock = MagicMock()
    mock.model = "nvidia_nim/meta/llama3"
    mock.provider_type = "nvidia_nim"
    mock.nvidia_nim_api_key = "test_key"
    mock.provider_rate_limit = 40
    mock.provider_rate_window = 60
    mock.provider_max_concurrency = 5
    mock.open_router_api_key = "test_openrouter_key"
    mock.deepseek_api_key = "test_deepseek_key"
    mock.wafer_api_key = "test_wafer_key"
    mock.lm_studio_base_url = "http://localhost:1234/v1"
    mock.ollama_base_url = "http://localhost:11434"
    mock.lmstudio_proxy = ""
    mock.llamacpp_proxy = ""
    mock.kimi_proxy = ""
    mock.wafer_proxy = ""
    mock.nim = NimSettings()
    mock.http_read_timeout = 300.0
    mock.http_write_timeout = 10.0
    mock.http_connect_timeout = 10.0
    mock.enable_model_thinking = True
    for key, value in overrides.items():
        setattr(mock, key, value)
    return mock


@pytest.fixture(autouse=True)
def reset_provider():
    """Reset the global _providers registry between tests."""
    import api.dependencies

    saved = api.dependencies._providers
    api.dependencies._providers = {}
    yield
    api.dependencies._providers = saved


@pytest.mark.asyncio
async def test_get_provider_singleton():
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings()

        p1 = get_provider()
        p2 = get_provider()

        assert isinstance(p1, NvidiaNimProvider)
        assert p1 is p2


@pytest.mark.asyncio
async def test_get_settings():
    settings = get_settings()
    assert settings is not None
    # Verify it calls the internal _get_settings
    with patch("api.dependencies._get_settings") as mock_get:
        get_settings()
        mock_get.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_provider():
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings()

        provider = get_provider()
        assert isinstance(provider, NvidiaNimProvider)
        provider._client = AsyncMock()

        await cleanup_provider()

        provider._client.close.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_provider_no_client():
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings()

        provider = get_provider()
        if hasattr(provider, "_client"):
            del provider._client

        await cleanup_provider()
        # Should not raise


@pytest.mark.asyncio
async def test_get_provider_open_router():
    """Test that provider_type=open_router returns OpenRouterProvider."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(provider_type="open_router")

        provider = get_provider()

        assert isinstance(provider, OpenRouterProvider)
        assert provider._base_url == "https://openrouter.ai/api/v1"
        assert provider._api_key == "test_openrouter_key"


@pytest.mark.asyncio
async def test_get_provider_lmstudio():
    """Test that provider_type=lmstudio returns LMStudioProvider."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(provider_type="lmstudio")

        provider = get_provider()

        assert isinstance(provider, LMStudioProvider)
        assert provider._base_url == "http://localhost:1234/v1"


@pytest.mark.asyncio
async def test_get_provider_ollama():
    """Test that provider_type=ollama returns OllamaProvider without an API key."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(provider_type="ollama")

        provider = get_provider()

        assert isinstance(provider, OllamaProvider)
        assert provider._base_url == "http://localhost:11434"
        assert provider._api_key == "ollama"


@pytest.mark.asyncio
async def test_get_provider_deepseek():
    """Test that provider_type=deepseek returns DeepSeekProvider."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(provider_type="deepseek")

        provider = get_provider()

        assert isinstance(provider, DeepSeekProvider)
        assert provider._base_url == "https://api.deepseek.com/anthropic"
        assert provider._api_key == "test_deepseek_key"
        assert provider._config.enable_thinking is True


@pytest.mark.asyncio
async def test_get_provider_deepseek_uses_fixed_base_url():
    """DeepSeek provider always uses the fixed provider base URL."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(
            provider_type="deepseek",
        )

        provider = get_provider()

        assert isinstance(provider, DeepSeekProvider)
        assert provider._base_url == "https://api.deepseek.com/anthropic"


@pytest.mark.asyncio
async def test_get_provider_deepseek_passes_enable_model_thinking():
    """DeepSeek provider receives the fallback thinking toggle."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(
            provider_type="deepseek",
            enable_model_thinking=False,
        )

        provider = get_provider()

        assert isinstance(provider, DeepSeekProvider)
        assert provider._config.enable_thinking is False


@pytest.mark.asyncio
async def test_get_provider_wafer():
    """Test that provider_type=wafer returns WaferProvider."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(provider_type="wafer")

        provider = get_provider()

        assert isinstance(provider, WaferProvider)
        assert provider._base_url == "https://pass.wafer.ai/v1"
        assert provider._api_key == "test_wafer_key"


@pytest.mark.asyncio
async def test_get_provider_lmstudio_uses_lm_studio_base_url():
    """LM Studio provider uses lm_studio_base_url from settings."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(
            provider_type="lmstudio",
            lm_studio_base_url="http://custom:9999/v1",
        )

        provider = get_provider()

        assert isinstance(provider, LMStudioProvider)
        assert provider._base_url == "http://custom:9999/v1"


@pytest.mark.asyncio
async def test_get_provider_passes_http_timeouts_from_settings():
    """Provider receives http timeouts from settings when creating client."""
    with (
        patch("api.dependencies.get_settings") as mock_settings,
        patch("providers.openai_compat.AsyncOpenAI") as mock_openai,
    ):
        mock_settings.return_value = _make_mock_settings(
            http_read_timeout=600.0,
            http_write_timeout=20.0,
            http_connect_timeout=5.0,
        )
        provider = get_provider()
        assert isinstance(provider, NvidiaNimProvider)
        call_kwargs = mock_openai.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 20.0
        assert timeout.connect == 5.0


@pytest.mark.asyncio
async def test_get_provider_passes_proxy_from_settings():
    """Provider receives configured proxy and builds a proxied HTTP client."""
    with (
        patch("api.dependencies.get_settings") as mock_settings,
        patch("providers.openai_compat.httpx.AsyncClient") as mock_http_client,
        patch("providers.openai_compat.AsyncOpenAI") as mock_openai,
    ):
        mock_settings.return_value = _make_mock_settings(
            nvidia_nim_proxy="http://proxy.example:8080"
        )

        provider = get_provider()

        assert isinstance(provider, NvidiaNimProvider)
        mock_http_client.assert_called_once()
        assert mock_http_client.call_args.kwargs["proxy"] == "http://proxy.example:8080"
        assert (
            mock_openai.call_args.kwargs["http_client"] is mock_http_client.return_value
        )


@pytest.mark.asyncio
async def test_get_provider_ignores_non_string_proxy_value():
    """Mock settings without proxy attrs should not fail provider construction."""
    with (
        patch("api.dependencies.get_settings") as mock_settings,
        patch("providers.openai_compat.AsyncOpenAI") as mock_openai,
    ):
        mock_settings.return_value = _make_mock_settings(
            nvidia_nim_proxy=MagicMock(name="proxy")
        )

        provider = get_provider()

        assert isinstance(provider, NvidiaNimProvider)
        assert mock_openai.call_args.kwargs["http_client"] is None


@pytest.mark.asyncio
async def test_get_provider_nvidia_nim_missing_api_key():
    """NVIDIA NIM with empty API key raises HTTPException 503."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(nvidia_nim_api_key="")

        with pytest.raises(HTTPException) as exc_info:
            get_provider()

        assert exc_info.value.status_code == 503
        assert "NVIDIA_NIM_API_KEY" in exc_info.value.detail
        assert "build.nvidia.com" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_provider_nvidia_nim_whitespace_only_api_key():
    """NVIDIA NIM with whitespace-only API key raises HTTPException 503."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(nvidia_nim_api_key="   ")

        with pytest.raises(HTTPException) as exc_info:
            get_provider()

        assert exc_info.value.status_code == 503
        assert "NVIDIA_NIM_API_KEY" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_provider_open_router_missing_api_key():
    """OpenRouter with empty API key raises HTTPException 503."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(
            provider_type="open_router",
            open_router_api_key="",
        )

        with pytest.raises(HTTPException) as exc_info:
            get_provider()

        assert exc_info.value.status_code == 503
        assert "OPENROUTER_API_KEY" in exc_info.value.detail
        assert "openrouter.ai" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_provider_deepseek_missing_api_key():
    """DeepSeek with empty API key raises HTTPException 503."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(
            provider_type="deepseek",
            deepseek_api_key="",
        )

        with pytest.raises(HTTPException) as exc_info:
            get_provider()

        assert exc_info.value.status_code == 503
        assert "DEEPSEEK_API_KEY" in exc_info.value.detail
        assert "platform.deepseek.com" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_provider_wafer_missing_api_key():
    """Wafer with empty API key raises HTTPException 503."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(
            provider_type="wafer",
            wafer_api_key="",
        )

        with pytest.raises(HTTPException) as exc_info:
            get_provider()

        assert exc_info.value.status_code == 503
        assert "WAFER_API_KEY" in exc_info.value.detail
        assert "wafer.ai" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_provider_unknown_type():
    """Unknown ``provider_type`` raises :exc:`~providers.exceptions.UnknownProviderTypeError`."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(provider_type="unknown")

        with pytest.raises(UnknownProviderTypeError, match="Unknown provider_type"):
            get_provider()


@pytest.mark.asyncio
async def test_cleanup_provider_close_raises():
    """cleanup_provider handles close() raising an exception."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings()

        provider = get_provider()
        assert isinstance(provider, NvidiaNimProvider)
        provider._client = AsyncMock()
        provider._client.close = AsyncMock(side_effect=RuntimeError("cleanup failed"))

        # Should propagate the error
        with pytest.raises(RuntimeError, match="cleanup failed"):
            await cleanup_provider()


# --- Provider Registry Tests ---


@pytest.mark.asyncio
async def test_get_provider_for_type_caches():
    """get_provider_for_type returns cached provider on second call."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings()

        p1 = get_provider_for_type("nvidia_nim")
        p2 = get_provider_for_type("nvidia_nim")

        assert p1 is p2
        assert isinstance(p1, NvidiaNimProvider)


@pytest.mark.asyncio
async def test_get_provider_for_type_different_types():
    """get_provider_for_type creates separate providers per type."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings()

        nim = get_provider_for_type("nvidia_nim")
        lmstudio = get_provider_for_type("lmstudio")

        assert isinstance(nim, NvidiaNimProvider)
        assert isinstance(lmstudio, LMStudioProvider)
        assert nim is not lmstudio


@pytest.mark.asyncio
async def test_get_provider_for_type_missing_key_raises_503():
    """get_provider_for_type raises HTTPException 503 for missing API key."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings(open_router_api_key="")

        with pytest.raises(HTTPException) as exc_info:
            get_provider_for_type("open_router")

        assert exc_info.value.status_code == 503
        assert "OPENROUTER_API_KEY" in exc_info.value.detail


@pytest.mark.asyncio
async def test_cleanup_provider_cleans_all():
    """cleanup_provider cleans up all providers in the registry."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings()

        nim = get_provider_for_type("nvidia_nim")
        lmstudio = get_provider_for_type("lmstudio")

        assert isinstance(nim, NvidiaNimProvider)
        assert isinstance(lmstudio, LMStudioProvider)

        nim._client = AsyncMock()
        lmstudio._client = AsyncMock()

        await cleanup_provider()

        nim._client.close.assert_called_once()
        lmstudio._client.close.assert_called_once()


def test_resolve_provider_per_app_uses_separate_registries() -> None:
    """With app set, each app gets its own provider cache (not process _providers)."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings()
        settings = _make_mock_settings()
        app1 = SimpleNamespace(state=State())
        app2 = SimpleNamespace(state=State())
        app1.state.provider_registry = ProviderRegistry()
        app2.state.provider_registry = ProviderRegistry()
        p1 = resolve_provider(
            "nvidia_nim", app=cast(Starlette, app1), settings=settings
        )
        p2 = resolve_provider(
            "nvidia_nim", app=cast(Starlette, app2), settings=settings
        )
        assert isinstance(p1, NvidiaNimProvider)
        assert isinstance(p2, NvidiaNimProvider)
        assert p1 is not p2


def test_resolve_provider_missing_registry_raises_service_unavailable() -> None:
    """HTTP apps must install app.state.provider_registry (e.g. via AppRuntime)."""
    with patch("api.dependencies.get_settings") as mock_settings:
        mock_settings.return_value = _make_mock_settings()
        settings = _make_mock_settings()
        app = SimpleNamespace(state=State())
        assert getattr(app.state, "provider_registry", None) is None
        with pytest.raises(
            ServiceUnavailableError, match="Provider registry is not configured"
        ):
            resolve_provider("nvidia_nim", app=cast(Starlette, app), settings=settings)


def test_resolve_provider_unrelated_value_error_is_not_unknown_provider_log() -> None:
    """Only :exc:`~providers.exceptions.UnknownProviderTypeError` logs unknown provider."""
    import api.dependencies as deps

    with (
        patch.object(deps, "get_settings", return_value=_make_mock_settings()),
        patch.object(
            ProviderRegistry,
            "get",
            side_effect=ValueError("unrelated config"),
        ),
        patch.object(deps.logger, "error") as log_err,
        pytest.raises(ValueError, match="unrelated config"),
    ):
        deps.resolve_provider("nvidia_nim", app=None, settings=_make_mock_settings())
    log_err.assert_not_called()
