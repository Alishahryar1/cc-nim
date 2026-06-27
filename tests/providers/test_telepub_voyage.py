"""Tests for Telepub Voyage OpenAI-compatible provider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import Settings
from providers.base import ProviderConfig
from providers.registry import ProviderRegistry
from providers.telepub_voyage import TelepubVoyageProvider


def _minimal_config() -> ProviderConfig:
    return ProviderConfig(api_key="test-key")


@pytest.mark.asyncio
async def test_list_model_ids_uses_static_when_models_list_fails() -> None:
    with patch(
        "providers.transports.openai_chat.transport.AsyncOpenAI"
    ) as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.models.list = AsyncMock(side_effect=OSError("unreachable"))
        mock_client.chat.completions.create = AsyncMock()
        mock_client.close = AsyncMock()
        provider = TelepubVoyageProvider(
            _minimal_config(),
            static_model_ids=frozenset({"alpha", "beta"}),
        )
        assert await provider.list_model_ids() == frozenset({"alpha", "beta"})
        await provider.cleanup()


@pytest.mark.asyncio
async def test_list_model_ids_merges_static_with_openai_models_response() -> None:
    with patch(
        "providers.transports.openai_chat.transport.AsyncOpenAI"
    ) as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.models.list = AsyncMock(
            return_value=SimpleNamespace(
                data=[
                    {"id": "upstream-a"},
                    {"id": "upstream-b"},
                ]
            )
        )
        mock_client.chat.completions.create = AsyncMock()
        mock_client.close = AsyncMock()
        provider = TelepubVoyageProvider(
            _minimal_config(),
            static_model_ids=frozenset({"upstream-a", "only-static"}),
        )
        assert await provider.list_model_ids() == frozenset(
            {"upstream-a", "upstream-b", "only-static"}
        )
        await provider.cleanup()


@pytest.mark.asyncio
async def test_registry_validates_multiple_telepub_models_from_static_list() -> None:
    """MODEL / MODEL_OPUS may use different telepub_voyage ids when all are in TELEPUB_VOYAGE_MODELS."""
    registry = ProviderRegistry()
    settings = Settings.model_construct(
        model="telepub_voyage/first-model",
        model_opus="telepub_voyage/second-model",
        model_sonnet=None,
        model_haiku=None,
        telepub_voyage_api_key="secret",
        telepub_voyage_models="first-model,second-model,third-model",
        telepub_voyage_base_url="https://example.invalid/voyage/api",
        telepub_voyage_proxy="",
        anthropic_auth_token="",
        log_api_error_tracebacks=False,
    )
    with patch(
        "providers.transports.openai_chat.transport.AsyncOpenAI"
    ) as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.models.list = AsyncMock(side_effect=OSError("no models endpoint"))
        mock_client.chat.completions.create = AsyncMock()
        mock_client.close = AsyncMock()
        await registry.validate_configured_models(settings)

    cached = registry.cached_model_ids().get("telepub_voyage", frozenset())
    assert cached == frozenset({"first-model", "second-model", "third-model"})
