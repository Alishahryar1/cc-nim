import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from config.nim import NimSettings
from config.settings import Settings
from providers.base import BaseProvider, ProviderConfig
from providers.deepseek import DeepSeekProvider
from providers.exceptions import ModelListResponseError, ServiceUnavailableError
from providers.llamacpp import LlamaCppProvider
from providers.model_listing import ProviderModelInfo
from providers.nvidia_nim import NvidiaNimProvider
from providers.ollama import OllamaProvider
from providers.open_router import OpenRouterProvider
from providers.runtime import ProviderRuntime
from providers.wafer import WaferProvider


def _settings(
    *,
    model: str = "nvidia_nim/nim-model",
    model_opus: str | None = None,
    model_sonnet: str | None = None,
    model_haiku: str | None = None,
    nvidia_nim_api_key: str = "",
    open_router_api_key: str = "",
    deepseek_api_key: str = "",
    wafer_api_key: str = "",
    opencode_api_key: str = "",
    zai_api_key: str = "",
) -> Settings:
    return Settings.model_construct(
        model=model,
        model_opus=model_opus,
        model_sonnet=model_sonnet,
        model_haiku=model_haiku,
        nvidia_nim_api_key=nvidia_nim_api_key,
        open_router_api_key=open_router_api_key,
        deepseek_api_key=deepseek_api_key,
        wafer_api_key=wafer_api_key,
        opencode_api_key=opencode_api_key,
        zai_api_key=zai_api_key,
        log_api_error_tracebacks=False,
    )


def _response(status_code: int, payload: object) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("GET", "https://example.test/models"),
    )


@pytest.mark.asyncio
async def test_nim_lists_openai_compatible_model_ids() -> None:
    config = ProviderConfig(api_key="test-key")
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI"):
        provider = NvidiaNimProvider(config, nim_settings=NimSettings())

    with patch.object(
        provider._client.models,
        "list",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(data=[SimpleNamespace(id="nvidia/model")]),
    ):
        assert await provider.list_model_ids() == frozenset({"nvidia/model"})


@pytest.mark.asyncio
async def test_native_anthropic_messages_provider_lists_model_ids() -> None:
    provider = LlamaCppProvider(
        ProviderConfig(api_key="llamacpp", base_url="http://localhost:8080/v1")
    )
    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=_response(200, {"data": [{"id": "local/model"}]}),
    ) as mock_get:
        assert await provider.list_model_ids() == frozenset({"local/model"})

    mock_get.assert_awaited_once_with("/models", headers={})


@pytest.mark.asyncio
async def test_deepseek_lists_models_from_root_endpoint() -> None:
    provider = DeepSeekProvider(ProviderConfig(api_key="deepseek-key"))
    with patch.object(
        provider._client.models,
        "list",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(data=[SimpleNamespace(id="deepseek-chat")]),
    ) as mock_list:
        assert await provider.list_model_ids() == frozenset({"deepseek-chat"})

    mock_list.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_wafer_lists_models_from_default_models_endpoint() -> None:
    provider = WaferProvider(ProviderConfig(api_key="wafer-key"))
    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=_response(200, {"data": [{"id": "DeepSeek-V4-Pro"}]}),
    ) as mock_get:
        assert await provider.list_model_ids() == frozenset({"DeepSeek-V4-Pro"})

    mock_get.assert_awaited_once_with(
        "/models", headers={"Authorization": "Bearer wafer-key"}
    )


@pytest.mark.asyncio
async def test_openrouter_lists_only_tool_capable_models() -> None:
    provider = OpenRouterProvider(ProviderConfig(api_key="open-router-key"))
    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=_response(
            200,
            {
                "data": [
                    {
                        "id": "tool-model",
                        "supported_parameters": ["tools", "max_tokens"],
                    },
                    {
                        "id": "tool-choice-model",
                        "supported_parameters": ["tool_choice"],
                    },
                    {
                        "id": "chat-only",
                        "supported_parameters": ["max_tokens", "temperature"],
                    },
                    {"id": "missing-metadata"},
                ]
            },
        ),
    ) as mock_get:
        assert await provider.list_model_ids() == frozenset(
            {"tool-model", "tool-choice-model"}
        )

    mock_get.assert_awaited_once_with(
        "/models", headers={"Authorization": "Bearer open-router-key"}
    )


@pytest.mark.asyncio
async def test_openrouter_lists_tool_metadata_with_thinking_support() -> None:
    provider = OpenRouterProvider(ProviderConfig(api_key="open-router-key"))
    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=_response(
            200,
            {
                "data": [
                    {
                        "id": "reasoning-tool-model",
                        "supported_parameters": [
                            "tools",
                            "reasoning",
                            "include_reasoning",
                        ],
                    },
                    {
                        "id": "plain-tool-model",
                        "supported_parameters": ["tool_choice", "include_reasoning"],
                    },
                    {
                        "id": "chat-only",
                        "supported_parameters": ["reasoning", "max_tokens"],
                    },
                ]
            },
        ),
    ):
        infos = await provider.list_model_infos()

    assert infos == frozenset(
        {
            ProviderModelInfo("reasoning-tool-model", supports_thinking=True),
            ProviderModelInfo("plain-tool-model", supports_thinking=False),
        }
    )


@pytest.mark.asyncio
async def test_openrouter_lists_empty_set_when_no_tool_capable_models() -> None:
    provider = OpenRouterProvider(ProviderConfig(api_key="open-router-key"))
    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=_response(
            200,
            {
                "data": [
                    {"id": "chat-only", "supported_parameters": ["max_tokens"]},
                    {"id": "missing-metadata"},
                ]
            },
        ),
    ):
        assert await provider.list_model_ids() == frozenset()


@pytest.mark.asyncio
async def test_openrouter_model_metadata_rejects_malformed_ids() -> None:
    provider = OpenRouterProvider(ProviderConfig(api_key="open-router-key"))
    with (
        patch.object(
            provider._client,
            "get",
            new_callable=AsyncMock,
            return_value=_response(
                200,
                {"data": [{"supported_parameters": ["tools", "reasoning"]}]},
            ),
        ),
        pytest.raises(ModelListResponseError, match="malformed"),
    ):
        await provider.list_model_infos()


@pytest.mark.asyncio
async def test_ollama_lists_native_tag_model_ids() -> None:
    provider = OllamaProvider(
        ProviderConfig(api_key="ollama", base_url="http://localhost:11434")
    )
    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=_response(
            200,
            {
                "models": [
                    {"name": "llama3.1:latest", "model": "llama3.1:latest"},
                    {"name": "qwen3"},
                ]
            },
        ),
    ) as mock_get:
        assert await provider.list_model_ids() == frozenset(
            {"llama3.1:latest", "qwen3"}
        )

    mock_get.assert_awaited_once_with("http://localhost:11434/api/tags")


@pytest.mark.asyncio
async def test_model_listing_rejects_malformed_payload() -> None:
    provider = LlamaCppProvider(
        ProviderConfig(api_key="llamacpp", base_url="http://localhost:8080/v1")
    )
    with (
        patch.object(
            provider._client,
            "get",
            new_callable=AsyncMock,
            return_value=_response(200, {"data": [{}]}),
        ),
        pytest.raises(ModelListResponseError, match="malformed"),
    ):
        await provider.list_model_ids()


@pytest.mark.asyncio
async def test_model_listing_raises_http_status_errors() -> None:
    provider = LlamaCppProvider(
        ProviderConfig(api_key="llamacpp", base_url="http://localhost:8080/v1")
    )
    with (
        patch.object(
            provider._client,
            "get",
            new_callable=AsyncMock,
            return_value=_response(503, {"error": "down"}),
        ),
        pytest.raises(httpx.HTTPStatusError),
    ):
        await provider.list_model_ids()


class FakeProvider(BaseProvider):
    def __init__(
        self,
        model_ids: frozenset[str] | None = None,
        *,
        model_infos: frozenset[ProviderModelInfo] | None = None,
        error: BaseException | None = None,
        started: asyncio.Event | None = None,
        peer_started: asyncio.Event | None = None,
    ):
        super().__init__(ProviderConfig(api_key="test"))
        self._model_ids = model_ids or frozenset()
        self._model_infos = model_infos
        self._error = error
        self._started = started
        self._peer_started = peer_started
        self.cleaned = False

    async def cleanup(self) -> None:
        self.cleaned = True

    async def _before_model_list(self) -> None:
        if self._started is not None:
            self._started.set()
        if self._peer_started is not None:
            await self._peer_started.wait()
        if self._error is not None:
            raise self._error

    async def list_model_ids(self) -> frozenset[str]:
        await self._before_model_list()
        if self._model_infos is not None:
            return frozenset(info.model_id for info in self._model_infos)
        return self._model_ids

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        await self._before_model_list()
        if self._model_infos is not None:
            return self._model_infos
        return frozenset(ProviderModelInfo(model_id) for model_id in self._model_ids)

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        if False:
            yield ""


@pytest.mark.asyncio
async def test_runtime_validation_succeeds_for_all_configured_models() -> None:
    settings = _settings(model_opus="open_router/anthropic/claude-opus")
    runtime = ProviderRuntime(
        settings,
        {
            "nvidia_nim": FakeProvider(frozenset({"nim-model"})),
            "open_router": FakeProvider(frozenset({"anthropic/claude-opus"})),
        },
    )

    await runtime.validate_configured_models()

    assert runtime.cached_model_ids() == {
        "nvidia_nim": frozenset({"nim-model"}),
        "open_router": frozenset({"anthropic/claude-opus"}),
    }


@pytest.mark.asyncio
async def test_runtime_validation_reports_missing_model_with_sources() -> None:
    settings = _settings(model_sonnet="nvidia_nim/nim-model")
    runtime = ProviderRuntime(
        settings,
        {"nvidia_nim": FakeProvider(frozenset({"different-model"}))},
    )

    with pytest.raises(ServiceUnavailableError) as exc_info:
        await runtime.validate_configured_models()

    message = exc_info.value.message
    assert "sources=MODEL,MODEL_SONNET" in message
    assert "provider=nvidia_nim" in message
    assert "model=nim-model" in message
    assert "problem=missing model" in message


@pytest.mark.asyncio
async def test_runtime_validation_aggregates_multiple_failures() -> None:
    settings = _settings(model_opus="open_router/anthropic/claude-opus")
    runtime = ProviderRuntime(
        settings,
        {
            "nvidia_nim": FakeProvider(frozenset({"different-model"})),
            "open_router": FakeProvider(
                error=ModelListResponseError("bad model-list shape")
            ),
        },
    )

    with pytest.raises(ServiceUnavailableError) as exc_info:
        await runtime.validate_configured_models()

    message = exc_info.value.message
    assert "sources=MODEL provider=nvidia_nim model=nim-model" in message
    assert "problem=missing model" in message
    assert "sources=MODEL_OPUS provider=open_router model=anthropic/claude-opus" in (
        message
    )
    assert "problem=malformed model-list response" in message


@pytest.mark.asyncio
async def test_runtime_validation_queries_providers_concurrently() -> None:
    nim_started = asyncio.Event()
    router_started = asyncio.Event()
    settings = _settings(model_opus="open_router/anthropic/claude-opus")
    runtime = ProviderRuntime(
        settings,
        {
            "nvidia_nim": FakeProvider(
                frozenset({"nim-model"}),
                started=nim_started,
                peer_started=router_started,
            ),
            "open_router": FakeProvider(
                frozenset({"anthropic/claude-opus"}),
                started=router_started,
                peer_started=nim_started,
            ),
        },
    )

    await asyncio.wait_for(runtime.validate_configured_models(), timeout=1.0)


@pytest.mark.asyncio
async def test_runtime_refresh_model_list_cache_uses_configured_remote_keys_and_referenced_local() -> (
    None
):
    settings = _settings(
        model="lmstudio/local-qwen",
        open_router_api_key="open-router-key",
    )
    runtime = ProviderRuntime(
        settings,
        {
            "open_router": FakeProvider(frozenset({"anthropic/claude-sonnet"})),
            "lmstudio": FakeProvider(frozenset({"local-qwen"})),
            "ollama": FakeProvider(frozenset({"llama3.1"})),
        },
    )

    await runtime.refresh_model_list_cache()

    assert runtime.cached_model_ids() == {
        "open_router": frozenset({"anthropic/claude-sonnet"}),
        "lmstudio": frozenset({"local-qwen"}),
    }


@pytest.mark.asyncio
async def test_runtime_refresh_model_list_cache_keeps_prior_cache_on_failure() -> None:
    settings = _settings(
        model="nvidia_nim/cached-model",
        nvidia_nim_api_key="nim-key",
    )
    runtime = ProviderRuntime(
        settings,
        {"nvidia_nim": FakeProvider(error=RuntimeError("upstream down"))},
    )
    runtime.cache_model_ids("nvidia_nim", {"cached-model"})

    await runtime.refresh_model_list_cache()

    assert runtime.cached_model_ids() == {"nvidia_nim": frozenset({"cached-model"})}


def test_runtime_metadata_cache_exposes_ids_and_prefixed_infos() -> None:
    runtime = ProviderRuntime(_settings())
    runtime.cache_model_infos(
        "open_router",
        {
            ProviderModelInfo("reasoning-model", supports_thinking=True),
            ProviderModelInfo("plain-model", supports_thinking=False),
        },
    )

    assert runtime.cached_model_ids() == {
        "open_router": frozenset({"reasoning-model", "plain-model"})
    }
    assert (
        runtime.cached_model_supports_thinking("open_router", "reasoning-model") is True
    )
    assert runtime.cached_model_supports_thinking("open_router", "plain-model") is False
    assert runtime.cached_prefixed_model_infos() == (
        ProviderModelInfo("open_router/plain-model", supports_thinking=False),
        ProviderModelInfo("open_router/reasoning-model", supports_thinking=True),
    )


def test_runtime_model_id_cache_keeps_unknown_thinking_support() -> None:
    runtime = ProviderRuntime(_settings())
    runtime.cache_model_ids("open_router", {"plain-model"})

    assert runtime.cached_model_ids() == {"open_router": frozenset({"plain-model"})}
    assert runtime.cached_model_supports_thinking("open_router", "plain-model") is None
    assert runtime.cached_prefixed_model_infos() == (
        ProviderModelInfo("open_router/plain-model", supports_thinking=None),
    )


def test_runtime_cached_prefixed_model_refs_are_deterministic() -> None:
    runtime = ProviderRuntime(_settings())
    runtime.cache_model_ids("deepseek", {"deepseek-chat"})
    runtime.cache_model_ids("open_router", {"z-model", "a-model"})

    assert runtime.cached_prefixed_model_refs() == (
        "open_router/a-model",
        "open_router/z-model",
        "deepseek/deepseek-chat",
    )
