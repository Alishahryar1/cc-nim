import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.nim import NimSettings
from free_claude_code.config.provider_catalog import (
    DEEPSEEK_DEFAULT_BASE,
    NVIDIA_NIM_DEFAULT_BASE,
    OPENROUTER_DEFAULT_BASE,
    WAFER_DEFAULT_BASE,
)
from free_claude_code.config.settings import Settings
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.deepseek import DeepSeekProvider
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.open_router import OpenRouterProvider
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.providers.runtime.model_cache import ProviderModelCache
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager
from tests.providers.support import passthrough_rate_limiter, profiled_provider


def _settings(
    *,
    model: str = "nvidia_nim/nim-model",
    model_fable: str | None = None,
    model_opus: str | None = None,
    model_sonnet: str | None = None,
    model_haiku: str | None = None,
    nvidia_nim_api_key: str = "",
    open_router_api_key: str = "",
    deepseek_api_key: str = "",
    wafer_api_key: str = "",
    opencode_api_key: str = "",
    zai_api_key: str = "",
    vertex_project_id: str = "",
) -> Settings:
    return Settings.model_construct(
        model=model,
        model_fable=model_fable,
        model_opus=model_opus,
        model_sonnet=model_sonnet,
        model_haiku=model_haiku,
        nvidia_nim_api_key=nvidia_nim_api_key,
        open_router_api_key=open_router_api_key,
        deepseek_api_key=deepseek_api_key,
        wafer_api_key=wafer_api_key,
        opencode_api_key=opencode_api_key,
        zai_api_key=zai_api_key,
        vertex_project_id=vertex_project_id,
        log_api_error_tracebacks=False,
    )


def _manager(
    settings: Settings,
    providers: dict[str, BaseProvider] | None = None,
) -> ProviderRuntimeManager:
    providers = providers or {}
    return ProviderRuntimeManager(
        settings,
        runtime_factory=lambda snapshot: ProviderRuntime(snapshot, dict(providers)),
    )


@pytest.mark.asyncio
async def test_nim_lists_openai_compatible_model_ids() -> None:
    config = ProviderConfig(api_key="test-key", base_url=NVIDIA_NIM_DEFAULT_BASE)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = NvidiaNimProvider(
            config, nim_settings=NimSettings(), rate_limiter=passthrough_rate_limiter()
        )

    with patch.object(
        provider._client.models,
        "list",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(data=[SimpleNamespace(id="nvidia/model")]),
    ):
        assert await provider.list_model_ids() == frozenset({"nvidia/model"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    [
        profiled_provider(
            "llamacpp",
            ProviderConfig(api_key="llamacpp", base_url="http://localhost:8080/v1"),
            rate_limiter=passthrough_rate_limiter(),
        ),
        profiled_provider(
            "ollama",
            ProviderConfig(api_key="ollama", base_url="http://localhost:11434"),
            rate_limiter=passthrough_rate_limiter(),
        ),
    ],
)
async def test_local_openai_chat_providers_list_model_ids(
    provider: OpenAIChatProvider,
) -> None:
    with patch.object(
        provider._client.models,
        "list",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(data=[SimpleNamespace(id="local/model")]),
    ) as mock_list:
        assert await provider.list_model_ids() == frozenset({"local/model"})

    mock_list.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_deepseek_lists_models_from_root_endpoint() -> None:
    provider = DeepSeekProvider(
        ProviderConfig(api_key="deepseek-key", base_url=DEEPSEEK_DEFAULT_BASE),
        rate_limiter=passthrough_rate_limiter(),
    )
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
    provider = profiled_provider(
        "wafer",
        ProviderConfig(api_key="wafer-key", base_url=WAFER_DEFAULT_BASE),
        rate_limiter=passthrough_rate_limiter(),
    )
    with patch.object(
        provider._client.models,
        "list",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(data=[SimpleNamespace(id="DeepSeek-V4-Pro")]),
    ) as mock_list:
        assert await provider.list_model_ids() == frozenset({"DeepSeek-V4-Pro"})

    mock_list.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_openrouter_lists_only_tool_capable_models() -> None:
    provider = OpenRouterProvider(
        ProviderConfig(api_key="open-router-key", base_url=OPENROUTER_DEFAULT_BASE),
        rate_limiter=passthrough_rate_limiter(),
    )
    with patch.object(
        provider._client.models,
        "list",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="tool-model",
                    supported_parameters=["tools", "max_tokens"],
                ),
                SimpleNamespace(
                    id="tool-choice-model",
                    supported_parameters=["tool_choice"],
                ),
                SimpleNamespace(
                    id="chat-only",
                    supported_parameters=["max_tokens", "temperature"],
                ),
                SimpleNamespace(id="missing-metadata", supported_parameters=None),
            ]
        ),
    ) as mock_list:
        assert await provider.list_model_ids() == frozenset(
            {"tool-model", "tool-choice-model"}
        )

    mock_list.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_openrouter_lists_tool_metadata_with_thinking_support() -> None:
    provider = OpenRouterProvider(
        ProviderConfig(api_key="open-router-key", base_url=OPENROUTER_DEFAULT_BASE),
        rate_limiter=passthrough_rate_limiter(),
    )
    with patch.object(
        provider._client.models,
        "list",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(
                    id="reasoning-tool-model",
                    supported_parameters=[
                        "tools",
                        "reasoning",
                        "include_reasoning",
                    ],
                ),
                SimpleNamespace(
                    id="plain-tool-model",
                    supported_parameters=["tool_choice", "include_reasoning"],
                ),
                SimpleNamespace(
                    id="chat-only",
                    supported_parameters=["reasoning", "max_tokens"],
                ),
            ]
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
    provider = OpenRouterProvider(
        ProviderConfig(api_key="open-router-key", base_url=OPENROUTER_DEFAULT_BASE),
        rate_limiter=passthrough_rate_limiter(),
    )
    with patch.object(
        provider._client.models,
        "list",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(id="chat-only", supported_parameters=["max_tokens"]),
                SimpleNamespace(id="missing-metadata", supported_parameters=None),
            ]
        ),
    ):
        assert await provider.list_model_ids() == frozenset()


@pytest.mark.asyncio
async def test_openrouter_model_metadata_rejects_malformed_ids() -> None:
    provider = OpenRouterProvider(
        ProviderConfig(api_key="open-router-key", base_url=OPENROUTER_DEFAULT_BASE),
        rate_limiter=passthrough_rate_limiter(),
    )
    with (
        patch.object(
            provider._client.models,
            "list",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(
                data=[SimpleNamespace(supported_parameters=["tools", "reasoning"])]
            ),
        ),
        pytest.raises(ModelListResponseError, match="malformed"),
    ):
        await provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_listing_rejects_malformed_payload() -> None:
    provider = profiled_provider(
        "llamacpp",
        ProviderConfig(api_key="llamacpp", base_url="http://localhost:8080/v1"),
        rate_limiter=passthrough_rate_limiter(),
    )
    with (
        patch.object(
            provider._client.models,
            "list",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(data=[SimpleNamespace()]),
        ),
        pytest.raises(ModelListResponseError, match="malformed"),
    ):
        await provider.list_model_ids()


@pytest.mark.asyncio
async def test_model_listing_propagates_upstream_errors() -> None:
    provider = profiled_provider(
        "llamacpp",
        ProviderConfig(api_key="llamacpp", base_url="http://localhost:8080/v1"),
        rate_limiter=passthrough_rate_limiter(),
    )
    with (
        patch.object(
            provider._client.models,
            "list",
            new_callable=AsyncMock,
            side_effect=RuntimeError("upstream unavailable"),
        ),
        pytest.raises(RuntimeError, match="upstream unavailable"),
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
        super().__init__(
            ProviderConfig(api_key="test", base_url="https://test.invalid")
        )
        self._model_ids = model_ids or frozenset()
        self._model_infos = model_infos
        self._error = error
        self._started = started
        self._peer_started = peer_started
        self.cleaned = False

    def preflight_stream(
        self,
        request: Any,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        return None

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
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        if False:
            yield ""


@pytest.mark.asyncio
async def test_runtime_validation_succeeds_for_all_configured_models() -> None:
    settings = _settings(
        model_opus="open_router/anthropic/claude-opus",
        nvidia_nim_api_key="nim-key",
        open_router_api_key="open-router-key",
    )
    runtime = _manager(
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
    runtime = _manager(
        settings,
        {"nvidia_nim": FakeProvider(frozenset({"different-model"}))},
    )

    with pytest.raises(ApplicationUnavailableError) as exc_info:
        await runtime.validate_configured_models()

    message = exc_info.value.message
    assert "sources=MODEL,MODEL_SONNET" in message
    assert "provider=nvidia_nim" in message
    assert "model=nim-model" in message
    assert "problem=missing model" in message


@pytest.mark.asyncio
async def test_runtime_validation_aggregates_multiple_failures() -> None:
    settings = _settings(model_opus="open_router/anthropic/claude-opus")
    runtime = _manager(
        settings,
        {
            "nvidia_nim": FakeProvider(frozenset({"different-model"})),
            "open_router": FakeProvider(
                error=ModelListResponseError("bad model-list shape")
            ),
        },
    )

    with pytest.raises(ApplicationUnavailableError) as exc_info:
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
    runtime = _manager(
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
    runtime = _manager(
        settings,
        {
            "open_router": FakeProvider(frozenset({"anthropic/claude-sonnet"})),
            "lmstudio": FakeProvider(frozenset({"local-qwen"})),
            "ollama": FakeProvider(frozenset({"llama3.1"})),
        },
    )

    result = await runtime.refresh_model_list_cache()

    assert runtime.cached_model_ids() == {
        "open_router": frozenset({"anthropic/claude-sonnet"}),
        "lmstudio": frozenset({"local-qwen"}),
    }
    assert result.refreshed_provider_ids == ("open_router", "lmstudio")
    assert result.failed_provider_ids == ()


@pytest.mark.asyncio
async def test_runtime_refresh_model_list_cache_treats_vertex_project_as_configuration() -> (
    None
):
    settings = _settings(
        model="nvidia_nim/nim-model",
        vertex_project_id="vertex-project",
    )
    runtime = _manager(
        settings,
        {"vertex": FakeProvider(frozenset({"google/gemini-3.5-flash"}))},
    )

    result = await runtime.refresh_model_list_cache()

    assert runtime.cached_model_ids() == {
        "vertex": frozenset({"google/gemini-3.5-flash"})
    }
    assert result.refreshed_provider_ids == ("vertex",)
    assert result.failed_provider_ids == ()


@pytest.mark.asyncio
async def test_runtime_refresh_model_list_cache_keeps_prior_cache_on_failure() -> None:
    settings = _settings(
        model="nvidia_nim/cached-model",
        nvidia_nim_api_key="nim-key",
    )
    runtime = _manager(
        settings,
        {"nvidia_nim": FakeProvider(error=RuntimeError("upstream down"))},
    )
    runtime.cache_model_infos(
        "nvidia_nim",
        {ProviderModelInfo("cached-model")},
    )

    result = await runtime.refresh_model_list_cache()

    assert runtime.cached_model_ids() == {"nvidia_nim": frozenset({"cached-model"})}
    assert result.refreshed_provider_ids == ()
    assert result.failed_provider_ids == ("nvidia_nim",)


def test_runtime_metadata_cache_exposes_ids_and_prefixed_infos() -> None:
    cache = ProviderModelCache()
    cache.cache_model_infos(
        "open_router",
        {
            ProviderModelInfo("reasoning-model", supports_thinking=True),
            ProviderModelInfo("plain-model", supports_thinking=False),
        },
    )

    assert cache.cached_model_ids() == {
        "open_router": frozenset({"reasoning-model", "plain-model"})
    }
    assert (
        cache.cached_model_supports_thinking("open_router", "reasoning-model") is True
    )
    assert cache.cached_model_supports_thinking("open_router", "plain-model") is False
    assert cache.cached_prefixed_model_infos() == (
        ProviderModelInfo("open_router/plain-model", supports_thinking=False),
        ProviderModelInfo("open_router/reasoning-model", supports_thinking=True),
    )


def test_runtime_metadata_cache_enforces_replaced_provider_scope() -> None:
    cache = ProviderModelCache({"open_router", "lmstudio"})
    cache.cache_model_ids("open_router", {"old-model"})
    cache.cache_model_ids("lmstudio", {"local-model"})

    cache.set_available_providers({"deepseek", "lmstudio"})
    cache.cache_model_ids("open_router", {"late-old-model"})
    cache.cache_model_ids("deepseek", {"new-model"})

    assert cache.cached_model_ids() == {
        "deepseek": frozenset({"new-model"}),
        "lmstudio": frozenset({"local-model"}),
    }


def test_runtime_model_id_cache_keeps_unknown_thinking_support() -> None:
    cache = ProviderModelCache()
    cache.cache_model_ids("open_router", {"plain-model"})

    assert cache.cached_model_ids() == {"open_router": frozenset({"plain-model"})}
    assert cache.cached_model_supports_thinking("open_router", "plain-model") is None
    assert cache.cached_prefixed_model_infos() == (
        ProviderModelInfo("open_router/plain-model", supports_thinking=None),
    )


def test_runtime_cached_prefixed_model_refs_are_deterministic() -> None:
    cache = ProviderModelCache()
    cache.cache_model_ids("deepseek", {"deepseek-chat"})
    cache.cache_model_ids("open_router", {"z-model", "a-model"})

    assert cache.cached_prefixed_model_refs() == (
        "open_router/a-model",
        "open_router/z-model",
        "deepseek/deepseek-chat",
    )
