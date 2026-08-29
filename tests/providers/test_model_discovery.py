import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

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
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.deepseek import DeepSeekProvider
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.open_router import OpenRouterProvider
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.providers.runtime.model_cache import ProviderModelCache
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    profiled_provider,
)


def _settings(
    *,
    model: str = "nvidia_nim/nim-model",
    model_fable: str | None = None,
    model_opus: str | None = None,
    model_sonnet: str | None = None,
    model_haiku: str | None = None,
    model_fallbacks: tuple[str, ...] | None = None,
    nvidia_nim_api_key: str = "",
    open_router_api_key: str = "",
    deepseek_api_key: str = "",
    wafer_api_key: str = "",
    opencode_api_key: str = "",
    zai_api_key: str = "",
    vertex_project_id: str = "",
    ollama_base_url: str | None = None,
) -> Settings:
    return Settings.model_construct(
        model=model,
        model_fable=model_fable,
        model_opus=model_opus,
        model_sonnet=model_sonnet,
        model_haiku=model_haiku,
        model_fallbacks=model_fallbacks,
        nvidia_nim_api_key=nvidia_nim_api_key,
        open_router_api_key=open_router_api_key,
        deepseek_api_key=deepseek_api_key,
        wafer_api_key=wafer_api_key,
        opencode_api_key=opencode_api_key,
        zai_api_key=zai_api_key,
        vertex_project_id=vertex_project_id,
        log_api_error_tracebacks=False,
        **({"ollama_base_url": ollama_base_url} if ollama_base_url is not None else {}),
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


def _infos(*model_ids: str) -> frozenset[ProviderModelInfo]:
    return frozenset(ProviderModelInfo(model_id) for model_id in model_ids)


def test_provider_catalog_contract_is_metadata_only() -> None:
    assert not hasattr(BaseProvider, "list_model_ids")
    assert getattr(BaseProvider.list_model_infos, "__isabstractmethod__", False)


@pytest.mark.asyncio
async def test_nim_lists_openai_compatible_model_infos() -> None:
    config = make_provider_config(api_key="test-key", base_url=NVIDIA_NIM_DEFAULT_BASE)
    with patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"):
        provider = NvidiaNimProvider(
            config, nim_settings=NimSettings(), admission=immediate_admission()
        )

    with patch.object(
        provider._client.models,
        "list",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(data=[SimpleNamespace(id="nvidia/model")]),
    ):
        assert await provider.list_model_infos() == _infos("nvidia/model")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    [
        profiled_provider(
            "llamacpp",
            make_provider_config(
                api_key="llamacpp", base_url="http://localhost:8080/v1"
            ),
            admission=immediate_admission(),
        ),
        profiled_provider(
            "ollama",
            make_provider_config(api_key="ollama", base_url="http://localhost:11434"),
            admission=immediate_admission(),
        ),
    ],
)
async def test_local_openai_chat_providers_list_model_infos(
    provider: OpenAIChatProvider,
) -> None:
    with patch.object(
        provider._client.models,
        "list",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(data=[SimpleNamespace(id="local/model")]),
    ) as mock_list:
        assert await provider.list_model_infos() == _infos("local/model")

    mock_list.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_deepseek_lists_models_from_root_endpoint() -> None:
    provider = DeepSeekProvider(
        make_provider_config(api_key="deepseek-key", base_url=DEEPSEEK_DEFAULT_BASE),
        admission=immediate_admission(),
    )
    with patch.object(
        provider._client.models,
        "list",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(data=[SimpleNamespace(id="deepseek-chat")]),
    ) as mock_list:
        assert await provider.list_model_infos() == _infos("deepseek-chat")

    mock_list.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_wafer_lists_models_from_default_models_endpoint() -> None:
    provider = profiled_provider(
        "wafer",
        make_provider_config(api_key="wafer-key", base_url=WAFER_DEFAULT_BASE),
        admission=immediate_admission(),
    )
    with patch.object(
        provider._client.models,
        "list",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(data=[SimpleNamespace(id="DeepSeek-V4-Pro")]),
    ) as mock_list:
        assert await provider.list_model_infos() == _infos("DeepSeek-V4-Pro")

    mock_list.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_openrouter_lists_only_tool_capable_models() -> None:
    provider = OpenRouterProvider(
        make_provider_config(
            api_key="open-router-key", base_url=OPENROUTER_DEFAULT_BASE
        ),
        admission=immediate_admission(),
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
        assert await provider.list_model_infos() == frozenset(
            {
                ProviderModelInfo("tool-model", supports_thinking=False),
                ProviderModelInfo("tool-choice-model", supports_thinking=False),
            }
        )

    mock_list.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_openrouter_lists_tool_metadata_with_thinking_support() -> None:
    provider = OpenRouterProvider(
        make_provider_config(
            api_key="open-router-key", base_url=OPENROUTER_DEFAULT_BASE
        ),
        admission=immediate_admission(),
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
        make_provider_config(
            api_key="open-router-key", base_url=OPENROUTER_DEFAULT_BASE
        ),
        admission=immediate_admission(),
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
        assert await provider.list_model_infos() == frozenset()


@pytest.mark.asyncio
async def test_openrouter_model_metadata_rejects_malformed_ids() -> None:
    provider = OpenRouterProvider(
        make_provider_config(
            api_key="open-router-key", base_url=OPENROUTER_DEFAULT_BASE
        ),
        admission=immediate_admission(),
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
        make_provider_config(api_key="llamacpp", base_url="http://localhost:8080/v1"),
        admission=immediate_admission(),
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
        await provider.list_model_infos()


@pytest.mark.asyncio
async def test_model_listing_propagates_upstream_errors() -> None:
    provider = profiled_provider(
        "llamacpp",
        make_provider_config(api_key="llamacpp", base_url="http://localhost:8080/v1"),
        admission=immediate_admission(),
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
        await provider.list_model_infos()


class FakeProvider(BaseProvider):
    def __init__(
        self,
        model_infos: frozenset[ProviderModelInfo] = frozenset(),
        *,
        error: BaseException | None = None,
        started: asyncio.Event | None = None,
        peer_started: asyncio.Event | None = None,
    ):
        super().__init__(
            make_provider_config(api_key="test", base_url="https://test.invalid")
        )
        self._model_infos = model_infos
        self._error = error
        self._started = started
        self._peer_started = peer_started
        self.cleaned = False
        self.model_list_calls = 0

    def preflight_messages(
        self,
        request: Any,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        return None

    def preflight_responses(
        self,
        request: Any,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        return None

    async def cleanup(self) -> None:
        self.cleaned = True

    async def _before_model_list(self) -> None:
        self.model_list_calls += 1
        if self._started is not None:
            self._started.set()
        if self._peer_started is not None:
            await self._peer_started.wait()
        if self._error is not None:
            raise self._error

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        await self._before_model_list()
        return self._model_infos

    async def stream_messages(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        if False:
            yield ""

    async def stream_responses(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        if False:
            yield ""


@pytest.mark.asyncio
async def test_runtime_warm_caches_all_referenced_provider_models() -> None:
    settings = _settings(
        model_fallbacks=("open_router/anthropic/claude-opus",),
        nvidia_nim_api_key="nim-key",
        open_router_api_key="open-router-key",
    )
    nim = FakeProvider(_infos("nim-model"))
    router = FakeProvider(_infos("anthropic/claude-opus"))
    runtime = _manager(
        settings,
        {
            "nvidia_nim": nim,
            "open_router": router,
        },
    )

    result = await runtime.warm_referenced_model_cache()

    assert result.refreshed_provider_ids == ("nvidia_nim", "open_router")
    assert result.failed_provider_ids == ()
    assert runtime.cached_model_ids() == {
        "nvidia_nim": frozenset({"nim-model"}),
        "open_router": frozenset({"anthropic/claude-opus"}),
    }
    assert nim.model_list_calls == 1
    assert router.model_list_calls == 1


@pytest.mark.asyncio
async def test_runtime_warm_treats_model_lists_as_discovery_metadata() -> None:
    settings = _settings(
        model_sonnet="nvidia_nim/nim-model",
        nvidia_nim_api_key="nim-key",
    )
    runtime = _manager(
        settings,
        {"nvidia_nim": FakeProvider(_infos("different-model"))},
    )

    result = await runtime.warm_referenced_model_cache()

    assert result.refreshed_provider_ids == ("nvidia_nim",)
    assert result.failed_provider_ids == ()
    assert runtime.cached_model_ids() == {"nvidia_nim": frozenset({"different-model"})}


@pytest.mark.asyncio
async def test_runtime_warm_reports_query_failures_without_blocking() -> None:
    settings = _settings(
        model_opus="open_router/anthropic/claude-opus",
        nvidia_nim_api_key="nim-key",
        open_router_api_key="open-router-key",
    )
    runtime = _manager(
        settings,
        {
            "nvidia_nim": FakeProvider(_infos("nim-model")),
            "open_router": FakeProvider(
                error=ModelListResponseError("bad model-list shape")
            ),
        },
    )

    with patch(
        "free_claude_code.providers.runtime.discovery.logger.warning"
    ) as warning:
        result = await runtime.warm_referenced_model_cache()

    assert result.refreshed_provider_ids == ("nvidia_nim",)
    assert result.failed_provider_ids == ("open_router",)
    assert runtime.cached_model_ids() == {"nvidia_nim": frozenset({"nim-model"})}
    logged = " ".join(str(arg) for call in warning.call_args_list for arg in call.args)
    assert "open_router" in logged
    assert "malformed model-list response: bad model-list shape" in logged


@pytest.mark.asyncio
async def test_runtime_warm_queries_referenced_providers_concurrently() -> None:
    nim_started = asyncio.Event()
    router_started = asyncio.Event()
    settings = _settings(model_opus="open_router/anthropic/claude-opus")
    runtime = _manager(
        settings,
        {
            "nvidia_nim": FakeProvider(
                _infos("nim-model"),
                started=nim_started,
                peer_started=router_started,
            ),
            "open_router": FakeProvider(
                _infos("anthropic/claude-opus"),
                started=router_started,
                peer_started=nim_started,
            ),
        },
    )

    await asyncio.wait_for(runtime.warm_referenced_model_cache(), timeout=1.0)


def _offline_local_providers() -> dict[str, BaseProvider]:
    """Local providers are always configured; this host is running none of them."""
    return {
        provider_id: FakeProvider(error=RuntimeError("connection refused"))
        for provider_id in ("ollama", "lmstudio", "llamacpp")
    }


@pytest.mark.asyncio
async def test_startup_discovery_queries_each_successful_provider_once() -> None:
    settings = _settings(
        nvidia_nim_api_key="nim-key",
        open_router_api_key="open-router-key",
    )
    nim = FakeProvider(_infos("nim-model"))
    router = FakeProvider(_infos("anthropic/claude-sonnet"))
    runtime = _manager(
        settings,
        {**_offline_local_providers(), "nvidia_nim": nim, "open_router": router},
    )

    await runtime.warm_referenced_model_cache()
    runtime.start_model_list_refresh()
    refresh_task = runtime._refresh_task
    assert refresh_task is not None
    await refresh_task

    assert nim.model_list_calls == 1
    assert router.model_list_calls == 1
    assert runtime.cached_model_ids() == {
        "nvidia_nim": frozenset({"nim-model"}),
        "open_router": frozenset({"anthropic/claude-sonnet"}),
    }


@pytest.mark.asyncio
async def test_failed_startup_warm_remains_eligible_for_background_refresh() -> None:
    settings = _settings(nvidia_nim_api_key="nim-key")
    nim = FakeProvider(error=RuntimeError("upstream unavailable"))
    runtime = _manager(settings, {**_offline_local_providers(), "nvidia_nim": nim})

    warm_result = await runtime.warm_referenced_model_cache()
    runtime.start_model_list_refresh()
    refresh_task = runtime._refresh_task
    assert refresh_task is not None
    await refresh_task

    assert warm_result.failed_provider_ids == ("nvidia_nim",)
    assert nim.model_list_calls == 2
    assert runtime.cached_model_ids() == {}


@pytest.mark.asyncio
async def test_runtime_refresh_model_list_cache_discovers_running_local_providers() -> (
    None
):
    # https://github.com/Alishahryar1/free-claude-code/issues/1296
    # A local provider the customer has not routed to is still queried, so its
    # models can reach the client model picker before they pick one.
    settings = _settings(
        model="lmstudio/local-qwen",
        open_router_api_key="open-router-key",
    )
    runtime = _manager(
        settings,
        {
            "open_router": FakeProvider(_infos("anthropic/claude-sonnet")),
            "lmstudio": FakeProvider(_infos("local-qwen")),
            "ollama": FakeProvider(_infos("llama3.1")),
            "llamacpp": FakeProvider(error=RuntimeError("connection refused")),
        },
    )

    result = await runtime.refresh_model_list_cache()

    assert runtime.cached_model_ids() == {
        "open_router": frozenset({"anthropic/claude-sonnet"}),
        "lmstudio": frozenset({"local-qwen"}),
        "ollama": frozenset({"llama3.1"}),
    }
    assert result.refreshed_provider_ids == ("open_router", "lmstudio", "ollama")
    # llamacpp is configured by default but is not running: absent, not failed.
    assert result.failed_provider_ids == ()


@pytest.mark.asyncio
async def test_runtime_refresh_model_list_cache_reports_a_routed_local_failure() -> (
    None
):
    # A local provider the customer routed to is theirs to fix, so it keeps
    # being reported, unlike one queried only on the chance it is running.
    settings = _settings(model="ollama/llama3.1")
    runtime = _manager(settings, _offline_local_providers())

    result = await runtime.refresh_model_list_cache()

    assert result.failed_provider_ids == ("ollama",)
    assert runtime.cached_model_ids() == {}


@pytest.mark.asyncio
async def test_runtime_refresh_model_list_cache_reports_a_relocated_local_failure() -> (
    None
):
    # Pointing a local provider somewhere other than the shipped default is as
    # deliberate as routing a model to it, so an unreachable host is reported
    # even though no configured model names the provider.
    settings = _settings(ollama_base_url="http://gpu-box:11434")
    runtime = _manager(settings, _offline_local_providers())

    result = await runtime.refresh_model_list_cache()

    assert result.failed_provider_ids == ("ollama",)


@pytest.mark.asyncio
async def test_runtime_refresh_model_list_cache_stays_quiet_for_default_local_hosts() -> (
    None
):
    # The default base URL is present on every install, so it carries no intent
    # and an absent daemon must not be reported as a provider failure.
    settings = _settings()
    runtime = _manager(settings, _offline_local_providers())

    result = await runtime.refresh_model_list_cache()

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
        {
            **_offline_local_providers(),
            "vertex": FakeProvider(_infos("google/gemini-3.5-flash")),
        },
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
        {
            **_offline_local_providers(),
            "nvidia_nim": FakeProvider(error=RuntimeError("upstream down")),
        },
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
    assert cache.cached_model_info(
        "open_router", "reasoning-model"
    ) == ProviderModelInfo("reasoning-model", supports_thinking=True)
    assert cache.cached_model_info("open_router", "plain-model") == ProviderModelInfo(
        "plain-model", supports_thinking=False
    )
    assert cache.cached_prefixed_model_infos() == (
        ProviderModelInfo("open_router/plain-model", supports_thinking=False),
        ProviderModelInfo("open_router/reasoning-model", supports_thinking=True),
    )


def test_runtime_metadata_cache_enforces_replaced_provider_scope() -> None:
    cache = ProviderModelCache({"open_router", "lmstudio"})
    cache.cache_model_infos("open_router", _infos("old-model"))
    cache.cache_model_infos("lmstudio", _infos("local-model"))

    cache.set_available_providers({"deepseek", "lmstudio"})
    cache.cache_model_infos("open_router", _infos("late-old-model"))
    cache.cache_model_infos("deepseek", _infos("new-model"))

    assert cache.cached_model_ids() == {
        "deepseek": frozenset({"new-model"}),
        "lmstudio": frozenset({"local-model"}),
    }


def test_runtime_metadata_cache_keeps_unknown_thinking_support() -> None:
    cache = ProviderModelCache()
    cache.cache_model_infos("open_router", _infos("plain-model"))

    assert cache.cached_model_ids() == {"open_router": frozenset({"plain-model"})}
    assert cache.cached_model_info("open_router", "plain-model") == ProviderModelInfo(
        "plain-model"
    )
    assert cache.cached_prefixed_model_infos() == (
        ProviderModelInfo("open_router/plain-model", supports_thinking=None),
    )


def test_runtime_cached_prefixed_model_infos_are_deterministic() -> None:
    cache = ProviderModelCache()
    cache.cache_model_infos("deepseek", _infos("deepseek-chat"))
    cache.cache_model_infos("open_router", _infos("z-model", "a-model"))

    assert cache.cached_prefixed_model_infos() == (
        ProviderModelInfo("open_router/a-model"),
        ProviderModelInfo("open_router/z-model"),
        ProviderModelInfo("deepseek/deepseek-chat"),
    )
