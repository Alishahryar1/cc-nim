"""Tests for GitHub Models OpenAI-compatible provider."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from api.models.anthropic import Message, MessagesRequest
from core.anthropic.stream_contracts import parse_sse_text
from providers.base import ProviderConfig
from providers.exceptions import ModelListResponseError
from providers.github_models import GITHUB_MODELS_DEFAULT_BASE, GitHubModelsProvider
from providers.github_models.client import GITHUB_MODELS_CATALOG_URL


@pytest.fixture
def github_models_config() -> ProviderConfig:
    return ProviderConfig(
        api_key="test-github-models-token",
        base_url=GITHUB_MODELS_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    @asynccontextmanager
    async def _slot():
        yield

    with patch("providers.transports.openai_chat.transport.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def github_models_provider(
    github_models_config: ProviderConfig,
) -> GitHubModelsProvider:
    return GitHubModelsProvider(github_models_config)


def _request(model: str = "openai/gpt-4.1") -> MessagesRequest:
    return MessagesRequest(
        model=model,
        max_tokens=100,
        messages=[Message(role="user", content="hi")],
    )


def _chunk(delta: SimpleNamespace, *, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
        usage=SimpleNamespace(completion_tokens=5, prompt_tokens=8),
    )


async def _stream(*chunks: SimpleNamespace) -> AsyncIterator[SimpleNamespace]:
    for chunk in chunks:
        yield chunk


def _catalog_response(payload: object) -> httpx.Response:
    return httpx.Response(
        200,
        json=payload,
        request=httpx.Request("GET", GITHUB_MODELS_CATALOG_URL),
    )


def test_default_base_url_constant() -> None:
    assert GITHUB_MODELS_DEFAULT_BASE == "https://models.github.ai/inference"


def test_init_uses_default_base_url_api_key_and_github_headers(
    github_models_config: ProviderConfig,
) -> None:
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI") as mock_openai:
        provider = GitHubModelsProvider(github_models_config)

    assert provider._api_key == "test-github-models-token"
    assert provider._base_url == GITHUB_MODELS_DEFAULT_BASE
    assert provider._catalog_url == GITHUB_MODELS_CATALOG_URL
    assert mock_openai.call_args.kwargs["base_url"] == GITHUB_MODELS_DEFAULT_BASE
    assert mock_openai.call_args.kwargs["api_key"] == "test-github-models-token"
    assert mock_openai.call_args.kwargs["default_headers"] == {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2026-03-10",
    }


def test_init_strips_trailing_slash(github_models_config: ProviderConfig) -> None:
    config = github_models_config.model_copy(
        update={"base_url": f"{GITHUB_MODELS_DEFAULT_BASE}/"}
    )

    with patch("providers.transports.openai_chat.transport.AsyncOpenAI"):
        provider = GitHubModelsProvider(config)

    assert provider._base_url == GITHUB_MODELS_DEFAULT_BASE


def test_model_list_headers_use_bearer_auth(
    github_models_provider: GitHubModelsProvider,
) -> None:
    assert github_models_provider._model_list_headers() == {
        "Accept": "application/vnd.github+json",
        "Authorization": "Bearer test-github-models-token",
        "X-GitHub-Api-Version": "2026-03-10",
    }


def test_build_request_body_uses_shared_openai_chat_policy(
    github_models_provider: GitHubModelsProvider,
) -> None:
    request = _request()

    body = github_models_provider._build_request_body(request, thinking_enabled=True)

    assert body["model"] == "openai/gpt-4.1"
    assert body["max_tokens"] == 100
    assert "extra_body" not in body


@pytest.mark.asyncio
async def test_lists_stream_tool_capable_models_only(
    github_models_provider: GitHubModelsProvider,
) -> None:
    with patch.object(
        github_models_provider._model_list_client,
        "get",
        new_callable=AsyncMock,
        return_value=_catalog_response(
            [
                {
                    "id": "openai/gpt-4.1",
                    "capabilities": ["streaming", "tool-calling"],
                },
                {
                    "id": "openai/text-only",
                    "capabilities": ["streaming"],
                },
                {
                    "id": "openai/no-stream-tools",
                    "capabilities": ["tool-calling"],
                },
            ]
        ),
    ) as mock_get:
        assert await github_models_provider.list_model_ids() == frozenset(
            {"openai/gpt-4.1"}
        )

    mock_get.assert_awaited_once_with(
        GITHUB_MODELS_CATALOG_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer test-github-models-token",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )


@pytest.mark.asyncio
async def test_model_list_rejects_malformed_payload(
    github_models_provider: GitHubModelsProvider,
) -> None:
    with (
        patch.object(
            github_models_provider._model_list_client,
            "get",
            new_callable=AsyncMock,
            return_value=_catalog_response({"data": []}),
        ),
        pytest.raises(ModelListResponseError, match="top-level array"),
    ):
        await github_models_provider.list_model_ids()


@pytest.mark.asyncio
async def test_model_list_returns_empty_set_when_no_models_support_streaming_tools(
    github_models_provider: GitHubModelsProvider,
) -> None:
    with patch.object(
        github_models_provider._model_list_client,
        "get",
        new_callable=AsyncMock,
        return_value=_catalog_response(
            [
                {"id": "openai/text-only", "capabilities": ["streaming"]},
                {"id": "openai/non-stream-tool", "capabilities": ["tool-calling"]},
            ]
        ),
    ):
        assert await github_models_provider.list_model_ids() == frozenset()


@pytest.mark.asyncio
async def test_stream_response_text(
    github_models_provider: GitHubModelsProvider,
) -> None:
    delta = SimpleNamespace(
        content="Hello from GitHub Models",
        reasoning_content=None,
        tool_calls=None,
    )

    with patch.object(
        github_models_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=_stream(_chunk(delta)),
    ) as mock_create:
        events = [
            event async for event in github_models_provider.stream_response(_request())
        ]

    parsed = parse_sse_text("".join(events))
    assert any(
        event.event == "content_block_delta"
        and event.data.get("delta", {}).get("text") == "Hello from GitHub Models"
        for event in parsed
    )
    assert mock_create.call_args.kwargs["model"] == "openai/gpt-4.1"
    assert mock_create.call_args.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_stream_response_tool_call(
    github_models_provider: GitHubModelsProvider,
) -> None:
    tool_call = SimpleNamespace(
        index=0,
        id="call_1",
        function=SimpleNamespace(name="echo", arguments='{"value":"x"}'),
    )
    delta = SimpleNamespace(
        content=None, reasoning_content=None, tool_calls=[tool_call]
    )
    request = MessagesRequest.model_validate(
        {
            "model": "openai/gpt-4.1",
            "messages": [{"role": "user", "content": "Use the tool"}],
            "tools": [
                {
                    "name": "echo",
                    "description": "Echo a value",
                    "input_schema": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                }
            ],
        }
    )

    with patch.object(
        github_models_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=_stream(_chunk(delta, finish_reason="tool_calls")),
    ):
        events = [
            event async for event in github_models_provider.stream_response(request)
        ]

    parsed = parse_sse_text("".join(events))
    assert any(
        event.event == "content_block_start"
        and event.data.get("content_block", {}).get("type") == "tool_use"
        and event.data.get("content_block", {}).get("name") == "echo"
        for event in parsed
    )
    assert any(
        event.event == "content_block_delta"
        and event.data.get("delta", {}).get("partial_json") == '{"value":"x"}'
        for event in parsed
    )


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(
    github_models_provider: GitHubModelsProvider,
) -> None:
    delta = SimpleNamespace(
        content=None,
        reasoning_content="Thinking via GitHub Models",
        tool_calls=None,
    )

    with patch.object(
        github_models_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=_stream(_chunk(delta)),
    ):
        events = [
            event async for event in github_models_provider.stream_response(_request())
        ]

    parsed = parse_sse_text("".join(events))
    assert any(
        event.event == "content_block_delta"
        and event.data.get("delta", {}).get("thinking") == "Thinking via GitHub Models"
        for event in parsed
    )


@pytest.mark.asyncio
async def test_cleanup(github_models_provider: GitHubModelsProvider) -> None:
    github_models_provider._client = AsyncMock()
    github_models_provider._model_list_client = AsyncMock()

    await github_models_provider.cleanup()

    github_models_provider._client.close.assert_called_once()
    github_models_provider._model_list_client.aclose.assert_called_once()
