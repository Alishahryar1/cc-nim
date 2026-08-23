"""Tests for the OMLX OpenAI-compatible local provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.config.provider_catalog import OMLX_DEFAULT_BASE
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import (
    REASONING_OFF,
    immediate_admission,
    make_provider_config,
    profiled_provider,
    reasoning_for,
)

OMLX_MODEL = "omlx/local-model"


@pytest.fixture
def provider() -> OpenAIChatProvider:
    return profiled_provider(
        "omlx",
        make_provider_config(api_key="test-omlx-key", base_url=OMLX_DEFAULT_BASE),
        admission=immediate_admission(),
    )


def test_default_base_url_constant() -> None:
    assert OMLX_DEFAULT_BASE == "http://localhost:8001/v1"


def test_init_uses_openai_chat_client() -> None:
    config = make_provider_config(
        api_key="test-omlx-key",
        base_url="http://localhost:8001/v1/",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch(
        "free_claude_code.providers.openai_chat.provider.AsyncOpenAI"
    ) as openai_client:
        provider = profiled_provider("omlx", config, admission=immediate_admission())

    assert provider._provider_name == "OMLX"
    assert provider._base_url == "http://localhost:8001/v1"
    assert provider._api_key == "test-omlx-key"
    timeout = openai_client.call_args.kwargs["timeout"]
    assert (timeout.read, timeout.write, timeout.connect) == (600.0, 15.0, 5.0)


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("http://localhost:8001", "http://localhost:8001/v1"),
        ("http://localhost:8001/", "http://localhost:8001/v1"),
        ("http://localhost:8001/v1", "http://localhost:8001/v1"),
        ("http://localhost:8001/v1/", "http://localhost:8001/v1"),
    ],
)
def test_init_normalizes_openai_base_url(configured: str, expected: str) -> None:
    config = make_provider_config(api_key="test-omlx-key", base_url=configured)
    with patch(
        "free_claude_code.providers.openai_chat.provider.AsyncOpenAI"
    ) as openai_client:
        provider = profiled_provider("omlx", config, admission=immediate_admission())

    assert provider._base_url == expected
    assert openai_client.call_args.kwargs["base_url"] == expected


def test_build_request_body_uses_openai_chat_shape(
    provider: OpenAIChatProvider,
) -> None:
    request = make_messages_request(OMLX_MODEL, max_tokens=None)

    body = provider._build_request_body(request, reasoning=reasoning_for(request))

    assert body["model"] == OMLX_MODEL
    assert "max_tokens" not in body
    assert body["messages"][0]["role"] == "system"
    assert "thinking" not in body


def test_replay_strips_thinking_blocks_when_disabled(
    provider: OpenAIChatProvider,
) -> None:
    request = make_messages_request(
        OMLX_MODEL,
        system=None,
        messages=[
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "private", "signature": "s"},
                    {"type": "text", "text": "visible"},
                ],
            },
        ],
    )

    body = provider._build_request_body(request, reasoning=REASONING_OFF)

    assert body["messages"][1]["content"] == "visible"
    assert "extra_body" not in body


@pytest.mark.asyncio
async def test_stream_response_uses_shared_openai_chat_provider(
    provider: OpenAIChatProvider,
) -> None:
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello from OMLX",
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    chunk.usage = MagicMock(prompt_tokens=8, completion_tokens=4)

    async def stream():
        yield chunk

    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=stream(),
    ) as create:
        output = "".join(
            [
                event
                async for event in provider.stream_response(
                    make_messages_request(OMLX_MODEL)
                )
            ]
        )

    assert create.call_args.kwargs["stream"] is True
    assert create.call_args.kwargs["model"] == OMLX_MODEL
    assert "Hello from OMLX" in output
    assert parse_sse_text(output)[-1].event == "message_stop"
