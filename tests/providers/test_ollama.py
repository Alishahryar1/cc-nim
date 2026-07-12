"""Tests for the Ollama OpenAI-compatible provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.config.provider_catalog import OLLAMA_DEFAULT_BASE
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import OpenAIChatProvider
from tests.providers.request_factory import make_messages_request
from tests.providers.support import passthrough_rate_limiter, profiled_provider

OLLAMA_MODEL = "llama3.1:8b"


def _provider(base_url: str = OLLAMA_DEFAULT_BASE) -> OpenAIChatProvider:
    return profiled_provider(
        "ollama",
        ProviderConfig(api_key="ollama", base_url=base_url),
        rate_limiter=passthrough_rate_limiter(),
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("http://localhost:11434", "http://localhost:11434/v1"),
        ("http://localhost:11434/", "http://localhost:11434/v1"),
        ("http://localhost:11434/v1", "http://localhost:11434/v1"),
    ],
)
def test_init_normalizes_openai_base_url(configured: str, expected: str) -> None:
    with patch(
        "free_claude_code.providers.openai_chat.provider.AsyncOpenAI"
    ) as openai_client:
        provider = _provider(configured)

    assert provider._provider_name == "OLLAMA"
    assert provider._base_url == expected
    assert provider._api_key == "ollama"
    assert openai_client.call_args.kwargs["base_url"] == expected


def test_build_request_body_uses_openai_chat_shape() -> None:
    body = _provider()._build_request_body(make_messages_request(OLLAMA_MODEL))

    assert body["model"] == OLLAMA_MODEL
    assert body["messages"][0]["role"] == "system"
    assert "thinking" not in body
    assert "extra_body" not in body


@pytest.mark.asyncio
async def test_stream_response_uses_shared_openai_chat_provider() -> None:
    provider = _provider()
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello from Ollama",
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
                    make_messages_request(OLLAMA_MODEL)
                )
            ]
        )

    assert create.call_args.kwargs["stream"] is True
    assert create.call_args.kwargs["model"] == OLLAMA_MODEL
    assert "Hello from Ollama" in output
    assert parse_sse_text(output)[-1].event == "message_stop"
