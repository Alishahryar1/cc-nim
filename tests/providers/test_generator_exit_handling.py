"""GeneratorExit must be caught during stream iteration to avoid connection refused errors.

The old ``except asyncio.CancelledError, GeneratorExit:`` (Python 3 syntax) caught
only ``CancelledError`` and bound the name ``GeneratorExit`` — the actual
``GeneratorExit`` was never caught.  The fix catches both exceptions separately:
``CancelledError`` is re-raised (caller must handle cancellation), while
``GeneratorExit`` returns cleanly (the generator is closing because the caller
stopped iterating).
"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from free_claude_code.config.nim import NimSettings
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.open_router import OpenRouterProvider
from free_claude_code.providers.openai_codex.auth import (
    OpenAIAccess,
    OpenAIAuthManager,
)
from free_claude_code.providers.openai_codex.provider import OpenAICodexProvider
from tests.providers.request_factory import make_messages_request


def _config(base_url: str) -> ProviderConfig:
    return ProviderConfig(
        api_key="test_key",
        base_url=base_url,
        rate_limit=1_000_000,
        rate_window=1,
        max_concurrency=1_000,
        http_read_timeout=30.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )


def _admission():
    return ProviderAdmissionController(
        provider_name="test",
        rate_limit=1_000_000,
        rate_window=1.0,
        max_concurrency=1_000,
        max_attempts=5,
        base_delay=0.0,
        max_delay=0.0,
        jitter=0.0,
    )


def _stream_that_raises_cancelled_error():
    """Simulate a stream that raises CancelledError mid-iteration."""

    async def stream():
        chunk = MagicMock()
        chunk.choices = [
            MagicMock(
                delta=MagicMock(content="partial", reasoning_content=""),
                finish_reason=None,
            )
        ]
        chunk.usage = None
        yield chunk
        raise asyncio.CancelledError()

    return stream()


class _FakeCodexAuth(OpenAIAuthManager):
    def __init__(self) -> None:
        self.access_calls = 0

    async def access(self, *, force_refresh: bool = False) -> OpenAIAccess:
        self.access_calls += 1
        return OpenAIAccess("access_token", "account_1", False)

    async def recover_unauthorized(self, rejected_token: str) -> OpenAIAccess:
        return OpenAIAccess("new_token", "account_1", False)


def _codex_config() -> ProviderConfig:
    return ProviderConfig(
        api_key="",
        base_url="https://chatgpt.com/backend-api/codex",
        rate_limit=100,
        rate_window=1,
        max_concurrency=2,
    )


def _codex_request() -> MessagesRequest:
    return MessagesRequest(
        model="gpt-test",
        max_tokens=1024,
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )


def _codex_sse(*events) -> str:
    import json

    return "".join(
        f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
        for event_type, payload in events
    )


def _codex_complete_stream(text: str = "hello") -> str:
    return _codex_sse(
        (
            "response.created",
            {"type": "response.created", "response": {"id": "resp_1"}},
        ),
        (
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": text},
        ),
        (
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
            },
        ),
    )


async def _collect(stream: AsyncIterator[str]) -> str:
    return "".join([chunk async for chunk in stream])


# ---------------------------------------------------------------------------
# OpenRouter (OpenAI Chat) provider tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openrouter_generator_exit_closes_cleanly() -> None:
    """When aclose() throws GeneratorExit into the provider stream generator,
    it should close cleanly without raising an error."""
    admission = _admission()
    provider = OpenRouterProvider(
        _config("https://openrouter.ai/api/v1"), admission=admission
    )

    async def multi_chunk_stream(**_kwargs):
        for text in ["a", "b", "c"]:
            chunk = MagicMock()
            chunk.choices = [
                MagicMock(
                    delta=MagicMock(content=text, reasoning_content=""),
                    finish_reason="stop" if text == "c" else None,
                )
            ]
            chunk.usage = None
            yield chunk

    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        side_effect=multi_chunk_stream,
    ):
        gen = cast(
            AsyncGenerator[str],
            provider.stream_response(make_messages_request(), request_id="req_genexit"),
        )
        first = await gen.__anext__()
        assert "event: message_start" in first
        # Close mid-stream — triggers GeneratorExit inside run()
        await gen.aclose()


@pytest.mark.asyncio
async def test_openrouter_aclose_during_stream_does_not_raise() -> None:
    """Calling aclose() mid-stream should not raise an unhandled exception."""
    admission = _admission()
    provider = OpenRouterProvider(
        _config("https://openrouter.ai/api/v1"), admission=admission
    )

    # Use a stream that yields multiple chunks so we can close mid-stream
    async def multi_chunk_stream(**_kwargs):
        for text in ["a", "b", "c"]:
            chunk = MagicMock()
            chunk.choices = [
                MagicMock(
                    delta=MagicMock(content=text, reasoning_content=""),
                    finish_reason="stop" if text == "c" else None,
                )
            ]
            chunk.usage = None
            yield chunk

    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        side_effect=multi_chunk_stream,
    ):
        gen = cast(
            AsyncGenerator[str],
            provider.stream_response(make_messages_request(), request_id="req_aclose"),
        )
        # Get first event
        first = await gen.__anext__()
        assert "event:" in first
        # Close mid-stream — should not raise
        await gen.aclose()


@pytest.mark.asyncio
async def test_openrouter_cancelled_error_is_reraised() -> None:
    """CancelledError should propagate (not be swallowed) during streaming."""
    admission = _admission()
    provider = OpenRouterProvider(
        _config("https://openrouter.ai/api/v1"), admission=admission
    )

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=_stream_that_raises_cancelled_error(),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        [
            event
            async for event in provider.stream_response(
                make_messages_request(), request_id="req_cancelled"
            )
        ]


# ---------------------------------------------------------------------------
# NVIDIA NIM provider tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nvidia_nim_aclose_during_stream_does_not_raise() -> None:
    """Calling aclose() mid-stream on NIM should not raise an unhandled error."""
    admission = _admission()
    provider = NvidiaNimProvider(
        _config("https://integrate.api.nvidia.com/v1"),
        nim_settings=NimSettings(),
        admission=admission,
    )

    async def multi_chunk_stream(**_kwargs):
        for text in ["a", "b", "c"]:
            chunk = MagicMock()
            chunk.choices = [
                MagicMock(
                    delta=MagicMock(content=text, reasoning_content=""),
                    finish_reason="stop" if text == "c" else None,
                )
            ]
            chunk.usage = None
            yield chunk

    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        side_effect=multi_chunk_stream,
    ):
        gen = cast(
            AsyncGenerator[str],
            provider.stream_response(
                make_messages_request(), request_id="req_nim_aclose"
            ),
        )
        first = await gen.__anext__()
        assert "event:" in first
        await gen.aclose()


@pytest.mark.asyncio
async def test_nvidia_nim_cancelled_error_is_reraised() -> None:
    """CancelledError should propagate during NIM streaming."""
    admission = _admission()
    provider = NvidiaNimProvider(
        _config("https://integrate.api.nvidia.com/v1"),
        nim_settings=NimSettings(),
        admission=admission,
    )

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=_stream_that_raises_cancelled_error(),
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        [
            event
            async for event in provider.stream_response(
                make_messages_request(), request_id="req_nim_cancelled"
            )
        ]


# ---------------------------------------------------------------------------
# Codex provider tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_codex_aclose_during_stream_does_not_raise() -> None:
    """Calling aclose() mid-stream on Codex should not raise an unhandled error."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "slug": "gpt-test",
                            "visibility": "list",
                            "supported_reasoning_levels": [],
                        }
                    ]
                },
                request=request,
            )
        # Return a multi-chunk SSE stream
        body = _codex_sse(
            (
                "response.created",
                {"type": "response.created", "response": {"id": "resp_1"}},
            ),
            (
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "hello"},
            ),
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "usage": {"input_tokens": 3, "output_tokens": 1},
                    },
                },
            ),
        )
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICodexProvider(
        _codex_config(),
        auth=_FakeCodexAuth(),
        admission=_admission(),
        client=client,
    )

    gen = cast(
        AsyncGenerator[str],
        provider.stream_response(
            _codex_request(),
            request_id="req_codex_aclose",
            response_model="gpt-test",
        ),
    )
    first = await gen.__anext__()
    assert "event: message_start" in first
    await gen.aclose()
    await client.aclose()


@pytest.mark.asyncio
async def test_codex_cancelled_error_is_reraised() -> None:
    """CancelledError should propagate during Codex streaming."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "slug": "gpt-test",
                            "visibility": "list",
                            "supported_reasoning_levels": [],
                        }
                    ]
                },
                request=request,
            )
        body = _codex_complete_stream("partial")
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICodexProvider(
        _codex_config(),
        auth=_FakeCodexAuth(),
        admission=_admission(),
        client=client,
    )

    async def consume_and_cancel():
        async for _event in provider.stream_response(
            _codex_request(),
            request_id="req_codex_cancelled",
            response_model="gpt-test",
        ):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await consume_and_cancel()
    await client.aclose()
