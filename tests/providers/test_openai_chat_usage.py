"""OpenAI-chat streamed usage helper tests."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import openai
import pytest
from httpx import Request, Response

from core.anthropic.stream_contracts import parse_sse_text
from providers.base import ProviderConfig
from providers.rate_limit import GlobalRateLimiter
from providers.transports.openai_chat import OpenAIChatTransport
from providers.transports.openai_chat.usage import (
    clone_without_stream_usage,
    is_stream_usage_rejection,
    request_stream_usage,
    usage_int,
)


class _UsageTestProvider(OpenAIChatTransport):
    def __init__(self):
        super().__init__(
            ProviderConfig(
                api_key="test_key",
                base_url="https://provider.example/v1",
                rate_limit=100,
                rate_window=60,
            ),
            provider_name="USAGE_TEST",
            base_url="https://provider.example/v1",
            api_key="test_key",
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        return {"model": request.model, "messages": [{"role": "user", "content": "x"}]}


def _bad_request(message: str, body: object | None = None) -> openai.BadRequestError:
    response = Response(
        400,
        request=Request("POST", "https://provider.example/v1/chat/completions"),
    )
    return openai.BadRequestError(message, response=response, body=body)


async def _stream(chunks):
    for chunk in chunks:
        yield chunk


def _chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    usage: Any = None,
):
    if content is None and finish_reason is None:
        return SimpleNamespace(choices=[], usage=usage)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    reasoning_content=None,
                    tool_calls=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def test_request_stream_usage_adds_stream_options_when_absent():
    body = {"model": "m"}

    request_stream_usage(body)

    assert body["stream_options"] == {"include_usage": True}


def test_request_stream_usage_preserves_existing_stream_options():
    stream_options = {"foo": "bar"}
    body = {"model": "m", "stream_options": stream_options}

    request_stream_usage(body)

    assert body["stream_options"] == {"foo": "bar", "include_usage": True}
    assert body["stream_options"] is stream_options


def test_clone_without_stream_usage_removes_only_include_usage():
    body = {
        "model": "m",
        "stream_options": {"foo": "bar", "include_usage": True},
    }

    retry_body = clone_without_stream_usage(body)

    assert retry_body == {"model": "m", "stream_options": {"foo": "bar"}}
    assert body["stream_options"] == {"foo": "bar", "include_usage": True}


def test_clone_without_stream_usage_drops_empty_stream_options():
    body = {"model": "m", "stream_options": {"include_usage": True}}

    retry_body = clone_without_stream_usage(body)

    assert retry_body == {"model": "m"}


def test_usage_int_reads_dict_object_and_model_extra():
    assert usage_int({"prompt_tokens": 11}, "prompt_tokens") == 11
    assert usage_int(SimpleNamespace(completion_tokens=7), "completion_tokens") == 7
    assert (
        usage_int(
            SimpleNamespace(model_extra={"prompt_cache_hit_tokens": 3}),
            "prompt_cache_hit_tokens",
        )
        == 3
    )
    assert usage_int(SimpleNamespace(prompt_tokens=None), "prompt_tokens") is None
    assert usage_int({"prompt_tokens": True}, "prompt_tokens") is None


def test_stream_usage_rejection_matches_usage_option_400():
    error = _bad_request(
        "Unrecognized request argument supplied: stream_options",
        {"error": {"message": "stream_options is unsupported"}},
    )

    assert is_stream_usage_rejection(error)


def test_stream_usage_rejection_does_not_match_unrelated_400():
    error = _bad_request(
        "messages: invalid role",
        {"error": {"message": "messages contains invalid role"}},
    )

    assert not is_stream_usage_rejection(error)


@pytest.mark.asyncio
async def test_openai_chat_stream_requests_usage_and_uses_provider_prompt_tokens():
    GlobalRateLimiter.reset_instance()
    try:
        provider = _UsageTestProvider()
        request = SimpleNamespace(model="m")
        usage = SimpleNamespace(prompt_tokens=22, completion_tokens=4)
        create = AsyncMock(
            return_value=_stream(
                [
                    _chunk(content="hello"),
                    _chunk(finish_reason="stop"),
                    _chunk(usage=usage),
                ]
            )
        )

        with patch.object(provider._client.chat.completions, "create", create):
            events = [
                event
                async for event in provider.stream_response(request, input_tokens=7)
            ]

        create.assert_awaited_once()
        await_args = create.await_args
        assert await_args is not None
        assert await_args.kwargs["stream_options"] == {"include_usage": True}
        parsed = parse_sse_text("".join(events))
        start_usage = next(
            event.data["message"]["usage"]
            for event in parsed
            if event.event == "message_start"
        )
        final_usage = next(
            event.data["usage"] for event in parsed if event.event == "message_delta"
        )
        assert start_usage["input_tokens"] == 7
        assert final_usage == {"input_tokens": 22, "output_tokens": 4}
    finally:
        GlobalRateLimiter.reset_instance()


@pytest.mark.asyncio
async def test_openai_chat_stream_retries_without_usage_when_option_is_rejected():
    GlobalRateLimiter.reset_instance()
    try:
        provider = _UsageTestProvider()
        body = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
        request_stream_usage(body)
        create = AsyncMock(
            side_effect=[
                _bad_request(
                    "stream_options is unsupported",
                    {"error": {"message": "stream_options is unsupported"}},
                ),
                object(),
            ]
        )

        with patch.object(provider._client.chat.completions, "create", create):
            _stream_obj, used_body = await provider._create_stream(body)

        assert create.await_count == 2
        assert create.await_args_list[0].kwargs["stream_options"] == {
            "include_usage": True
        }
        assert "stream_options" not in create.await_args_list[1].kwargs
        assert "stream_options" not in used_body
    finally:
        GlobalRateLimiter.reset_instance()
