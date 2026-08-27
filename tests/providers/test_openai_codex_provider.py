import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    text_content,
)
from free_claude_code.core.diagnostics import ERROR_DETAIL_DISPLAY_CAP_BYTES
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningEffort, ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_codex.auth import (
    OpenAIAccess,
    OpenAIAuthManager,
)
from free_claude_code.providers.openai_codex.provider import OpenAICodexProvider
from tests.providers.support import make_provider_config


class _FakeAuth(OpenAIAuthManager):
    def __init__(self) -> None:
        self.access_calls = 0
        self.recovery_calls = 0
        self.current_token = "access_1"

    async def access(self, *, force_refresh: bool = False) -> OpenAIAccess:
        self.access_calls += 1
        return OpenAIAccess(self.current_token, "account_1", False)

    async def recover_unauthorized(self, rejected_token: str) -> OpenAIAccess:
        assert rejected_token == self.current_token
        self.recovery_calls += 1
        self.current_token = "access_2"
        return OpenAIAccess(self.current_token, "account_1", False)


class _CloseTrackingResponse(httpx.Response):
    """HTTPX response that records close order and can fail during cleanup."""

    def __init__(
        self,
        status_code: int,
        *,
        close_error: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(status_code, **kwargs)
        self.close_calls = 0
        self._close_error = close_error

    async def aclose(self) -> None:
        self.close_calls += 1
        await super().aclose()
        if self._close_error is not None:
            raise self._close_error


class _CloseAwareAuth(_FakeAuth):
    def __init__(self, responses: list[_CloseTrackingResponse]) -> None:
        super().__init__()
        self._responses = responses

    async def recover_unauthorized(self, rejected_token: str) -> OpenAIAccess:
        assert self._responses[-1].close_calls == 1
        return await super().recover_unauthorized(rejected_token)


def _config() -> ProviderConfig:
    return make_provider_config(
        api_key="",
        base_url="https://chatgpt.com/backend-api/codex",
        rate_limit=100,
        rate_window=1,
        max_concurrency=2,
    )


def _admission(*, max_concurrency: int = 2) -> ProviderAdmissionController:
    return ProviderAdmissionController(
        provider_name="openai",
        rate_limit=100,
        rate_window=1,
        max_concurrency=max_concurrency,
        base_delay=0,
        max_delay=0,
        jitter=0,
    )


def _request() -> MessagesRequest:
    return MessagesRequest(
        model="gpt-test",
        max_tokens=1024,
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )


def _responses_request() -> OpenAIResponsesRequest:
    return OpenAIResponsesRequest.model_validate(
        {
            "model": "gpt-test",
            "input": [{"role": "user", "content": "hello"}],
            "stream": False,
            "store": True,
            "previous_response_id": "resp_previous",
            "max_output_tokens": 1024,
            "metadata": {"source": "test"},
            "future_option": {"enabled": True},
        }
    )


def _client_control_request() -> MessagesRequest:
    return MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"user_id": "example-user"},
            "thinking": {"type": "adaptive", "display": "omitted"},
            "context_management": {
                "edits": [
                    {
                        "type": "clear_thinking_20251015",
                        "keep": "all",
                    }
                ]
            },
            "output_config": {"effort": "high"},
            "stream": True,
        }
    )


def _sse(*events: tuple[str, dict[str, Any]]) -> str:
    return "".join(
        f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
        for event_type, payload in events
    )


def _complete_stream(text: str) -> str:
    return _sse(
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


@pytest.mark.asyncio
async def test_provider_uses_subscription_headers_and_visible_model_catalog() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "slug": "gpt-visible",
                            "visibility": "list",
                            "supported_in_api": False,
                            "supported_reasoning_levels": [{"effort": "high"}],
                        },
                        {"slug": "gpt-hidden", "visibility": "hide"},
                    ]
                },
                request=request,
            )
        assert request.url.path.endswith("/responses")
        return httpx.Response(
            200,
            content=_complete_stream("hello").encode(),
            request=request,
        )

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    auth = _FakeAuth()
    provider = OpenAICodexProvider(
        _config(), auth=auth, admission=_admission(), client=client
    )

    infos = await provider.list_model_infos()
    body = await _collect(
        provider.stream_messages(
            _request(),
            input_tokens=3,
            request_id="req_test",
            response_model="claude-opus-4",
            reasoning=ReasoningPolicy.provider_default(),
        )
    )

    assert {(info.model_id, info.supports_thinking) for info in infos} == {
        ("gpt-visible", True)
    }
    response_request = requests[-1]
    assert response_request.headers["authorization"] == "Bearer access_1"
    assert response_request.headers["chatgpt-account-id"] == "account_1"
    assert response_request.headers["originator"] == "codex_cli_rs"
    assert response_request.headers["accept"] == "text/event-stream"
    assert response_request.headers["session_id"]
    payload = json.loads(response_request.content)
    assert payload["model"] == "gpt-test"
    assert payload["store"] is False
    events = parse_sse_text(body)
    assert_anthropic_stream_contract(events)
    assert text_content(events) == "hello"
    await client.aclose()


@pytest.mark.asyncio
async def test_model_discovery_close_failure_preserves_success_and_permit() -> None:
    responses: list[_CloseTrackingResponse] = []

    def handler(request: httpx.Request) -> httpx.Response:
        response = _CloseTrackingResponse(
            200,
            json={
                "models": [
                    {
                        "slug": "gpt-visible",
                        "visibility": "list",
                        "supported_in_api": False,
                    }
                ]
            },
            request=request,
            close_error=RuntimeError("cleanup failed"),
        )
        responses.append(response)
        return response

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICodexProvider(
        _config(),
        auth=_FakeAuth(),
        admission=_admission(max_concurrency=1),
        client=client,
    )

    first = await asyncio.wait_for(provider.list_model_infos(), timeout=1)
    second = await asyncio.wait_for(provider.list_model_infos(), timeout=1)

    assert {info.model_id for info in first} == {"gpt-visible"}
    assert {info.model_id for info in second} == {"gpt-visible"}
    assert [response.close_calls for response in responses] == [1, 1]
    await client.aclose()


@pytest.mark.asyncio
async def test_generation_close_failure_preserves_success_and_permit() -> None:
    responses: list[_CloseTrackingResponse] = []

    def handler(request: httpx.Request) -> httpx.Response:
        response = _CloseTrackingResponse(
            200,
            text=_complete_stream("hello"),
            headers={"content-type": "text/event-stream"},
            request=request,
            close_error=RuntimeError("cleanup failed"),
        )
        responses.append(response)
        return response

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICodexProvider(
        _config(),
        auth=_FakeAuth(),
        admission=_admission(max_concurrency=1),
        client=client,
    )

    first = await asyncio.wait_for(
        _collect(provider.stream_messages(_request(), response_model="claude-opus-4")),
        timeout=1,
    )
    second = await asyncio.wait_for(
        _collect(provider.stream_messages(_request(), response_model="claude-opus-4")),
        timeout=1,
    )

    assert text_content(parse_sse_text(first)) == "hello"
    assert text_content(parse_sse_text(second)) == "hello"
    assert [response.close_calls for response in responses] == [1, 1]
    await client.aclose()


@pytest.mark.asyncio
async def test_generation_close_failure_preserves_provider_failure() -> None:
    responses: list[_CloseTrackingResponse] = []

    def handler(request: httpx.Request) -> httpx.Response:
        response = _CloseTrackingResponse(
            400,
            json={"error": {"message": "original provider failure"}},
            request=request,
            close_error=RuntimeError("cleanup failed"),
        )
        responses.append(response)
        return response

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICodexProvider(
        _config(), auth=_FakeAuth(), admission=_admission(), client=client
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        await _collect(
            provider.stream_messages(
                _request(),
                request_id="req_close_failure",
                response_model="claude-opus-4",
            )
        )

    assert exc_info.value.status_code == 400
    assert "original provider failure" in exc_info.value.message
    assert "cleanup failed" not in exc_info.value.message
    assert [response.close_calls for response in responses] == [1]
    await client.aclose()


@pytest.mark.asyncio
async def test_provider_accepts_claude_client_controls_before_upstream_io() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=_complete_stream("hello"),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICodexProvider(
        _config(), auth=_FakeAuth(), admission=_admission(), client=client
    )
    request = _client_control_request()
    original_request = request.model_dump()
    reasoning = ReasoningPolicy.on(effort=ReasoningEffort.HIGH)

    provider.preflight_messages(request, reasoning=reasoning)
    assert requests == []

    body = await _collect(
        provider.stream_messages(
            request,
            request_id="req_client_controls",
            response_model="claude-opus-4",
            reasoning=reasoning,
        )
    )

    assert len(requests) == 1
    upstream = requests[0]
    assert upstream.url.path.endswith("/responses")
    payload = json.loads(upstream.content)
    assert payload["model"] == "gpt-test"
    assert payload["reasoning"] == {"effort": "high", "summary": "auto"}
    assert "max_output_tokens" not in payload
    assert "metadata" not in payload
    assert "context_management" not in payload
    assert "output_config" not in payload
    assert request.model_dump() == original_request
    assert_anthropic_stream_contract(parse_sse_text(body))
    await client.aclose()


@pytest.mark.asyncio
async def test_provider_relays_native_responses_with_private_field_policy() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=_complete_stream("hello"),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICodexProvider(
        _config(), auth=_FakeAuth(), admission=_admission(), client=client
    )
    request = _responses_request()
    original = request.model_dump()

    provider.preflight_responses(request)
    assert requests == []
    body = await _collect(
        provider.stream_responses(
            request,
            request_id="req_native_responses",
            response_model="openai/gpt-test",
        )
    )

    assert len(requests) == 1
    upstream = json.loads(requests[0].content)
    assert upstream["model"] == "gpt-test"
    assert upstream["input"] == [{"role": "user", "content": "hello"}]
    assert upstream["stream"] is True
    assert upstream["store"] is False
    assert upstream["future_option"] == {"enabled": True}
    assert "previous_response_id" not in upstream
    assert "max_output_tokens" not in upstream
    assert "metadata" not in upstream
    assert request.model_dump() == original

    events = parse_sse_text(body)
    assert [event.event for event in events] == [
        "response.created",
        "response.output_text.delta",
        "response.completed",
    ]
    assert events[0].data["response"]["id"] == "resp_1"
    assert events[0].data["response"]["model"] == "openai/gpt-test"
    assert events[-1].data["response"]["id"] == "resp_1"
    assert events[-1].data["response"]["model"] == "openai/gpt-test"
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value", "error_path"),
    [
        (
            "context_management",
            {
                "edits": [
                    {
                        "type": "clear_thinking_20251015",
                        "keep": {"type": "thinking_turns", "value": 2},
                    }
                ]
            },
            "context_management",
        ),
        (
            "output_config",
            {"format": {"type": "json_schema"}},
            "output_config.format",
        ),
    ],
)
async def test_provider_preflight_rejects_unrepresentable_client_controls(
    field: str,
    value: object,
    error_path: str,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500, request=request)

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICodexProvider(
        _config(), auth=_FakeAuth(), admission=_admission(), client=client
    )
    payload = {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        field: value,
    }

    with pytest.raises(InvalidRequestError, match=error_path):
        provider.preflight_messages(MessagesRequest.model_validate(payload))

    assert requests == []
    await client.aclose()


@pytest.mark.asyncio
async def test_provider_round_trips_portable_tool_name_alias() -> None:
    original = "mcp__codex_provider__" + "x" * 70
    request = MessagesRequest.model_validate(
        {
            "model": "gpt-test",
            "messages": [{"role": "user", "content": "use the tool"}],
            "tools": [
                {
                    "name": original,
                    "description": "Long tool",
                    "input_schema": {"type": "object"},
                }
            ],
            "tool_choice": {"type": "tool", "name": original},
        }
    )
    payloads: list[dict[str, Any]] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        payload = json.loads(http_request.content)
        payloads.append(payload)
        alias = payload["tools"][0]["name"]
        body = _sse(
            (
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": alias,
                    },
                },
            ),
            (
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": alias,
                        "arguments": "{}",
                    },
                },
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
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
            request=http_request,
        )

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICodexProvider(
        _config(), auth=_FakeAuth(), admission=_admission(), client=client
    )

    body = await _collect(provider.stream_messages(request))

    payload = payloads[0]
    alias = payload["tools"][0]["name"]
    assert alias != original
    assert len(alias) <= 64
    assert payload["tool_choice"] == {"type": "function", "name": alias}
    events = parse_sse_text(body)
    assert_anthropic_stream_contract(events)
    tool_start = next(
        event.data["content_block"]
        for event in events
        if event.event == "content_block_start"
    )
    assert tool_start["name"] == original
    assert alias not in body
    await client.aclose()


@pytest.mark.asyncio
async def test_early_truncated_attempt_is_retried_without_duplicate_output() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            body = _sse(
                (
                    "response.created",
                    {"type": "response.created", "response": {"id": "first"}},
                ),
                (
                    "response.output_text.delta",
                    {"type": "response.output_text.delta", "delta": "abandoned"},
                ),
            )
        else:
            body = _complete_stream("kept")
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
        _config(), auth=_FakeAuth(), admission=_admission(), client=client
    )

    body = await _collect(
        provider.stream_messages(
            _request(),
            request_id="req_retry",
            response_model="claude-opus-4",
        )
    )

    events = parse_sse_text(body)
    assert_anthropic_stream_contract(events)
    assert attempts == 2
    assert text_content(events) == "kept"
    assert "abandoned" not in body
    await client.aclose()


@pytest.mark.asyncio
async def test_pre_stream_retry_discards_held_message_start() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                json={"error": {"message": "try again"}},
                request=request,
            )
        return httpx.Response(
            200,
            text=_complete_stream("kept"),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICodexProvider(
        _config(), auth=_FakeAuth(), admission=_admission(), client=client
    )

    body = await _collect(
        provider.stream_messages(
            _request(),
            request_id="req_pre_stream",
            response_model="claude-opus-4",
        )
    )

    events = parse_sse_text(body)
    assert_anthropic_stream_contract(events)
    assert attempts == 2
    assert body.count("event: message_start") == 1
    assert text_content(events) == "kept"
    await client.aclose()


@pytest.mark.asyncio
async def test_non_streaming_success_is_retried_then_reports_bounded_body() -> None:
    attempts = 0
    tail_sentinel = "must-not-escape-the-diagnostic-cap"
    upstream_body = (
        "upstream contract changed\n"
        + "x" * ERROR_DETAIL_DISPLAY_CAP_BYTES
        + tail_sentinel
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            content=upstream_body,
            headers={"content-type": "application/json"},
            request=request,
        )

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICodexProvider(
        _config(), auth=_FakeAuth(), admission=_admission(), client=client
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        await _collect(
            provider.stream_messages(
                _request(),
                request_id="req_non_stream",
                response_model="claude-opus-4",
            )
        )

    assert attempts > 1
    assert "upstream contract changed" in exc_info.value.message
    assert f"truncated after {ERROR_DETAIL_DISPLAY_CAP_BYTES} bytes" in (
        exc_info.value.message
    )
    assert tail_sentinel not in exc_info.value.message
    assert "Request ID: req_non_stream" in exc_info.value.message
    await client.aclose()


@pytest.mark.asyncio
async def test_truncated_attempt_after_commit_is_not_retried_or_duplicated() -> None:
    attempts = 0
    committed_text = "x" * 70_000

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            text=_sse(
                (
                    "response.created",
                    {"type": "response.created", "response": {"id": "first"}},
                ),
                (
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "delta": committed_text,
                    },
                ),
            ),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICodexProvider(
        _config(), auth=_FakeAuth(), admission=_admission(), client=client
    )
    chunks: list[str] = []

    with pytest.raises(ExecutionFailure):
        async for chunk in provider.stream_messages(
            _request(),
            request_id="req_committed",
            response_model="claude-opus-4",
        ):
            chunks.extend((chunk,))

    body = "".join(chunks)
    assert attempts == 1
    assert body.count(committed_text) == 1
    assert body.count("event: message_start") == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_unauthorized_response_forces_one_auth_refresh() -> None:
    authorizations: list[str] = []
    unauthorized_responses: list[_CloseTrackingResponse] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorizations.append(request.headers["authorization"])
        if len(authorizations) == 1:
            response = _CloseTrackingResponse(
                401,
                json={"error": {"message": "expired"}},
                request=request,
            )
            unauthorized_responses.append(response)
            return response
        return httpx.Response(
            200,
            text=_complete_stream("recovered"),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    auth = _CloseAwareAuth(unauthorized_responses)
    provider = OpenAICodexProvider(
        _config(), auth=auth, admission=_admission(), client=client
    )

    body = await _collect(
        provider.stream_messages(
            _request(),
            request_id="req_auth",
            response_model="claude-opus-4",
        )
    )

    assert authorizations == ["Bearer access_1", "Bearer access_2"]
    assert auth.recovery_calls == 1
    assert unauthorized_responses[0].close_calls == 1
    assert text_content(parse_sse_text(body)) == "recovered"
    await client.aclose()


@pytest.mark.asyncio
async def test_model_discovery_re_request_is_a_second_admitted_attempt() -> None:
    authorizations: list[str] = []
    unauthorized_responses: list[_CloseTrackingResponse] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorizations.append(request.headers["authorization"])
        if len(authorizations) == 1:
            response = _CloseTrackingResponse(401, request=request)
            unauthorized_responses.append(response)
            return response
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "slug": "gpt-visible",
                        "visibility": "list",
                        "supported_in_api": False,
                        "supported_reasoning_levels": [{"effort": "high"}],
                    }
                ]
            },
            request=request,
        )

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    auth = _CloseAwareAuth(unauthorized_responses)
    provider = OpenAICodexProvider(
        _config(), auth=auth, admission=_admission(), client=client
    )

    with patch("free_claude_code.providers.admission.trace_event") as trace:
        infos = await provider.list_model_infos()

    assert {info.model_id for info in infos} == {"gpt-visible"}
    assert authorizations == ["Bearer access_1", "Bearer access_2"]
    assert auth.recovery_calls == 1
    assert unauthorized_responses[0].close_calls == 1
    attempt_rows = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event", "").startswith("provider.attempt.")
    ]
    assert [row["event"] for row in attempt_rows] == [
        "provider.attempt.started",
        "provider.attempt.resolved",
        "provider.attempt.started",
        "provider.attempt.resolved",
    ]
    assert [row["attempt"] for row in attempt_rows] == [1, 1, 2, 2]
    assert [row.get("outcome") for row in attempt_rows] == [
        None,
        "corrected",
        None,
        "accepted",
    ]
    assert len({row["execution_id"] for row in attempt_rows}) == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_auth_refresh_failure_does_not_repeat_rejected_provider_call() -> None:
    class FailingRecoveryAuth(_FakeAuth):
        async def recover_unauthorized(self, rejected_token: str) -> OpenAIAccess:
            assert rejected_token == self.current_token
            self.recovery_calls += 1
            raise httpx.ReadError(
                "refresh interrupted",
                request=httpx.Request("POST", "https://auth.openai.com/oauth/token"),
            )

    authorizations: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorization = request.headers["authorization"]
        authorizations.append(authorization)
        if authorization == "Bearer access_1":
            return httpx.Response(401, request=request)
        return httpx.Response(
            200,
            text=_complete_stream("recovered"),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    auth = FailingRecoveryAuth()
    provider = OpenAICodexProvider(
        _config(), auth=auth, admission=_admission(), client=client
    )

    with pytest.raises(ExecutionFailure, match="refresh interrupted") as exc_info:
        await _collect(
            provider.stream_messages(
                _request(),
                request_id="req_auth_transient",
                response_model="claude-opus-4",
            )
        )

    assert isinstance(exc_info.value.__cause__, httpx.ReadError)
    assert authorizations == ["Bearer access_1"]
    assert auth.recovery_calls == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_second_unauthorized_response_is_terminal_without_refresh_loop() -> None:
    authorizations: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorizations.append(request.headers["authorization"])
        return httpx.Response(
            401,
            json={"error": {"message": "still expired"}},
            request=request,
        )

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    auth = _FakeAuth()
    provider = OpenAICodexProvider(
        _config(), auth=auth, admission=_admission(), client=client
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        await _collect(
            provider.stream_messages(
                _request(),
                request_id="req_auth_terminal",
                response_model="claude-opus-4",
            )
        )

    assert exc_info.value.status_code == 401
    assert authorizations == ["Bearer access_1", "Bearer access_2"]
    assert auth.recovery_calls == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_final_attempt_unauthorized_preserves_the_provider_401() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        status = 503 if attempts < 5 else 401
        return httpx.Response(
            status,
            json={"error": {"message": f"attempt {attempts}"}},
            request=request,
        )

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    auth = _FakeAuth()
    provider = OpenAICodexProvider(
        _config(), auth=auth, admission=_admission(), client=client
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        await _collect(
            provider.stream_messages(
                _request(),
                request_id="req_final_401",
                response_model="claude-opus-4",
            )
        )

    assert attempts == 5
    assert auth.recovery_calls == 0
    assert exc_info.value.status_code == 401
    assert "Request ID: req_final_401" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)
    assert exc_info.value.__cause__.response.status_code == 401
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_failure_redacts_credentials_from_customer_diagnostic() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_sse(
                (
                    "response.failed",
                    {
                        "type": "response.failed",
                        "response": {
                            "error": {
                                "code": "invalid_request",
                                "message": (
                                    "Authorization: Bearer "
                                    "sk-this-must-never-be-returned"
                                ),
                            }
                        },
                    },
                )
            ),
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    client = httpx.AsyncClient(
        base_url="https://chatgpt.com/backend-api/codex/",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenAICodexProvider(
        _config(), auth=_FakeAuth(), admission=_admission(), client=client
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        await _collect(
            provider.stream_messages(
                _request(),
                request_id="req_redaction",
                response_model="claude-opus-4",
            )
        )

    assert "sk-this-must-never-be-returned" not in exc_info.value.message
    assert "Authorization: <redacted>" in exc_info.value.message
    await client.aclose()
