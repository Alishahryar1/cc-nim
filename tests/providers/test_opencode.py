"""Tests for catalog-driven OpenCode Chat/Responses routing."""

import asyncio
import json
from collections.abc import Callable
from unittest.mock import AsyncMock, patch

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    text_content,
)
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.providers.model_listing import ModelListResponseError
from free_claude_code.providers.opencode import (
    OpenCodeProvider,
    create_opencode_provider,
)
from free_claude_code.providers.opencode.catalog import (
    OPENCODE_CATALOG_URL,
    OpenCodeCatalog,
    OpenCodeUpstreamTransport,
    parse_open_code_catalog,
)
from tests.inference_support import collect_anthropic
from tests.providers.request_factory import canonical_request
from tests.providers.support import (
    capture_openai_chat_wire_body,
    immediate_admission,
    make_provider_config,
    reasoning_for,
)


def _config():
    return make_provider_config(
        api_key="test_opencode_key",
        base_url="https://opencode.ai/zen/v1",
        rate_limit=100,
        rate_window=1,
    )


def _catalog_payload(
    models: dict[str, object] | None = None,
    *,
    provider_package: str | None = "@ai-sdk/openai-compatible",
    provider_key: str = "opencode",
) -> dict[str, object]:
    provider: dict[str, object] = {
        "models": models
        or {
            "chat-selector": {
                "id": "chat-upstream",
                "reasoning": False,
            },
            "responses-selector": {
                "id": "responses-upstream",
                "provider": {"npm": "@ai-sdk/openai"},
                "reasoning": True,
            },
        }
    }
    if provider_package is not None:
        provider["npm"] = provider_package
    return {provider_key: provider}


def _catalog_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _request(model: str, **overrides: object) -> MessagesRequest:
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 128,
    }
    payload.update(overrides)
    return MessagesRequest.model_validate(payload)


def _responses_event_stream(text: str) -> str:
    events: tuple[dict[str, object], ...] = (
        {
            "type": "response.output_text.delta",
            "sequence_number": 0,
            "item_id": "item_text",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
            "logprobs": [],
        },
        {
            "type": "response.completed",
            "sequence_number": 1,
            "response": {
                "id": "resp_1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "background": False,
                "billing": {"payer": "developer"},
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "max_output_tokens": 128,
                "max_tool_calls": None,
                "model": "responses-upstream",
                "output": [],
                "parallel_tool_calls": True,
                "previous_response_id": None,
                "prompt_cache_key": None,
                "prompt_cache_retention": None,
                "reasoning": {"effort": None, "summary": None},
                "safety_identifier": None,
                "service_tier": "default",
                "store": False,
                "temperature": 1.0,
                "text": {"format": {"type": "text"}, "verbosity": "medium"},
                "tool_choice": "auto",
                "tools": [],
                "top_logprobs": 0,
                "top_p": 1.0,
                "truncation": "disabled",
                "usage": {
                    "input_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 3,
                },
                "user": None,
            },
        },
    )
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


def _chat_event_stream(text: str) -> str:
    chunks = (
        {
            "id": "chatcmpl_1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "chat-upstream",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl_1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "chat-upstream",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
        },
    )
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + (
        "data: [DONE]\n\n"
    )


def _provider_with_wire_transports(
    payload: dict[str, object],
) -> tuple[OpenCodeProvider, list[httpx2.Request], list[httpx.Request]]:
    generation_requests: list[httpx2.Request] = []
    catalog_requests: list[httpx.Request] = []

    def generation_handler(request: httpx2.Request) -> httpx2.Response:
        generation_requests.append(request)
        if request.url.path.endswith("/responses"):
            body = _responses_event_stream("responses-ok")
        elif request.url.path.endswith("/chat/completions"):
            body = _chat_event_stream("chat-ok")
        else:
            raise AssertionError(f"unexpected generation path {request.url.path}")
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=body,
        )

    def catalog_handler(request: httpx.Request) -> httpx.Response:
        catalog_requests.append(request)
        return httpx.Response(200, json=payload)

    generation_client = AsyncOpenAI(
        api_key="test_opencode_key",
        base_url="https://opencode.ai/zen/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(generation_handler)
        ),
    )
    with patch(
        "free_claude_code.providers.openai_chat.provider.AsyncOpenAI",
        return_value=generation_client,
    ):
        provider = create_opencode_provider(
            "opencode_zen",
            _config(),
            immediate_admission(provider_name="opencode_zen"),
            catalog_client=_catalog_client(catalog_handler),
        )
    return provider, generation_requests, catalog_requests


async def _collect(provider: OpenCodeProvider, model: str, **overrides: object) -> str:
    return "".join(
        await collect_anthropic(
            provider.stream_response(
                canonical_request(_request(model, **overrides)),
                input_tokens=2,
                request_id="req_opencode",
                provider_model=(_request(model, **overrides)).model,
            )
        )
    )


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
def test_client_identifies_as_first_party_opencode_user_agent(
    provider_id: str,
) -> None:
    with (
        patch(
            "free_claude_code.providers.openai_chat.provider.AsyncOpenAI"
        ) as mock_openai,
        patch("httpx.AsyncClient"),
    ):
        create_opencode_provider(
            provider_id,
            _config(),
            immediate_admission(provider_name=provider_id),
        )

    assert mock_openai.call_args.kwargs["default_headers"] == {"User-Agent": "opencode"}


def test_catalog_resolves_package_precedence_status_alias_and_reasoning() -> None:
    snapshot = parse_open_code_catalog(
        _catalog_payload(
            {
                "provider-default": {"id": "provider-default", "reasoning": False},
                "responses-alias": {
                    "id": "actual-responses-id",
                    "provider": {"npm": "@ai-sdk/openai"},
                    "status": "beta",
                    "reasoning": True,
                },
                "anthropic": {
                    "id": "anthropic-id",
                    "provider": {"npm": "@ai-sdk/anthropic"},
                },
                "google": {
                    "id": "google-id",
                    "provider": {"npm": "@ai-sdk/google"},
                },
                "unknown-package": {
                    "id": "unknown-id",
                    "provider": {"npm": "@vendor/future"},
                },
                "alpha": {"id": "alpha-id", "status": "alpha"},
                "deprecated": {"id": "old-id", "status": "deprecated"},
            }
        ),
        provider_key="opencode",
        provider_name="OPENCODE_ZEN",
    )

    assert set(snapshot.routes) == {
        "provider-default",
        "responses-alias",
        "anthropic",
        "google",
        "unknown-package",
    }
    provider_default = snapshot.route("provider-default")
    assert provider_default is not None
    assert provider_default.transport is OpenCodeUpstreamTransport.CHAT_COMPLETIONS
    responses = snapshot.route("responses-alias")
    assert responses is not None
    assert responses.upstream_model_id == "actual-responses-id"
    assert responses.transport is OpenCodeUpstreamTransport.RESPONSES
    assert responses.supports_thinking is True
    for selector in ("anthropic", "google", "unknown-package"):
        route = snapshot.route(selector)
        assert route is not None
        assert route.transport is OpenCodeUpstreamTransport.CHAT_COMPLETIONS
    assert snapshot.model_infos == frozenset(
        {
            ProviderModelInfo("provider-default", supports_thinking=False),
            ProviderModelInfo("responses-alias", supports_thinking=True),
            ProviderModelInfo("anthropic"),
            ProviderModelInfo("google"),
            ProviderModelInfo("unknown-package"),
        }
    )


def test_catalog_defaults_missing_package_to_chat_completions() -> None:
    snapshot = parse_open_code_catalog(
        _catalog_payload(
            {"defaulted": {"id": "upstream"}},
            provider_package=None,
        ),
        provider_key="opencode",
        provider_name="OPENCODE_ZEN",
    )

    route = snapshot.route("defaulted")
    assert route is not None
    assert route.transport is OpenCodeUpstreamTransport.CHAT_COMPLETIONS


@pytest.mark.parametrize(
    "payload,match",
    [
        ({}, "provider section"),
        ({"opencode": {"models": []}}, "models object"),
        (_catalog_payload({" ": {"id": "x"}}), "model selector"),
        (_catalog_payload({"x": {"id": " "}}), "model 'x' id"),
        (_catalog_payload({"x": {"id": "x", "status": "future"}}), "status"),
        (_catalog_payload({"x": {"id": "x", "reasoning": "yes"}}), "reasoning"),
        (
            _catalog_payload({"x": {"id": "x", "provider": {"npm": " "}}}),
            "provider npm",
        ),
        (
            _catalog_payload({"x": {"id": "x", "status": "deprecated"}}),
            "no active or beta models",
        ),
    ],
)
def test_catalog_rejects_malformed_route_critical_data(
    payload: object,
    match: str,
) -> None:
    with pytest.raises(ModelListResponseError, match=match):
        parse_open_code_catalog(
            payload,
            provider_key="opencode",
            provider_name="OPENCODE_ZEN",
        )


def test_catalog_rejects_duplicate_normalized_selectors() -> None:
    with pytest.raises(ModelListResponseError, match="duplicate normalized"):
        parse_open_code_catalog(
            _catalog_payload(
                {
                    "same": {"id": "first"},
                    " same ": {"id": "second"},
                }
            ),
            provider_key="opencode",
            provider_name="OPENCODE_ZEN",
        )


@pytest.mark.asyncio
async def test_cold_catalog_load_is_coalesced_and_never_sends_api_key() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        await asyncio.sleep(0)
        return httpx.Response(200, json=_catalog_payload())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    catalog = OpenCodeCatalog(
        _config(),
        provider_key="opencode",
        provider_name="OPENCODE_ZEN",
        admission=immediate_admission(provider_name="opencode_zen"),
        client=client,
    )
    try:
        first, second = await asyncio.gather(catalog.snapshot(), catalog.snapshot())
    finally:
        await catalog.cleanup()

    assert first is second
    assert len(requests) == 1
    assert str(requests[0].url) == OPENCODE_CATALOG_URL
    assert requests[0].headers["user-agent"] == "opencode"
    assert "authorization" not in requests[0].headers
    assert "test_opencode_key" not in repr(requests[0].headers)


@pytest.mark.asyncio
async def test_refresh_atomically_replaces_snapshot_and_retains_last_good_on_error() -> (
    None
):
    payloads: list[object] = [
        _catalog_payload({"first": {"id": "first-upstream"}}),
        _catalog_payload({"second": {"id": "second-upstream"}}),
        _catalog_payload({"broken": {"id": ""}}),
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payloads.pop(0))

    catalog = OpenCodeCatalog(
        _config(),
        provider_key="opencode",
        provider_name="OPENCODE_ZEN",
        admission=immediate_admission(provider_name="opencode_zen"),
        client=_catalog_client(handler),
    )
    try:
        first = await catalog.refresh()
        second = await catalog.refresh()
        with pytest.raises(ModelListResponseError):
            await catalog.refresh()
        cached = await catalog.snapshot()
    finally:
        await catalog.cleanup()

    assert set(first.routes) == {"first"}
    assert set(second.routes) == {"second"}
    assert cached is second


@pytest.mark.asyncio
async def test_cold_catalog_failure_is_canonical_and_never_guesses_transport() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, json={"error": "busy"})

    catalog = OpenCodeCatalog(
        _config(),
        provider_key="opencode",
        provider_name="OPENCODE_ZEN",
        admission=immediate_admission(
            provider_name="opencode_zen",
            max_attempts=2,
        ),
        client=_catalog_client(handler),
    )
    try:
        with pytest.raises(ExecutionFailure) as exc_info:
            await catalog.snapshot(request_id="req_catalog")
    finally:
        await catalog.cleanup()

    assert attempts == 2
    assert exc_info.value.retryable is True
    assert catalog.current_snapshot is None


@pytest.mark.asyncio
async def test_direct_inference_routes_responses_without_prior_model_listing() -> None:
    provider, generation_requests, catalog_requests = _provider_with_wire_transports(
        _catalog_payload()
    )
    try:
        body = await _collect(provider, "responses-selector")
    finally:
        await provider.cleanup()

    assert len(catalog_requests) == 1
    assert [request.url.path for request in generation_requests] == [
        "/zen/v1/responses"
    ]
    payload = json.loads(generation_requests[0].content)
    assert payload["model"] == "responses-upstream"
    events = parse_sse_text(body)
    assert_anthropic_stream_contract(events)
    assert text_content(events) == "responses-ok"


@pytest.mark.asyncio
async def test_chat_catalog_route_uses_existing_chat_transport_and_upstream_id() -> (
    None
):
    provider, generation_requests, _catalog_requests = _provider_with_wire_transports(
        _catalog_payload()
    )
    try:
        body = await _collect(provider, "chat-selector")
    finally:
        await provider.cleanup()

    assert [request.url.path for request in generation_requests] == [
        "/zen/v1/chat/completions"
    ]
    payload = json.loads(generation_requests[0].content)
    assert payload["model"] == "chat-upstream"
    events = parse_sse_text(body)
    assert_anthropic_stream_contract(events)
    assert text_content(events) == "chat-ok"


@pytest.mark.asyncio
async def test_unknown_model_rejects_before_either_generation_endpoint() -> None:
    provider, generation_requests, catalog_requests = _provider_with_wire_transports(
        _catalog_payload()
    )
    try:
        with pytest.raises(InvalidRequestError, match="does not advertise"):
            await _collect(provider, "not-advertised")
    finally:
        await provider.cleanup()

    assert len(catalog_requests) == 1
    assert generation_requests == []


@pytest.mark.asyncio
async def test_model_listing_and_dispatch_use_same_snapshot() -> None:
    provider, generation_requests, catalog_requests = _provider_with_wire_transports(
        _catalog_payload()
    )
    try:
        infos = await provider.list_model_infos()
        body = await _collect(provider, "responses-selector")
    finally:
        await provider.cleanup()

    assert infos == frozenset(
        {
            ProviderModelInfo("chat-selector", supports_thinking=False),
            ProviderModelInfo("responses-selector", supports_thinking=True),
        }
    )
    assert len(catalog_requests) == 1
    assert generation_requests[0].url.path.endswith("/responses")
    assert text_content(parse_sse_text(body)) == "responses-ok"


@pytest.mark.asyncio
async def test_cold_route_specific_conversion_failure_precedes_generation() -> None:
    provider, generation_requests, _catalog_requests = _provider_with_wire_transports(
        _catalog_payload()
    )
    try:
        with pytest.raises(InvalidRequestError, match="stop_sequences"):
            await _collect(
                provider,
                "responses-selector",
                stop_sequences=["done"],
            )
    finally:
        await provider.cleanup()

    assert generation_requests == []


@pytest.mark.asyncio
async def test_warm_preflight_rejects_unknown_and_route_specific_fields() -> None:
    provider, generation_requests, _catalog_requests = _provider_with_wire_transports(
        _catalog_payload()
    )
    try:
        await provider.list_model_infos()
        with pytest.raises(InvalidRequestError, match="does not advertise"):
            provider.preflight_stream(
                canonical_request(_request("missing")),
                provider_model=(_request("missing")).model,
            )
        with pytest.raises(InvalidRequestError, match="stop_sequences"):
            provider.preflight_stream(
                canonical_request(
                    _request("responses-selector", stop_sequences=["done"])
                ),
                provider_model=(
                    _request("responses-selector", stop_sequences=["done"])
                ).model,
            )
    finally:
        await provider.cleanup()

    assert generation_requests == []


@pytest.mark.asyncio
async def test_cleanup_attempts_both_owned_clients_when_generation_close_fails() -> (
    None
):
    provider, _generation_requests, _catalog_requests = _provider_with_wire_transports(
        _catalog_payload()
    )
    generation_close = AsyncMock(side_effect=RuntimeError("generation close failed"))
    catalog_close = AsyncMock()
    provider._client.close = generation_close
    provider._catalog._client.aclose = catalog_close

    with pytest.raises(RuntimeError, match="generation close failed"):
        await provider.cleanup()

    generation_close.assert_awaited_once()
    catalog_close.assert_awaited_once()


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
def test_build_request_body_replays_tool_reasoning_natively(
    provider_id: str,
) -> None:
    with (
        patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"),
        patch("httpx.AsyncClient"),
    ):
        provider = create_opencode_provider(
            provider_id,
            _config(),
            immediate_admission(provider_name=provider_id),
        )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "I should inspect the file.",
                            "signature": "sig",
                        },
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "Read",
                            "input": {"path": "README.md"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "file contents",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled"},
        }
    )

    body = provider._build_request_body(
        canonical_request(request),
        reasoning=reasoning_for(request),
        provider_model=(request).model,
    )

    assistant = body["messages"][0]
    assert assistant["content"] == ""
    assert assistant["reasoning_content"] == "I should inspect the file."
    assert "<think>" not in assistant["content"]
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert body["messages"][1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "file contents",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
async def test_tool_only_history_sends_empty_reasoning_content_on_wire(
    provider_id: str,
) -> None:
    with (
        patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"),
        patch("httpx.AsyncClient"),
    ):
        provider = create_opencode_provider(
            provider_id,
            _config(),
            immediate_admission(provider_name=provider_id),
        )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_missing",
                            "name": "Read",
                            "input": {"path": "README.md"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_missing",
                            "content": "file contents",
                        }
                    ],
                },
            ],
        }
    )

    body = provider._build_request_body(
        canonical_request(request),
        reasoning=reasoning_for(request),
        provider_model=(request).model,
    )
    wire = await capture_openai_chat_wire_body(body)

    assistant = wire["messages"][0]
    assert assistant["content"] == ""
    assert assistant["reasoning_content"] == ""
    assert assistant["tool_calls"][0]["id"] == "call_missing"
    assert assistant["tool_calls"][0]["function"]["name"] == "Read"
    assert wire["messages"][1] == {
        "role": "tool",
        "tool_call_id": "call_missing",
        "content": "file contents",
    }
