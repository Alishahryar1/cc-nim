"""Both OpenAI egresses borrow isolated request credentials and one retry budget."""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY
from free_claude_code.providers.admission import (
    ProviderExecution,
    ProviderExecutionState,
)
from free_claude_code.providers.endpoint import HttpEndpoint
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
)
from free_claude_code.providers.openai_responses import OpenAIResponsesTransport
from tests.providers.support import immediate_admission, make_provider_config


class Context:
    def __init__(
        self, name: str, *, token: str | None = "key", authorization: str | None = None
    ) -> None:
        self.name = name
        self.token = token
        self.authorization = authorization
        self.calls: list[bool] = []

    async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint:
        self.calls.append(force_refresh)
        headers = {"X-Session": self.name}
        if self.authorization:
            headers["authorization"] = self.authorization
        return HttpEndpoint(
            f"https://{self.name}.invalid/",
            headers,
            f"{self.token}-fresh" if force_refresh and self.token else self.token,
        )


def _response(request: httpx2.Request) -> httpx2.Response:
    if request.url.path.endswith("/responses"):
        payload = {
            "type": "response.completed",
            "response": {
                "id": "resp",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "upstream",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
        }
    else:
        payload = {
            "id": "chat",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "upstream",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text=f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n",
    )


def _transport(
    client: AsyncOpenAI, *, responses: bool, pool: httpx2.AsyncBaseTransport
) -> OpenAIChatProvider | OpenAIResponsesTransport:
    admission = immediate_admission(max_attempts=3)
    if responses:
        return OpenAIResponsesTransport(
            client=client,
            admission=admission,
            provider_name="TEST",
            read_timeout_s=3,
            log_raw_sse_events=False,
            endpoint_transport=pool,
        )
    return OpenAIChatProvider(
        make_provider_config(None, "https://original.invalid"),
        profile=OpenAIChatProfile(
            OpenAIChatRequestPolicy("TEST", ReasoningReplayMode.REASONING_CONTENT),
            NO_REASONING,
        ),
        admission=admission,
        client=client,
        endpoint_transport=pool,
    )


def _stream(
    transport: OpenAIChatProvider | OpenAIResponsesTransport,
    context: Context,
    *,
    responses_ingress: bool,
) -> AsyncIterator[str]:
    if responses_ingress:
        return transport.stream_responses(
            OpenAIResponsesRequest(model="upstream", input="hi"),
            input_tokens=1,
            request_id=None,
            response_model="public",
            reasoning=DEFAULT_REASONING_POLICY,
            endpoint_context=context,
        )
    return transport.stream_messages(
        MessagesRequest(model="upstream", messages=[{"role": "user", "content": "hi"}]),
        input_tokens=1,
        request_id=None,
        response_model="public",
        reasoning=DEFAULT_REASONING_POLICY,
        endpoint_context=context,
    )


async def _consume(stream: AsyncIterator[str]) -> None:
    assert [event async for event in stream]


@pytest.mark.asyncio
@pytest.mark.parametrize("force_failure", [False, True])
async def test_responses_credential_failure_is_outside_generation_budget(
    force_failure: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    executions: list[ProviderExecution] = []

    class FailingContext(Context):
        async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint:
            endpoint = await super().endpoint(force_refresh=force_refresh)
            if force_refresh or not force_failure:
                raise httpx.ReadError("credential service unavailable")
            return endpoint

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(401, json={"error": {"message": "expired"}})

    pool = httpx2.MockTransport(handler)
    async with AsyncOpenAI(
        api_key="base", http_client=httpx2.AsyncClient(transport=pool)
    ) as client:
        transport = _transport(client, responses=True, pool=pool)
        start_execution = transport._admission.start_execution

        def record_execution(*, request_id: str | None = None) -> ProviderExecution:
            execution = start_execution(request_id=request_id)
            executions.append(execution)
            return execution

        monkeypatch.setattr(transport._admission, "start_execution", record_execution)
        context = FailingContext("a")
        with pytest.raises(ExecutionFailure):
            await _consume(_stream(transport, context, responses_ingress=True))
    assert calls == int(force_failure)
    assert context.calls == ([False, True] if force_failure else [False])
    assert len(executions) == 1
    assert executions[0].attempts_started == int(force_failure)
    assert executions[0].state is ProviderExecutionState.FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("forced_refresh", [False, True])
async def test_preparation_exit_releases_recovery_for_other_requests(
    cancel: bool,
    forced_refresh: bool,
) -> None:
    calls = 0
    preparing = asyncio.Event()
    finish_preparation = asyncio.Event()
    healthy_prepared = asyncio.Event()

    class InterruptedContext(Context):
        async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint:
            endpoint = await super().endpoint(force_refresh=force_refresh)
            if force_refresh if forced_refresh else len(self.calls) == 2:
                preparing.set()
                await finish_preparation.wait()
                raise httpx.ReadError("credential service unavailable")
            return endpoint

    class HealthyContext(Context):
        async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint:
            endpoint = await super().endpoint(force_refresh=force_refresh)
            healthy_prepared.set()
            return endpoint

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx2.Response(503, json={"error": {"message": "unavailable"}})
        if forced_refresh and calls == 2:
            return httpx2.Response(401, json={"error": {"message": "expired"}})
        return _response(request)

    pool = httpx2.MockTransport(handler)
    async with AsyncOpenAI(
        api_key="base", http_client=httpx2.AsyncClient(transport=pool)
    ) as client:
        transport = _transport(client, responses=True, pool=pool)
        first = asyncio.create_task(
            _consume(
                _stream(transport, InterruptedContext("a"), responses_ingress=True)
            )
        )
        async with asyncio.timeout(2):
            await preparing.wait()
            waiting = asyncio.create_task(
                _consume(
                    _stream(
                        transport, HealthyContext("healthy"), responses_ingress=True
                    )
                )
            )
            await healthy_prepared.wait()
            assert not waiting.done()
            if cancel:
                first.cancel()
            else:
                finish_preparation.set()
            with pytest.raises(asyncio.CancelledError if cancel else ExecutionFailure):
                await first
            await waiting
            await _consume(
                _stream(transport, Context("healthy"), responses_ingress=True)
            )
    assert calls == (4 if forced_refresh else 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("responses_ingress", [False, True])
async def test_concurrent_endpoint_views_do_not_share_headers_or_close_base(
    responses: bool, responses_ingress: bool
) -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return _response(request)

    pool = httpx2.MockTransport(handler)
    async with AsyncOpenAI(
        api_key="base-secret",
        organization="old-org",
        project="old-project",
        base_url="https://original.invalid",
        default_headers={"X-Old": "base"},
        default_query={"old": "secret"},
        http_client=httpx2.AsyncClient(
            transport=pool,
            auth=("old-user", "old-password"),
            headers={"X-Secret": "old"},
            params={"old-http": "secret"},
            cookies={"old-cookie": "secret"},
        ),
    ) as client:
        transport = _transport(client, responses=responses, pool=pool)
        a = Context("a", authorization="Bearer session-a")
        b = Context("b", token=None)
        await asyncio.gather(
            _consume(_stream(transport, a, responses_ingress=responses_ingress)),
            _consume(_stream(transport, b, responses_ingress=responses_ingress)),
        )
        for request in seen:
            assert "X-Old" not in request.headers and not request.url.query
            assert "X-Secret" not in request.headers and "Cookie" not in request.headers
            assert (
                "OpenAI-Organization" not in request.headers
                and "OpenAI-Project" not in request.headers
            )
            if request.url.host == "a.invalid":
                assert request.headers.get_list("authorization") == ["Bearer session-a"]
                assert request.headers["X-Session"] == "a"
            else:
                assert "authorization" not in request.headers
                assert request.headers["X-Session"] == "b"
        assert len(seen) == 2
        assert client.api_key == "base-secret" and client.organization == "old-org"
        if isinstance(transport, OpenAIChatProvider):
            await transport.cleanup()
        assert not client.is_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("responses_ingress", [False, True])
@pytest.mark.parametrize("status", [401, 403])
async def test_http_auth_refresh_is_request_scoped_and_bounded(
    responses: bool, responses_ingress: bool, status: int
) -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            assert request.headers["Authorization"] == "Bearer key"
            return httpx2.Response(status, json={"error": {"message": "expired"}})
        assert request.headers["Authorization"] == "Bearer key-fresh"
        return _response(request)

    pool = httpx2.MockTransport(handler)
    async with AsyncOpenAI(
        api_key="base",
        base_url="https://unused.invalid",
        http_client=httpx2.AsyncClient(transport=pool),
    ) as client:
        context = Context("a")
        await _consume(
            _stream(
                _transport(client, responses=responses, pool=pool),
                context,
                responses_ingress=responses_ingress,
            )
        )
    assert calls == 2 and context.calls == [False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
async def test_repeated_rejection_cannot_refresh_again_with_budget_remaining(
    responses: bool,
) -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(
            401,
            headers={"x-should-retry": "true"},
            json={"error": {"message": "expired"}},
        )

    pool = httpx2.MockTransport(handler)
    async with AsyncOpenAI(
        api_key="base",
        base_url="https://unused.invalid",
        http_client=httpx2.AsyncClient(transport=pool),
    ) as client:
        context = Context("a")
        with pytest.raises(ExecutionFailure):
            await _consume(
                _stream(
                    _transport(client, responses=responses, pool=pool),
                    context,
                    responses_ingress=True,
                )
            )
    assert calls == 2 and context.calls == [False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("responses_ingress", [False, True])
@pytest.mark.parametrize(
    "responses,wrapped", [(False, True), (True, True), (True, False)]
)
async def test_stream_authentication_refreshes_before_commit(
    responses: bool, responses_ingress: bool, wrapped: bool
) -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        if calls > 1:
            return _response(request)
        failure = {
            "code": "invalid_api_key",
            "type": "authentication_error",
            "message": "expired",
        }
        payload = {"error": failure} if wrapped else {**failure, "type": "error"}
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(payload)}\n\n",
        )

    pool = httpx2.MockTransport(handler)
    async with AsyncOpenAI(
        api_key="base", http_client=httpx2.AsyncClient(transport=pool)
    ) as client:
        context = Context("a")
        await _consume(
            _stream(
                _transport(client, responses=responses, pool=pool),
                context,
                responses_ingress=responses_ingress,
            )
        )
    assert calls == 2 and context.calls == [False, True]
