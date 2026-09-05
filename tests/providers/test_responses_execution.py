"""Lifecycle ownership through both real Responses backends and client projections."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import AsyncGeneratorType

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY
from free_claude_code.providers.admission import (
    ProviderAttempt,
    ProviderExecution,
    ProviderExecutionState,
)
from free_claude_code.providers.http import maybe_await_aclose
from free_claude_code.providers.openai_codex.provider import OpenAICodexProvider
from free_claude_code.providers.openai_responses import OpenAIResponsesTransport
from free_claude_code.providers.openai_responses import events as responses_events
from tests.providers.support import immediate_admission
from tests.providers.test_endpoint_context_transports import Context
from tests.providers.test_openai_codex_provider import _config, _FakeAuth
from tests.providers.test_openai_responses_transport import (
    _completed_response,
    _sse,
    _text_delta,
)


class Body(httpx.AsyncByteStream, httpx2.AsyncByteStream):
    """One wire response with an observable read/close boundary."""

    def __init__(self, text: str, *, block: bool = False, close_error: bool = False):
        self.text = text
        self.block = block
        self.close_error = close_error
        self.read_tail = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.text.encode()
        self.read_tail.set()
        if self.block:
            await asyncio.Future()

    async def aclose(self) -> None:
        self.closed = True
        if self.close_error:
            raise RuntimeError("physical close failed")


@dataclass
class Harness:
    provider: OpenAIResponsesTransport | OpenAICodexProvider
    context: Context
    auth: _FakeAuth
    bodies: list[Body] = field(default_factory=list)
    requests: list[httpx.Request | httpx2.Request] = field(default_factory=list)
    executions: list[ProviderExecution] = field(default_factory=list)

    def stream(self, native: bool) -> AsyncIterator[str]:
        kwargs = (
            {"endpoint_context": self.context}
            if isinstance(self.provider, OpenAIResponsesTransport)
            else {}
        )
        if native:
            return self.provider.stream_responses(
                OpenAIResponsesRequest(model="upstream", input="hello"),
                input_tokens=1,
                request_id="req_lifecycle",
                response_model="public",
                reasoning=DEFAULT_REASONING_POLICY,
                **kwargs,
            )
        return self.provider.stream_messages(
            MessagesRequest(
                model="upstream", messages=[{"role": "user", "content": "hello"}]
            ),
            input_tokens=1,
            request_id="req_lifecycle",
            response_model="public",
            reasoning=DEFAULT_REASONING_POLICY,
            **kwargs,
        )


@asynccontextmanager
async def harness_for(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Harness]:
    admission = immediate_admission(max_attempts=3, max_concurrency=1)
    context, auth = Context("session"), _FakeAuth()

    def wire(request: httpx.Request | httpx2.Request) -> Body:
        index = len(harness.requests)
        if index:
            assert harness.bodies[index - 1].closed
        harness.requests.append(request)
        return harness.bodies[index]

    if backend == "sdk":
        pool = httpx2.MockTransport(
            lambda request: httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, stream=wire(request)
            )
        )
        client = AsyncOpenAI(
            api_key="test",
            http_client=httpx2.AsyncClient(transport=pool),
            max_retries=0,
        )
        provider = OpenAIResponsesTransport(
            client=client,
            endpoint_transport=pool,
            admission=admission,
            provider_name="test",
            read_timeout_s=1,
            log_raw_sse_events=False,
        )
    else:
        client = httpx.AsyncClient(
            base_url=_config().base_url,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=wire(request),
                )
            ),
        )
        provider = OpenAICodexProvider(
            _config(), client=client, auth=auth, admission=admission
        )
    harness = Harness(provider, context, auth)
    start_execution = admission.start_execution

    def record(*, request_id: str | None = None) -> ProviderExecution:
        execution = start_execution(request_id=request_id)
        harness.executions.append(execution)
        return execution

    monkeypatch.setattr(admission, "start_execution", record)
    try:
        yield harness
    finally:
        if isinstance(client, AsyncOpenAI):
            await client.close()
        else:
            await client.aclose()


def response_event(
    event_type: str, response_id: str = "resp_visible"
) -> dict[str, object]:
    return {
        "type": event_type,
        "response": {
            **_completed_response(),
            "id": response_id,
            "status": "completed"
            if event_type == "response.completed"
            else "in_progress",
        },
    }


def auth_failure() -> dict[str, object]:
    return {
        "type": "response.failed",
        "response": {
            **_completed_response(),
            "id": "resp_visible",
            "status": "failed",
            "error": {"code": "invalid_api_key", "message": "expired"},
        },
    }


async def collect(stream: AsyncIterator[str]) -> str:
    return "".join([chunk async for chunk in stream])


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sdk", "codex"])
async def test_decoded_error_accepts_attempt_before_classification(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepted: list[ProviderAttempt] = []
    accept = ProviderAttempt.accept

    async def record(attempt: ProviderAttempt) -> None:
        await accept(attempt)
        accepted.append(attempt)

    monkeypatch.setattr(ProviderAttempt, "accept", record)
    async with harness_for(backend, monkeypatch) as harness:
        harness.bodies.append(
            Body(
                _sse(
                    {
                        "type": "error",
                        "error": {
                            "code": "server_error",
                            "message": "try again",
                        },
                    }
                )
            )
        )
        harness.bodies.append(Body(_sse(response_event("response.completed"))))
        await collect(harness.stream(True))
        assert len(accepted) == 2
        assert len(harness.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sdk", "codex"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("complete", [False, True])
async def test_parser_finishes_before_physical_response_release(
    backend: str, native: bool, complete: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    parsers: list[AsyncGeneratorType] = []
    decode = responses_events._decode_sse

    def record(
        response: httpx.Response | httpx2.Response,
    ) -> AsyncGenerator[tuple[str, JsonObject]]:
        parser = decode(response)
        assert isinstance(parser, AsyncGeneratorType)
        parsers.append(parser)
        return parser

    class ParserAwareBody(Body):
        async def aclose(self) -> None:
            assert parsers and all(parser.ag_frame is None for parser in parsers)
            await super().aclose()

    monkeypatch.setattr(responses_events, "_decode_sse", record)
    async with harness_for(backend, monkeypatch) as harness:
        body = ParserAwareBody(
            _sse(response_event("response.completed"))
            if complete
            else _sse(response_event("response.created"), _text_delta("x" * 70_000))
        )
        harness.bodies.append(body)
        stream = harness.stream(native)
        if complete:
            await collect(stream)
        else:
            await anext(stream)
            await maybe_await_aclose(stream)
        assert body.closed
        assert len(parsers) == 1 and parsers[0].ag_frame is None


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sdk", "codex"])
@pytest.mark.parametrize("native", [False, True])
async def test_terminal_ends_without_reading_tail_and_releases_permit(
    backend: str, native: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with harness_for(backend, monkeypatch) as harness:
        first = Body(_sse(response_event("response.completed")), block=True)
        harness.bodies.extend([first, Body(first.text)])
        async with asyncio.timeout(2):
            await collect(harness.stream(native))
            await collect(harness.stream(native))
        assert first.closed and not first.read_tail.is_set()
        assert all(
            execution.state is ProviderExecutionState.SUCCEEDED
            for execution in harness.executions
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sdk", "codex"])
@pytest.mark.parametrize("native", [False, True])
async def test_hidden_retry_replaces_attempt_identity_and_keeps_logical_session(
    backend: str, native: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with harness_for(backend, monkeypatch) as harness:
        harness.bodies.extend(
            [
                Body(_sse(response_event("response.created", "resp_hidden"))),
                Body(
                    _sse(
                        response_event("response.created"),
                        response_event("response.completed"),
                    )
                ),
            ]
        )
        text = await collect(harness.stream(native))
        assert "resp_hidden" not in text
        events = parse_sse_text(text)
        assert (
            sum(
                event.event == ("response.created" if native else "message_start")
                for event in events
            )
            == 1
        )
        if native:
            assert all(
                event.data["response"]["id"] == "resp_visible" for event in events
            )
        assert len(harness.requests) == 2
        if backend == "codex":
            assert (
                harness.requests[0].headers["session_id"]
                == harness.requests[1].headers["session_id"]
            )
        (execution,) = harness.executions
        assert execution.attempts_started == 2
        assert execution.state is ProviderExecutionState.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sdk", "codex"])
@pytest.mark.parametrize("native", [False, True])
async def test_committed_auth_failure_cannot_replay_or_refresh_and_records_failure(
    backend: str, native: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with harness_for(backend, monkeypatch) as harness:
        harness.bodies.append(
            Body(
                _sse(
                    response_event("response.created"),
                    _text_delta("x" * 70_000),
                    auth_failure(),
                )
            )
        )
        chunks: list[str] = []

        async def consume() -> None:
            async for chunk in harness.stream(native):
                assert chunk
                chunks.append(chunk)

        if native:
            await consume()
            events = parse_sse_text("".join(chunks))
            assert sum(event.event == "response.failed" for event in events) == 1
            assert events[-1].data["response"]["id"] == "resp_visible"
        else:
            with pytest.raises(ExecutionFailure):
                await consume()
            assert parse_sse_text("".join(chunks))[-1].event == "content_block_stop"
        assert len(harness.requests) == 1 and harness.bodies[0].closed
        assert harness.auth.recovery_calls == 0
        assert harness.context.calls == ([False] if backend == "sdk" else [])
        assert harness.executions[0].state is ProviderExecutionState.FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sdk", "codex"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_consumer_exit_closes_physical_stream_and_abandons_operation(
    backend: str, native: bool, cancel: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with harness_for(backend, monkeypatch) as harness:
        first = Body(
            _sse(
                response_event("response.created"),
                _text_delta("x" if cancel else "x" * 70_000),
            ),
            block=True,
        )
        harness.bodies.extend([first, Body(_sse(response_event("response.completed")))])
        stream = harness.stream(native)
        async with asyncio.timeout(2):
            if cancel:
                task = asyncio.create_task(collect(stream))
                await first.read_tail.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await anext(stream)
                await maybe_await_aclose(stream)
            assert first.closed
            assert harness.executions[0].state is ProviderExecutionState.ABANDONED
            await collect(harness.stream(native))
        assert harness.executions[1].state is ProviderExecutionState.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sdk", "codex"])
@pytest.mark.parametrize("native", [False, True])
async def test_close_failure_preserves_success_and_permit_reuse(
    backend: str, native: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with harness_for(backend, monkeypatch) as harness:
        body = _sse(response_event("response.completed"))
        harness.bodies.extend([Body(body, close_error=True), Body(body)])
        async with asyncio.timeout(2):
            await collect(harness.stream(native))
            await collect(harness.stream(native))
        assert all(
            execution.state is ProviderExecutionState.SUCCEEDED
            for execution in harness.executions
        )
