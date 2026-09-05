"""Lifecycle ownership through both real Responses backends and client projections."""

import asyncio
import json
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
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY
from free_claude_code.providers.admission import (
    ProviderAttempt,
    ProviderExecution,
    ProviderExecutionState,
    ProviderOperationKind,
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
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("label", ["message", "envelope"])
async def test_responses_payload_type_controls_successful_presentation(
    backend: str,
    native: bool,
    label: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wire = "".join(
        f"event: {label}\ndata: {json.dumps(event)}\n\n"
        for event in [
            response_event("response.created"),
            _text_delta("preserved"),
            response_event("response.completed"),
        ]
    )
    async with harness_for(backend, monkeypatch) as harness:
        harness.bodies.extend(Body(wire) for _ in range(3))
        events = parse_sse_text(await collect(harness.stream(native)))
        assert events[-1].event == ("response.completed" if native else "message_stop")
        assert "preserved" in json.dumps([event.data for event in events])
        assert len(harness.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sdk", "codex"])
async def test_first_frame_rejection_does_not_accept_provider_recovery(
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
        assert len(accepted) == 1
        assert len(harness.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sdk", "codex"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "details,kind,status,retryable",
    [
        ({"status": 401}, FailureKind.AUTHENTICATION, 401, False),
        (
            {"code": None, "type": "authentication_error"},
            FailureKind.AUTHENTICATION,
            401,
            False,
        ),
        ({"status_code": "403"}, FailureKind.PERMISSION, 403, False),
        ({"code": 429}, FailureKind.RATE_LIMIT, 429, True),
        (
            {"status": 429, "type": "rate_limit_error", "code": "quota_exceeded"},
            FailureKind.RATE_LIMIT,
            429,
            True,
        ),
        (
            {"status": 413, "code": "rate_limit_exceeded"},
            FailureKind.INVALID_REQUEST,
            413,
            False,
        ),
        (
            {"status": 429, "code": "context_length_exceeded"},
            FailureKind.CONTEXT_WINDOW_EXCEEDED,
            400,
            False,
        ),
        ({"code": 402}, FailureKind.PERMISSION, 402, False),
        (
            {"status": 400, "type": "server_error"},
            FailureKind.INVALID_REQUEST,
            400,
            False,
        ),
        ({"code": "unknown_error"}, FailureKind.UPSTREAM, 502, False),
        ({"code": "overloaded_error"}, FailureKind.OVERLOADED, 529, True),
    ],
)
async def test_structured_failure_keeps_classification_and_safe_diagnostics(
    backend: str,
    native: bool,
    details: JsonObject,
    kind: FailureKind,
    status: int,
    retryable: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with harness_for(backend, monkeypatch) as harness:
        payload = auth_failure()
        response = payload["response"]
        assert isinstance(response, dict)
        response["error"] = {
            **details,
            "message": "upstream rejected",
            "api_key": "SECRET",
        }
        payload["request_id"] = "upstream-diagnostic-id"
        harness.bodies.extend(Body(_sse(payload)) for _ in range(3))

        # A presentation adapter may rewrite or even discard the native envelope.
        def rewrite(event_type: str, data: JsonObject) -> JsonObject:
            data.clear()
            return {"type": event_type}

        monkeypatch.setattr(
            responses_events.ResponsesEventSource,
            "normalize",
            lambda self, event_type, data: rewrite(event_type, data),
        )
        with pytest.raises(ExecutionFailure) as caught:
            await collect(harness.stream(native))
        failure = caught.value
        assert (failure.kind, failure.status_code, failure.retryable) == (
            kind,
            status,
            retryable,
        )
        assert "upstream-diagnostic-id" in failure.message
        assert "SECRET" not in failure.message and "<redacted>" in failure.message
        refresh = backend == "sdk" and status in {401, 403}
        assert len(harness.requests) == (3 if retryable else 2 if refresh else 1)
        if backend == "sdk":
            assert harness.context.calls == (
                [False, True] if refresh else [False] * len(harness.requests)
            )
        else:
            assert not harness.context.calls
        assert harness.auth.recovery_calls == 0
        assert all(body.closed for body in harness.bodies[: len(harness.requests)])


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sdk", "codex"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("phase", ["first_frame", "held", "committed"])
@pytest.mark.parametrize(
    "error_name", ["ReadTimeout", "ReadError", "RemoteProtocolError"]
)
async def test_transport_failure_retries_only_before_commit(
    backend: str,
    native: bool,
    phase: str,
    error_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http = httpx2 if backend == "sdk" else httpx
    error = getattr(http, error_name)("connection lost")
    committed = phase == "committed"

    class BrokenBody(Body):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield self.text.encode()
            raise error

    async with harness_for(backend, monkeypatch) as harness:
        harness.bodies.extend(
            [
                BrokenBody(
                    ""
                    if phase == "first_frame"
                    else _sse(
                        response_event("response.created"),
                        _text_delta("x" * 70_000 if committed else "hidden"),
                    )
                ),
                Body(
                    _sse(
                        _text_delta("replacement"), response_event("response.completed")
                    )
                ),
            ]
        )
        if committed and not native:
            with pytest.raises(ExecutionFailure) as caught:
                await collect(harness.stream(native))
            assert caught.value.retryable
            assert caught.value.__cause__ is error
        else:
            events = parse_sse_text(await collect(harness.stream(native)))
            assert events[-1].event == (
                "response.failed"
                if committed
                else "response.completed"
                if native
                else "message_stop"
            )
        assert len(harness.requests) == (1 if committed else 2)
        assert harness.bodies[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sdk", "codex"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
async def test_sse_framing_preserves_unicode_across_single_byte_chunks(
    backend: str,
    native: bool,
    newline: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = "A\u0085B\u2028C\u2029D"

    class FragmentedBody(Body):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            for byte in self.text.encode():
                yield bytes([byte])

    # Literal Unicode and split CRLF/UTF-8 bytes exercise framing, not JSON escapes.
    arguments = json.dumps({"value": value}, ensure_ascii=False)
    payloads = [
        response_event("response.created"),
        _text_delta(value),
        {"type": "response.reasoning_summary_text.delta", "delta": value},
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "type": "function_call",
                "id": "fc_unicode",
                "call_id": "call_unicode",
                "name": "inspect",
                "arguments": "",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "item_id": "fc_unicode",
            "output_index": 1,
            "delta": arguments,
        },
        response_event("response.completed"),
    ]
    wire = "\ufeff: comment" + newline + newline
    wire += "".join(
        "data: " + json.dumps(event, ensure_ascii=False) + newline * 2
        for event in payloads
    )
    async with harness_for(backend, monkeypatch) as harness:
        harness.bodies.extend(FragmentedBody(wire) for _ in range(3))
        events = parse_sse_text(await collect(harness.stream(native)))
        deltas = [event.data.get("delta") for event in events]
        assert (
            value in deltas
            if native
            else {"type": "text_delta", "text": value} in deltas
        )
        if native:
            assert deltas.count(value) == 2 and arguments in deltas
        else:
            assert {"type": "thinking_delta", "thinking": value} in deltas
            assert {"type": "input_json_delta", "partial_json": arguments} in deltas
        assert len(harness.requests) == 1 and harness.bodies[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sdk", "codex"])
async def test_rejected_probe_does_not_release_waiters_as_success(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with harness_for(backend, monkeypatch) as harness:
        admission = harness.provider._admission
        owner = admission.start_execution()
        initial = await owner.open_attempt(ProviderOperationKind.GENERATION)
        await initial.fail(httpx.ReadError("offline"))
        await initial.aclose()
        await owner.aclose()
        failure = {
            "type": "error",
            "error": {"code": "rate_limit_exceeded", "message": "retry shortly"},
        }
        harness.bodies.extend(Body(_sse(failure)) for _ in range(3))
        with pytest.raises(ExecutionFailure):
            await collect(harness.stream(True))
        assert admission._episode is not None
        assert admission._episode.terminal_until is not None
        assert len(harness.requests) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sdk", "codex"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("complete", [False, True])
async def test_attempt_release_finishes_parser_and_physical_response(
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

    monkeypatch.setattr(responses_events, "_decode_sse", record)
    async with harness_for(backend, monkeypatch) as harness:
        body = Body(
            _sse(response_event("response.completed"))
            if complete
            else _sse(response_event("response.created"), _text_delta("x" * 70_000))
        )
        harness.bodies.append(body)
        release = harness.provider._admission._release_concurrency

        def release_slot() -> None:
            # httpx2 closes its physical response while unwinding the byte
            # iterator. Both lifetimes must end before the attempt releases.
            assert body.closed
            assert parsers and all(parser.ag_frame is None for parser in parsers)
            release()

        monkeypatch.setattr(
            harness.provider._admission, "_release_concurrency", release_slot
        )
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
