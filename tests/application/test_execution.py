"""Application-owned provider execution contracts."""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.routing import ResolvedModel, RoutedMessagesRequest
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.async_iterators import AsyncCloseable
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.reasoning import ReasoningPolicy


class FakeProvider:
    def __init__(self) -> None:
        self.preflight_calls: list[tuple[MessagesRequest, ReasoningPolicy]] = []
        self.stream_calls: list[dict[str, object]] = []
        self.stream_close_calls = 0

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> None:
        self.preflight_calls.append((request, reasoning))

    async def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        self._record_stream_call(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
        )
        try:
            yield "event: message_stop\ndata: {}\n\n"
        finally:
            self.stream_close_calls += 1

    def _record_stream_call(
        self,
        request: MessagesRequest,
        *,
        input_tokens: int,
        request_id: str | None,
        response_model: str | None,
        reasoning: ReasoningPolicy,
    ) -> None:
        self.stream_calls.append(
            {
                "request": request,
                "input_tokens": input_tokens,
                "request_id": request_id,
                "response_model": response_model,
                "reasoning": reasoning,
            }
        )


class WaitStep:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()


class EmptyForever:
    pass


EMPTY_FOREVER = EmptyForever()
type StreamStep = str | WaitStep | EmptyForever | BaseException


class ControlledProvider(FakeProvider):
    def __init__(self, steps: list[StreamStep]) -> None:
        super().__init__()
        self._steps = steps

    async def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        self._record_stream_call(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
        )
        try:
            for step in self._steps:
                if isinstance(step, str):
                    yield step
                elif isinstance(step, WaitStep):
                    step.started.set()
                    await step.release.wait()
                elif isinstance(step, EmptyForever):
                    while True:
                        yield ""
                else:
                    raise step
        finally:
            self.stream_close_calls += 1


class FailingPreflightProvider(FakeProvider):
    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> None:
        raise ValueError("invalid provider request")


class FailingStreamConstructionProvider(FakeProvider):
    def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        raise RuntimeError("stream construction failed")


def _routed_request() -> RoutedMessagesRequest:
    request = MessagesRequest(
        model="provider-model",
        messages=[Message(role="user", content="hello")],
    )
    return RoutedMessagesRequest(
        request=request,
        resolved=ResolvedModel(
            original_model="gateway-model",
            provider_id="provider",
            provider_model="provider-model",
            provider_model_ref="provider/provider-model",
            reasoning_preference=ReasoningPreference.CLIENT,
        ),
        reasoning=ReasoningPolicy.on(),
    )


def _executor_stream(
    provider: FakeProvider,
    *,
    timeout_seconds: float,
    request_id: str,
) -> AsyncIterator[str]:
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        token_counter=lambda _messages, _system, _tools: 17,
        progress_timeout_seconds=timeout_seconds,
    )
    return executor.stream(
        _routed_request(),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id=request_id,
    )


@pytest.mark.asyncio
async def test_executor_uses_structural_provider_port_and_preflights_eagerly() -> None:
    provider = FakeProvider()
    routed = _routed_request()
    request = routed.request
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        progress_timeout_seconds=60.0,
        token_counter=lambda _messages, _system, _tools: 17,
    )

    stream = executor.stream(
        routed,
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload=request.model_dump(),
        request_id="req_application",
    )

    assert provider.preflight_calls == [(request, ReasoningPolicy.on())]
    assert [chunk async for chunk in stream] == ["event: message_stop\ndata: {}\n\n"]
    assert provider.stream_calls == [
        {
            "request": request,
            "input_tokens": 17,
            "request_id": "req_application",
            "response_model": "gateway-model",
            "reasoning": ReasoningPolicy.on(),
        }
    ]
    assert provider.stream_close_calls == 1


@pytest.mark.asyncio
async def test_closing_executor_stream_closes_provider_stream_once() -> None:
    provider = FakeProvider()
    routed = _routed_request()
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        progress_timeout_seconds=60.0,
        token_counter=lambda _messages, _system, _tools: 17,
    )
    stream = executor.stream(
        routed,
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_early_close",
    )

    assert await anext(stream) == "event: message_stop\ndata: {}\n\n"
    assert isinstance(stream, AsyncCloseable)
    await stream.aclose()

    assert provider.stream_close_calls == 1


@pytest.mark.asyncio
async def test_stream_construction_failure_remains_deferred_to_iteration() -> None:
    provider = FailingStreamConstructionProvider()
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        progress_timeout_seconds=60.0,
        token_counter=lambda _messages, _system, _tools: 17,
    )

    stream = executor.stream(
        _routed_request(),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_deferred_construction",
    )

    with pytest.raises(RuntimeError, match="stream construction failed"):
        await anext(stream)


def test_executor_preflight_failure_stays_before_token_count_and_stream() -> None:
    provider = FailingPreflightProvider()
    token_counter = MagicMock(return_value=17)
    executor = ProviderExecutor(
        lambda _provider_id: provider,
        progress_timeout_seconds=60.0,
        token_counter=token_counter,
    )

    with pytest.raises(ValueError, match="invalid provider request"):
        executor.stream(
            _routed_request(),
            wire_api="messages",
            raw_log_label="FULL_PAYLOAD",
            raw_log_payload={},
            request_id="req_application",
        )

    token_counter.assert_not_called()
    assert provider.stream_calls == []


@pytest.mark.parametrize(
    "timeout_seconds",
    [0.0, -1.0, float("inf"), float("nan")],
)
def test_executor_rejects_invalid_progress_timeout(timeout_seconds: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        ProviderExecutor(
            lambda _provider_id: FakeProvider(),
            progress_timeout_seconds=timeout_seconds,
        )


@pytest.mark.asyncio
async def test_progress_timeout_before_first_chunk_is_canonical_and_correlated() -> (
    None
):
    provider = ControlledProvider([WaitStep()])
    request_id = "req_progress_timeout"

    with patch("free_claude_code.application.execution.trace_event") as trace_mock:
        stream = _executor_stream(
            provider,
            timeout_seconds=0.02,
            request_id=request_id,
        )
        with pytest.raises(ExecutionFailure) as exc_info:
            await anext(stream)

    failure = exc_info.value
    assert failure.kind == FailureKind.TIMEOUT
    assert failure.status_code == 504
    assert failure.retryable is False
    assert failure.message == (
        "Provider execution made no progress for 0.02 seconds.\n\n"
        f"Request ID: {request_id}"
    )
    assert provider.stream_close_calls == 1
    timeout_traces = [
        call.kwargs
        for call in trace_mock.call_args_list
        if call.kwargs.get("event") == "free_claude_code.provider.progress_timeout"
    ]
    assert timeout_traces == [
        {
            "stage": "execution",
            "event": "free_claude_code.provider.progress_timeout",
            "source": "application",
            "request_id": request_id,
            "provider_id": "provider",
            "timeout_seconds": 0.02,
        }
    ]


@pytest.mark.asyncio
async def test_progress_timeout_renews_after_output_without_crossing_yield() -> None:
    second_wait = WaitStep()
    third_wait = WaitStep()
    provider = ControlledProvider(["first", second_wait, "second", third_wait, "third"])
    stream = _executor_stream(
        provider,
        timeout_seconds=0.02,
        request_id="req_progress_renewal",
    )

    assert await anext(stream) == "first"
    await asyncio.sleep(0.05)
    second_wait.release.set()
    assert await anext(stream) == "second"

    with pytest.raises(ExecutionFailure) as exc_info:
        await anext(stream)

    assert third_wait.started.is_set()
    assert exc_info.value.kind == FailureKind.TIMEOUT
    assert provider.stream_close_calls == 1


@pytest.mark.asyncio
async def test_empty_chunks_do_not_renew_provider_progress() -> None:
    provider = ControlledProvider([EMPTY_FOREVER])
    stream = _executor_stream(
        provider,
        timeout_seconds=0.02,
        request_id="req_empty_progress",
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        await anext(stream)

    assert exc_info.value.kind == FailureKind.TIMEOUT
    assert provider.stream_close_calls == 1


@pytest.mark.asyncio
async def test_provider_owned_timeout_is_not_reclassified() -> None:
    provider = ControlledProvider([TimeoutError("provider-owned timeout")])
    stream = _executor_stream(
        provider,
        timeout_seconds=1.0,
        request_id="req_provider_timeout",
    )

    with pytest.raises(TimeoutError, match="provider-owned timeout"):
        await anext(stream)

    assert provider.stream_close_calls == 1


@pytest.mark.asyncio
async def test_cancelling_progress_wait_remains_cancellation() -> None:
    wait = WaitStep()
    provider = ControlledProvider([wait])
    stream = _executor_stream(
        provider,
        timeout_seconds=1.0,
        request_id="req_cancelled_progress",
    )
    task = asyncio.ensure_future(anext(stream))
    await wait.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider.stream_close_calls == 1
