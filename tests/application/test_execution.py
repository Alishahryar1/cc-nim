"""Application-owned provider execution contracts."""

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.routing import (
    ProviderModelTarget,
    ResolvedModelRoute,
    RoutedMessagesRequest,
)
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.async_iterators import AsyncCloseable
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.inference import InferenceEvent, TextDelta
from free_claude_code.core.reasoning import ReasoningPolicy


def _event(label: str) -> TextDelta:
    return TextDelta(block_id="test-block", delta=label)


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
    ) -> AsyncIterator[InferenceEvent]:
        self._record_stream_call(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
        )
        try:
            yield _event("provider-event")
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


class DelayStep:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds


type StreamStep = TextDelta | WaitStep | DelayStep | EmptyForever | BaseException


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
    ) -> AsyncIterator[InferenceEvent]:
        self._record_stream_call(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
        )
        try:
            for step in self._steps:
                if isinstance(step, TextDelta):
                    yield step
                elif isinstance(step, WaitStep):
                    step.started.set()
                    await step.release.wait()
                elif isinstance(step, DelayStep):
                    await asyncio.sleep(step.seconds)
                elif isinstance(step, EmptyForever):
                    await asyncio.Event().wait()
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


class ApplicationErrorPreflightProvider(FakeProvider):
    def __init__(self, error: InvalidRequestError) -> None:
        super().__init__()
        self._error = error

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> None:
        raise self._error


class FailingStreamConstructionProvider(FakeProvider):
    def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[InferenceEvent]:
        raise RuntimeError("stream construction failed")


def _target(provider_id: str, provider_model: str) -> ProviderModelTarget:
    return ProviderModelTarget(
        provider_id=provider_id,
        provider_model=provider_model,
        provider_model_ref=f"{provider_id}/{provider_model}",
    )


def _routed_request(
    *fallbacks: ProviderModelTarget,
) -> RoutedMessagesRequest:
    request = MessagesRequest(
        model="provider-model",
        messages=[Message(role="user", content="hello")],
    )
    return RoutedMessagesRequest(
        request=request,
        resolved=ResolvedModelRoute(
            original_model="gateway-model",
            primary=_target("provider", "provider-model"),
            fallbacks=fallbacks,
            reasoning_preference=ReasoningPreference.CLIENT,
        ),
        reasoning=ReasoningPolicy.on(),
    )


def _executor_stream(
    provider: FakeProvider,
    *,
    timeout_seconds: float,
    request_id: str,
) -> AsyncIterator[InferenceEvent]:
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
    assert [chunk async for chunk in stream] == [_event("provider-event")]
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


def _execution_failure(
    message: str,
    *,
    retryable: bool = True,
) -> ExecutionFailure:
    return ExecutionFailure(
        kind=FailureKind.OVERLOADED,
        status_code=529,
        message=message,
        retryable=retryable,
    )


@pytest.mark.asyncio
async def test_primary_success_never_resolves_fallback() -> None:
    primary = FakeProvider()
    fallback = FakeProvider()
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return {"provider": primary, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(resolve, progress_timeout_seconds=60.0)
    stream = executor.stream(
        _routed_request(_target("fallback", "fallback-model")),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_primary_success",
    )

    assert [chunk async for chunk in stream] == [_event("provider-event")]
    assert resolved_ids == ["provider"]
    assert fallback.preflight_calls == []
    assert fallback.stream_calls == []


@pytest.mark.asyncio
async def test_retryable_preframe_failure_selects_fallback_after_closing_primary() -> (
    None
):
    order: list[str] = []
    primary = ControlledProvider([_execution_failure("primary overloaded")])
    fallback = ControlledProvider([_event("fallback-frame")])

    def resolve(provider_id: str) -> FakeProvider:
        if provider_id == "fallback":
            assert primary.stream_close_calls == 1
            order.append("fallback-resolved")
        return {"provider": primary, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(
        resolve,
        token_counter=lambda _messages, _system, _tools: 17,
        progress_timeout_seconds=60.0,
        generation_id=7,
    )
    routed = _routed_request(_target("fallback", "fallback-model"))

    with patch("free_claude_code.application.execution.trace_event") as trace_mock:
        chunks = [
            chunk
            async for chunk in executor.stream(
                routed,
                wire_api="messages",
                raw_log_label="FULL_PAYLOAD",
                raw_log_payload={},
                request_id="req_fallback",
            )
        ]

    assert chunks == [_event("fallback-frame")]
    assert order == ["fallback-resolved"]
    assert primary.stream_close_calls == 1
    assert fallback.stream_close_calls == 1
    assert fallback.preflight_calls[0][0].model == "fallback-model"
    assert fallback.stream_calls[0] == {
        "request": fallback.preflight_calls[0][0],
        "input_tokens": 17,
        "request_id": "req_fallback",
        "response_model": "gateway-model",
        "reasoning": ReasoningPolicy.on(),
    }
    events = [call.kwargs for call in trace_mock.call_args_list]
    started = next(
        event
        for event in events
        if event.get("event") == "free_claude_code.model_fallback.started"
    )
    assert started == {
        "stage": "execution",
        "event": "free_claude_code.model_fallback.started",
        "source": "application",
        "request_id": "req_fallback",
        "wire_api": "messages",
        "from_provider_model_ref": "provider/provider-model",
        "to_provider_model_ref": "fallback/fallback-model",
        "candidate_index": 2,
        "candidate_count": 2,
        "failure_kind": "overloaded",
        "status_code": 529,
        "provider_retryable": True,
        "generation_id": 7,
    }
    selected = next(
        event
        for event in events
        if event.get("event") == "free_claude_code.model_fallback.selected"
    )
    assert selected["selected_provider_model_ref"] == "fallback/fallback-model"
    assert selected["candidate_index"] == 2


@pytest.mark.asyncio
async def test_fallback_chain_preserves_exact_last_failure() -> None:
    first = _execution_failure("first")
    second = _execution_failure("second")
    providers = {
        "provider": ControlledProvider([first]),
        "second": ControlledProvider([second]),
    }
    executor = ProviderExecutor(
        providers.__getitem__,
        progress_timeout_seconds=60.0,
    )
    stream = executor.stream(
        _routed_request(_target("second", "model")),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_exhausted",
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        await anext(stream)

    assert exc_info.value is second
    assert all(provider.stream_close_calls == 1 for provider in providers.values())


@pytest.mark.asyncio
async def test_multiple_retryable_failures_walk_fallbacks_in_order() -> None:
    providers = {
        "provider": ControlledProvider([_execution_failure("primary")]),
        "second": ControlledProvider([_execution_failure("second")]),
        "third": ControlledProvider([_event("third-frame")]),
    }
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return providers[provider_id]

    executor = ProviderExecutor(resolve, progress_timeout_seconds=60.0)
    stream = executor.stream(
        _routed_request(
            _target("second", "second-model"),
            _target("third", "third-model"),
        ),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_ordered_fallbacks",
    )

    assert [chunk async for chunk in stream] == [_event("third-frame")]
    assert resolved_ids == ["provider", "second", "third"]
    assert all(provider.stream_close_calls == 1 for provider in providers.values())


@pytest.mark.asyncio
async def test_empty_primary_completion_does_not_select_fallback() -> None:
    primary = ControlledProvider([])
    fallback = FakeProvider()
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return {"provider": primary, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(resolve, progress_timeout_seconds=60.0)
    stream = executor.stream(
        _routed_request(_target("fallback", "model")),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_empty_primary",
    )

    assert [chunk async for chunk in stream] == []
    assert resolved_ids == ["provider"]
    assert primary.stream_close_calls == 1


@pytest.mark.asyncio
async def test_unexpected_primary_failure_does_not_select_fallback() -> None:
    primary = ControlledProvider([RuntimeError("unexpected")])
    fallback = FakeProvider()
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return {"provider": primary, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(resolve, progress_timeout_seconds=60.0)
    stream = executor.stream(
        _routed_request(_target("fallback", "model")),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_unexpected_primary",
    )

    with pytest.raises(RuntimeError, match="unexpected"):
        await anext(stream)

    assert resolved_ids == ["provider"]
    assert primary.stream_close_calls == 1


@pytest.mark.asyncio
async def test_nonretryable_failure_never_resolves_fallback() -> None:
    primary_failure = _execution_failure("rejected", retryable=False)
    primary = ControlledProvider([primary_failure])
    fallback = FakeProvider()
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return {"provider": primary, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(resolve, progress_timeout_seconds=60.0)
    stream = executor.stream(
        _routed_request(_target("fallback", "model")),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_rejected",
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        await anext(stream)

    assert exc_info.value is primary_failure
    assert resolved_ids == ["provider"]


@pytest.mark.asyncio
async def test_failure_after_first_frame_never_selects_fallback() -> None:
    primary_failure = _execution_failure("after frame")
    primary = ControlledProvider([_event("first-frame"), primary_failure])
    fallback = FakeProvider()
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return {"provider": primary, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(resolve, progress_timeout_seconds=60.0)
    stream = executor.stream(
        _routed_request(_target("fallback", "model")),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_committed",
    )

    assert await anext(stream) == _event("first-frame")
    with pytest.raises(ExecutionFailure) as exc_info:
        await anext(stream)

    assert exc_info.value is primary_failure
    assert resolved_ids == ["provider"]


@pytest.mark.asyncio
async def test_lazy_fallback_preflight_application_error_stops_unchanged() -> None:
    primary = ControlledProvider([_execution_failure("primary")])
    preflight_error = InvalidRequestError("fallback is misconfigured")
    fallback = ApplicationErrorPreflightProvider(preflight_error)
    providers = {"provider": primary, "fallback": fallback}
    executor = ProviderExecutor(
        providers.__getitem__,
        progress_timeout_seconds=60.0,
    )
    stream = executor.stream(
        _routed_request(_target("fallback", "model")),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_preflight",
    )

    with pytest.raises(InvalidRequestError) as exc_info:
        await anext(stream)

    assert exc_info.value is preflight_error
    assert primary.stream_close_calls == 1
    assert fallback.stream_calls == []


@pytest.mark.asyncio
async def test_candidate_requests_are_isolated_from_provider_mutation() -> None:
    class MutatingProvider(ControlledProvider):
        async def stream_response(
            self,
            request: MessagesRequest,
            input_tokens: int = 0,
            *,
            request_id: str | None = None,
            response_model: str | None = None,
            reasoning: ReasoningPolicy,
        ) -> AsyncIterator[InferenceEvent]:
            request.messages[0].content = "mutated"
            async for chunk in super().stream_response(
                request,
                input_tokens,
                request_id=request_id,
                response_model=response_model,
                reasoning=reasoning,
            ):
                yield chunk

    primary = MutatingProvider([_execution_failure("primary")])
    fallback = FakeProvider()
    providers = {"provider": primary, "fallback": fallback}
    routed = _routed_request(_target("fallback", "model"))
    executor = ProviderExecutor(
        providers.__getitem__,
        progress_timeout_seconds=60.0,
    )

    assert [
        chunk
        async for chunk in executor.stream(
            routed,
            wire_api="messages",
            raw_log_label="FULL_PAYLOAD",
            raw_log_payload={},
            request_id="req_isolated",
        )
    ]
    fallback_request = fallback.stream_calls[0]["request"]
    assert isinstance(fallback_request, MessagesRequest)
    assert fallback_request.messages[0].content == "hello"
    assert routed.request.messages[0].content == "hello"


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

    assert await anext(stream) == _event("provider-event")
    assert isinstance(stream, AsyncCloseable)
    await stream.aclose()

    assert provider.stream_close_calls == 1


@pytest.mark.asyncio
async def test_stream_construction_failure_remains_deferred_to_iteration() -> None:
    provider = FailingStreamConstructionProvider()
    fallback = FakeProvider()
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return {"provider": provider, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(
        resolve,
        progress_timeout_seconds=60.0,
        token_counter=lambda _messages, _system, _tools: 17,
    )

    stream = executor.stream(
        _routed_request(_target("fallback", "model")),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_deferred_construction",
    )

    with pytest.raises(RuntimeError, match="stream construction failed"):
        await anext(stream)

    assert resolved_ids == ["provider"]


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
    provider = ControlledProvider(
        [_event("first"), second_wait, _event("second"), third_wait, _event("third")]
    )
    stream = _executor_stream(
        provider,
        timeout_seconds=0.02,
        request_id="req_progress_renewal",
    )

    assert await anext(stream) == _event("first")
    await asyncio.sleep(0.05)
    second_wait.release.set()
    assert await anext(stream) == _event("second")

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
async def test_fallback_transition_does_not_reset_shared_progress_deadline() -> None:
    timeout_seconds = 0.2
    primary = ControlledProvider(
        [DelayStep(0.06), _execution_failure("primary overloaded")]
    )
    fallback_wait = WaitStep()
    fallback = ControlledProvider([fallback_wait])
    providers = {"provider": primary, "fallback": fallback}
    executor = ProviderExecutor(
        providers.__getitem__,
        progress_timeout_seconds=timeout_seconds,
    )
    stream = executor.stream(
        _routed_request(_target("fallback", "model")),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_shared_deadline",
    )
    loop = asyncio.get_running_loop()
    started = loop.time()

    with (
        patch("free_claude_code.application.execution.trace_event") as trace_mock,
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        await anext(stream)

    elapsed = loop.time() - started
    assert fallback_wait.started.is_set()
    assert exc_info.value.kind == FailureKind.TIMEOUT
    assert elapsed < 0.24
    assert primary.stream_close_calls == fallback.stream_close_calls == 1
    timeout_trace = next(
        call.kwargs
        for call in trace_mock.call_args_list
        if call.kwargs.get("event") == "free_claude_code.provider.progress_timeout"
    )
    assert timeout_trace["provider_id"] == "fallback"


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
    fallback = FakeProvider()
    resolved_ids: list[str] = []

    def resolve(provider_id: str) -> FakeProvider:
        resolved_ids.append(provider_id)
        return {"provider": provider, "fallback": fallback}[provider_id]

    executor = ProviderExecutor(
        resolve,
        progress_timeout_seconds=1.0,
    )
    stream = executor.stream(
        _routed_request(_target("fallback", "model")),
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_cancelled_progress",
    )
    task = asyncio.ensure_future(anext(stream))
    await wait.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider.stream_close_calls == 1
    assert resolved_ids == ["provider"]
