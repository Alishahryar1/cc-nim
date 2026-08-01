"""Application-owned provider execution contracts."""

from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from free_claude_code.application.execution import ProviderExecutor, TokenCounter
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
        self.stream_calls.append(
            {
                "request": request,
                "input_tokens": input_tokens,
                "request_id": request_id,
                "response_model": response_model,
                "reasoning": reasoning,
            }
        )
        try:
            yield "event: message_stop\ndata: {}\n\n"
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


class ScriptedProvider(FakeProvider):
    """Emit a fixed script, then fail at preflight or mid-stream on demand."""

    def __init__(
        self,
        *,
        chunks: tuple[str, ...] = ("event: message_stop\ndata: {}\n\n",),
        failure: BaseException | None = None,
        preflight_failure: BaseException | None = None,
    ) -> None:
        super().__init__()
        self.chunks = chunks
        self.failure = failure
        self.preflight_failure = preflight_failure

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> None:
        super().preflight_stream(request, reasoning=reasoning)
        if self.preflight_failure is not None:
            raise self.preflight_failure

    async def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        self.stream_calls.append(
            {
                "request": request,
                "input_tokens": input_tokens,
                "request_id": request_id,
                "response_model": response_model,
                "reasoning": reasoning,
            }
        )
        try:
            for chunk in self.chunks:
                yield chunk
            if self.failure is not None:
                raise self.failure
        finally:
            self.stream_close_calls += 1


def _resolved(provider_id: str, provider_model: str) -> ResolvedModel:
    return ResolvedModel(
        original_model="gateway-model",
        provider_id=provider_id,
        provider_model=provider_model,
        provider_model_ref=f"{provider_id}/{provider_model}",
        reasoning_preference=ReasoningPreference.CLIENT,
    )


def _retryable_failure(message: str = "quota exhausted") -> ExecutionFailure:
    return ExecutionFailure(
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message=message,
        retryable=True,
    )


def _routed_request(*, backup: bool = False) -> RoutedMessagesRequest:
    request = MessagesRequest(
        model="provider-model",
        messages=[Message(role="user", content="hello")],
    )
    return RoutedMessagesRequest(
        request=request,
        resolved=_resolved("provider", "provider-model"),
        reasoning=ReasoningPolicy.on(),
        backups=(_resolved("backup-provider", "backup-model"),) if backup else (),
    )


def _failover_executor(
    primary: FakeProvider,
    backup: FakeProvider,
    *,
    token_counter: TokenCounter | None = None,
) -> ProviderExecutor:
    providers: dict[str, FakeProvider] = {
        "provider": primary,
        "backup-provider": backup,
    }
    return ProviderExecutor(
        providers.__getitem__,
        token_counter=token_counter or (lambda _messages, _system, _tools: 17),
    )


def _streamed_request(provider: FakeProvider, index: int = 0) -> MessagesRequest:
    request = provider.stream_calls[index]["request"]
    assert isinstance(request, MessagesRequest)
    return request


def _stream(
    executor: ProviderExecutor, routed: RoutedMessagesRequest
) -> AsyncIterator[str]:
    return executor.stream(
        routed,
        wire_api="messages",
        raw_log_label="FULL_PAYLOAD",
        raw_log_payload={},
        request_id="req_failover",
    )


@pytest.mark.asyncio
async def test_executor_uses_structural_provider_port_and_preflights_eagerly() -> None:
    provider = FakeProvider()
    routed = _routed_request()
    request = routed.request
    executor = ProviderExecutor(
        lambda _provider_id: provider,
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


@pytest.mark.asyncio
async def test_stream_time_failover_serves_the_backup_before_any_chunk() -> None:
    primary = ScriptedProvider(chunks=(), failure=_retryable_failure())
    backup = ScriptedProvider()
    routed = _routed_request(backup=True)
    executor = _failover_executor(primary, backup)

    stream = _stream(executor, routed)

    assert [chunk async for chunk in stream] == ["event: message_stop\ndata: {}\n\n"]
    assert len(primary.stream_calls) == 1
    assert len(backup.stream_calls) == 1
    assert primary.stream_close_calls == 1
    assert backup.stream_close_calls == 1
    # The backup enters through the generator, so it preflights lazily there.
    assert len(backup.preflight_calls) == 1


@pytest.mark.asyncio
async def test_failover_rewrites_the_request_model_for_the_backup() -> None:
    primary = ScriptedProvider(chunks=(), failure=_retryable_failure())
    backup = ScriptedProvider()
    routed = _routed_request(backup=True)

    stream = _stream(_failover_executor(primary, backup), routed)
    [chunk async for chunk in stream]

    backup_request = _streamed_request(backup)
    assert backup_request.model == "backup-model"
    assert backup_request is not routed.request
    assert backup_request.messages is not routed.request.messages
    assert routed.request.model == "provider-model"
    assert primary.stream_calls[0]["request"] is routed.request


@pytest.mark.asyncio
async def test_failover_reuses_the_token_count_and_reasoning_policy() -> None:
    primary = ScriptedProvider(chunks=(), failure=_retryable_failure())
    backup = ScriptedProvider()
    routed = _routed_request(backup=True)
    token_counter = MagicMock(return_value=17)

    stream = _stream(
        _failover_executor(primary, backup, token_counter=token_counter), routed
    )
    [chunk async for chunk in stream]

    token_counter.assert_called_once()
    assert backup.stream_calls[0]["input_tokens"] == 17
    assert backup.stream_calls[0]["reasoning"] is routed.reasoning
    assert backup.preflight_calls[0][1] is routed.reasoning


@pytest.mark.asyncio
async def test_no_failover_once_a_chunk_has_been_yielded() -> None:
    primary = ScriptedProvider(chunks=("first",), failure=_retryable_failure())
    backup = ScriptedProvider()
    executor = _failover_executor(primary, backup)

    stream = _stream(executor, _routed_request(backup=True))

    assert await anext(stream) == "first"
    with pytest.raises(ExecutionFailure) as exc_info:
        await anext(stream)

    assert exc_info.value.kind is FailureKind.RATE_LIMIT
    assert backup.stream_calls == []
    assert backup.preflight_calls == []


@pytest.mark.parametrize(
    ("kind", "status_code"),
    [
        (FailureKind.AUTHENTICATION, 401),
        (FailureKind.PERMISSION, 403),
        (FailureKind.INVALID_REQUEST, 400),
        (FailureKind.CONTEXT_WINDOW_EXCEEDED, 400),
    ],
)
@pytest.mark.asyncio
async def test_terminal_failures_never_fail_over(
    kind: FailureKind, status_code: int
) -> None:
    failure = ExecutionFailure(
        kind=kind,
        status_code=status_code,
        message="terminal failure",
        retryable=False,
    )
    primary = ScriptedProvider(chunks=(), failure=failure)
    backup = ScriptedProvider()
    executor = _failover_executor(primary, backup)

    with pytest.raises(ExecutionFailure) as exc_info:
        [chunk async for chunk in _stream(executor, _routed_request(backup=True))]

    assert exc_info.value.kind is kind
    assert backup.stream_calls == []


@pytest.mark.asyncio
async def test_backup_failure_is_the_error_the_caller_sees() -> None:
    primary = ScriptedProvider(chunks=(), failure=_retryable_failure("primary quota"))
    backup = ScriptedProvider(
        chunks=(),
        failure=ExecutionFailure(
            kind=FailureKind.OVERLOADED,
            status_code=529,
            message="backup overloaded",
            retryable=True,
        ),
    )
    executor = _failover_executor(primary, backup)

    with pytest.raises(ExecutionFailure) as exc_info:
        [chunk async for chunk in _stream(executor, _routed_request(backup=True))]

    assert exc_info.value.kind is FailureKind.OVERLOADED
    assert exc_info.value.message == "backup overloaded"
    assert primary.stream_close_calls == 1
    assert backup.stream_close_calls == 1


@pytest.mark.asyncio
async def test_preflight_failover_swaps_synchronously_without_raising() -> None:
    primary = ScriptedProvider(preflight_failure=_retryable_failure())
    backup = ScriptedProvider()
    routed = _routed_request(backup=True)

    stream = _stream(_failover_executor(primary, backup), routed)

    assert len(backup.preflight_calls) == 1
    assert [chunk async for chunk in stream] == ["event: message_stop\ndata: {}\n\n"]
    # Preflighted once synchronously; the generator must not repeat it.
    assert len(backup.preflight_calls) == 1
    assert primary.stream_calls == []
    assert _streamed_request(backup).model == "backup-model"


def test_terminal_preflight_failure_is_not_failed_over() -> None:
    primary = ScriptedProvider(
        preflight_failure=ExecutionFailure(
            kind=FailureKind.INVALID_REQUEST,
            status_code=400,
            message="unsupported tool shape",
            retryable=False,
        )
    )
    backup = ScriptedProvider()
    executor = _failover_executor(primary, backup)

    with pytest.raises(ExecutionFailure) as exc_info:
        _stream(executor, _routed_request(backup=True))

    assert exc_info.value.kind is FailureKind.INVALID_REQUEST
    assert backup.preflight_calls == []


def test_preflight_failure_without_a_backup_still_raises_synchronously() -> None:
    primary = ScriptedProvider(preflight_failure=_retryable_failure())
    executor = ProviderExecutor(
        lambda _provider_id: primary,
        token_counter=lambda _messages, _system, _tools: 17,
    )

    with pytest.raises(ExecutionFailure):
        _stream(executor, _routed_request())


@pytest.mark.asyncio
async def test_backup_preflight_failure_surfaces_through_the_stream() -> None:
    primary = ScriptedProvider(chunks=(), failure=_retryable_failure())
    backup = ScriptedProvider(preflight_failure=_retryable_failure("backup down"))
    executor = _failover_executor(primary, backup)

    with pytest.raises(ExecutionFailure) as exc_info:
        [chunk async for chunk in _stream(executor, _routed_request(backup=True))]

    assert exc_info.value.message == "backup down"
    assert backup.stream_calls == []


@pytest.mark.asyncio
async def test_stream_failover_traces_the_swap() -> None:
    primary = ScriptedProvider(chunks=(), failure=_retryable_failure())
    backup = ScriptedProvider()
    executor = _failover_executor(primary, backup)

    with patch("free_claude_code.application.execution.trace_event") as trace:
        stream = _stream(executor, _routed_request(backup=True))
        [chunk async for chunk in stream]

    rows = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event") == "free_claude_code.api.failover"
    ]
    assert len(rows) == 1
    assert rows[0]["phase"] == "stream"
    assert rows[0]["from_provider"] == "provider"
    assert rows[0]["to_provider"] == "backup-provider"
    assert rows[0]["gateway_model"] == "gateway-model"
    assert rows[0]["failure_kind"] == "rate_limit"
    assert rows[0]["exc_type"] == "ExecutionFailure"
    assert rows[0]["request_id"] == "req_failover"


@pytest.mark.asyncio
async def test_preflight_failover_traces_the_swap_and_the_serving_route() -> None:
    primary = ScriptedProvider(preflight_failure=_retryable_failure())
    backup = ScriptedProvider()
    executor = _failover_executor(primary, backup)

    with patch("free_claude_code.application.execution.trace_event") as trace:
        stream = _stream(executor, _routed_request(backup=True))
        [chunk async for chunk in stream]

    failover = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event") == "free_claude_code.api.failover"
    ]
    route = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event") == "free_claude_code.api.route.resolved"
    ]
    assert [row["phase"] for row in failover] == ["preflight"]
    assert route[0]["provider_id"] == "backup-provider"
    assert route[0]["provider_model"] == "backup-model"
    assert route[0]["provider_model_ref"] == "backup-provider/backup-model"


@pytest.mark.asyncio
async def test_healthy_primary_never_touches_the_backup() -> None:
    primary = ScriptedProvider()
    backup = ScriptedProvider()
    executor = _failover_executor(primary, backup)

    with patch("free_claude_code.application.execution.trace_event") as trace:
        stream = _stream(executor, _routed_request(backup=True))
        assert [chunk async for chunk in stream] == [
            "event: message_stop\ndata: {}\n\n"
        ]

    assert len(primary.preflight_calls) == 1
    assert backup.preflight_calls == []
    assert backup.stream_calls == []
    assert not [
        call
        for call in trace.call_args_list
        if call.kwargs.get("event") == "free_claude_code.api.failover"
    ]


@pytest.mark.asyncio
async def test_healthy_primary_never_builds_the_backup_request_body() -> None:
    primary = ScriptedProvider()
    backup = ScriptedProvider()
    routed = _routed_request(backup=True)
    executor = _failover_executor(primary, backup)

    with patch.object(MessagesRequest, "model_copy", autospec=True) as model_copy:
        stream = _stream(executor, routed)
        assert [chunk async for chunk in stream] == [
            "event: message_stop\ndata: {}\n\n"
        ]

    model_copy.assert_not_called()


@pytest.mark.asyncio
async def test_the_backup_body_is_built_once_across_preflight_and_stream() -> None:
    primary = ScriptedProvider(chunks=(), failure=_retryable_failure())
    backup = ScriptedProvider()
    routed = _routed_request(backup=True)

    stream = _stream(_failover_executor(primary, backup), routed)
    [chunk async for chunk in stream]

    assert backup.preflight_calls[0][0] is _streamed_request(backup)


def _cascade_routed_request(*backups: tuple[str, str]) -> RoutedMessagesRequest:
    request = MessagesRequest(
        model="provider-model",
        messages=[Message(role="user", content="hello")],
    )
    return RoutedMessagesRequest(
        request=request,
        resolved=_resolved("provider", "provider-model"),
        reasoning=ReasoningPolicy.on(),
        backups=tuple(_resolved(provider, model) for provider, model in backups),
    )


@pytest.mark.asyncio
async def test_failover_walks_the_whole_backup_chain() -> None:
    primary = ScriptedProvider(chunks=(), failure=_retryable_failure("primary"))
    first = ScriptedProvider(chunks=(), failure=_retryable_failure("first backup"))
    second = ScriptedProvider()
    providers = {
        "provider": primary,
        "first-backup": first,
        "second-backup": second,
    }
    executor = ProviderExecutor(
        providers.__getitem__,
        token_counter=lambda _messages, _system, _tools: 17,
    )
    routed = _cascade_routed_request(
        ("first-backup", "first-model"),
        ("second-backup", "second-model"),
    )

    stream = _stream(executor, routed)

    assert [chunk async for chunk in stream] == ["event: message_stop\ndata: {}\n\n"]
    assert _streamed_request(first).model == "first-model"
    assert _streamed_request(second).model == "second-model"
    assert primary.stream_close_calls == 1
    assert first.stream_close_calls == 1
    assert second.stream_close_calls == 1


@pytest.mark.asyncio
async def test_the_last_link_in_the_chain_surfaces_its_own_error() -> None:
    primary = ScriptedProvider(chunks=(), failure=_retryable_failure("primary"))
    first = ScriptedProvider(chunks=(), failure=_retryable_failure("first backup"))
    second = ScriptedProvider(chunks=(), failure=_retryable_failure("last backup"))
    providers = {
        "provider": primary,
        "first-backup": first,
        "second-backup": second,
    }
    executor = ProviderExecutor(
        providers.__getitem__,
        token_counter=lambda _messages, _system, _tools: 17,
    )
    routed = _cascade_routed_request(
        ("first-backup", "first-model"),
        ("second-backup", "second-model"),
    )

    with pytest.raises(ExecutionFailure) as exc_info:
        [chunk async for chunk in _stream(executor, routed)]

    assert exc_info.value.message == "last backup"


@pytest.mark.asyncio
async def test_preflight_swap_can_skip_several_links() -> None:
    primary = ScriptedProvider(preflight_failure=_retryable_failure("primary"))
    first = ScriptedProvider(preflight_failure=_retryable_failure("first backup"))
    second = ScriptedProvider()
    providers = {
        "provider": primary,
        "first-backup": first,
        "second-backup": second,
    }
    executor = ProviderExecutor(
        providers.__getitem__,
        token_counter=lambda _messages, _system, _tools: 17,
    )
    routed = _cascade_routed_request(
        ("first-backup", "first-model"),
        ("second-backup", "second-model"),
    )

    stream = _stream(executor, routed)

    assert [chunk async for chunk in stream] == ["event: message_stop\ndata: {}\n\n"]
    assert primary.stream_calls == []
    assert first.stream_calls == []
    assert len(second.preflight_calls) == 1
