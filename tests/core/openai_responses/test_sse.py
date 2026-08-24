"""OpenAI Responses presentation contracts for canonical inference events."""

from collections.abc import AsyncIterator

import pytest

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.async_iterators import AsyncCloseable
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.inference import (
    FinishReason,
    InferenceEvent,
    InferenceStreamLedger,
    ReplayArtifact,
    ReplayArtifactKind,
    ReplayArtifactOrigin,
    ReplayAttachment,
    ResponseStarted,
)
from free_claude_code.core.openai_responses import ResponsesPresentationSnapshot
from free_claude_code.core.openai_responses.presenter import (
    ResponsesEventPresenter,
    iter_responses_sse_from_events,
)
from free_claude_code.core.replay_envelope import decode_replay_envelope
from tests.inference_support import reported_usage, text_event_stream


class _CloseTrackingEvents:
    def __init__(
        self,
        values: list[InferenceEvent],
        *,
        iteration_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self._values = iter(values)
        self._iteration_error = iteration_error
        self._close_error = close_error
        self.close_calls = 0

    def __aiter__(self) -> _CloseTrackingEvents:
        return self

    async def __anext__(self) -> InferenceEvent:
        try:
            return next(self._values)
        except StopIteration:
            if self._iteration_error is not None:
                error = self._iteration_error
                self._iteration_error = None
                raise error from None
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error


def _snapshot() -> ResponsesPresentationSnapshot:
    return ResponsesPresentationSnapshot(
        model="nvidia_nim/test-model",
        parallel_tool_calls=True,
        tool_choice="auto",
        temperature=None,
        top_p=None,
        max_output_tokens=None,
    )


async def _aiter(events: list[InferenceEvent]) -> AsyncIterator[InferenceEvent]:
    for event in events:
        yield event


async def _collect(events: list[InferenceEvent]) -> list[str]:
    return [
        chunk
        async for chunk in iter_responses_sse_from_events(
            _aiter(events),
            _snapshot(),
        )
    ]


@pytest.mark.asyncio
async def test_text_stream_has_one_stable_lifecycle_and_monotonic_sequences() -> None:
    chunks = await _collect(
        text_event_stream(
            "hello",
            model="public-model",
            input_tokens=5,
            output_tokens=2,
        )
    )
    events = parse_sse_text("".join(chunks))

    assert events[0].event == "response.created"
    assert events[-1].event == "response.completed"
    response_ids = {
        event.data["response"]["id"]
        for event in events
        if isinstance(event.data.get("response"), dict)
    }
    assert len(response_ids) == 1
    assert [event.data["sequence_number"] for event in events] == list(
        range(len(events))
    )
    final = events[-1].data["response"]
    assert final["status"] == "completed"
    assert final["model"] == "nvidia_nim/test-model"
    assert final["output"][0]["content"][0]["text"] == "hello"
    assert final["usage"] == {
        "input_tokens": 5,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 7,
    }


@pytest.mark.asyncio
async def test_reasoning_replay_and_tool_call_remain_semantic_until_presented() -> None:
    ledger = InferenceStreamLedger("response_internal", "provider-model", 3)
    replay = ReplayArtifact(
        origin=ReplayArtifactOrigin.OPENAI,
        kind=ReplayArtifactKind.ENCRYPTED_REASONING,
        attachment=ReplayAttachment.REASONING,
        payload="opaque",
    )
    events: list[InferenceEvent] = [ledger.start_response()]
    events.append(ledger.start_reasoning_block(artifacts=(replay,)))
    events.append(ledger.emit_reasoning_delta("inspect"))
    events.append(ledger.stop_reasoning_block())
    events.append(ledger.start_tool_block(0, "call_1", "Read"))
    events.append(ledger.emit_tool_delta(0, '{"path":"README.md"}'))
    events.append(ledger.stop_tool_block(0))
    events.extend(
        ledger.finish_events(
            FinishReason.TOOL_CALLS,
            reported_usage(
                input_tokens=3,
                output_tokens=8,
                reasoning_output_tokens=2,
            ),
        )
    )

    parsed = parse_sse_text("".join(await _collect(events)))
    final = parsed[-1].data["response"]
    reasoning, tool = final["output"]
    assert reasoning["type"] == "reasoning"
    encrypted = reasoning["encrypted_content"]
    assert isinstance(encrypted, str)
    artifacts = decode_replay_envelope(
        encrypted,
        attachment=ReplayAttachment.REASONING,
    )
    assert artifacts is not None
    assert [artifact.payload for artifact in artifacts] == ["opaque"]
    assert tool == {
        "id": tool["id"],
        "type": "function_call",
        "status": "completed",
        "call_id": "call_1",
        "name": "Read",
        "arguments": '{"path":"README.md"}',
    }
    assert final["usage"]["output_tokens_details"] == {"reasoning_tokens": 2}


@pytest.mark.asyncio
async def test_output_limit_maps_to_incomplete_response() -> None:
    ledger = InferenceStreamLedger("response_internal", "provider-model")
    events: list[InferenceEvent] = [ledger.start_response()]
    events.extend(ledger.ensure_text_block())
    events.append(ledger.emit_text_delta("partial"))
    events.extend(ledger.close_all_blocks())
    events.extend(
        ledger.finish_events(
            FinishReason.OUTPUT_LIMIT,
            reported_usage(input_tokens=1, output_tokens=2),
        )
    )

    parsed = parse_sse_text("".join(await _collect(events)))

    assert parsed[-1].event == "response.incomplete"
    assert parsed[-1].data["response"]["status"] == "incomplete"
    assert parsed[-1].data["response"]["incomplete_details"] == {
        "reason": "max_output_tokens"
    }


@pytest.mark.asyncio
async def test_pre_start_failure_is_not_hidden_in_a_responses_envelope() -> None:
    failure = ExecutionFailure(
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="busy",
        retryable=True,
    )
    source = _CloseTrackingEvents([], iteration_error=failure)

    with pytest.raises(ExecutionFailure) as exc_info:
        [chunk async for chunk in iter_responses_sse_from_events(source, _snapshot())]

    assert exc_info.value is failure
    assert source.close_calls == 1


@pytest.mark.asyncio
async def test_post_start_failure_becomes_one_terminal_response_event() -> None:
    failure = ExecutionFailure(
        kind=FailureKind.OVERLOADED,
        status_code=529,
        message="busy",
        retryable=True,
    )
    source = _CloseTrackingEvents(
        [ResponseStarted("response_internal", "provider-model")],
        iteration_error=failure,
    )
    observed: list[BaseException] = []

    chunks = [
        chunk
        async for chunk in iter_responses_sse_from_events(
            source,
            _snapshot(),
            on_post_start_terminal_failure=observed.append,
        )
    ]
    parsed = parse_sse_text("".join(chunks))

    assert [event.event for event in parsed] == [
        "response.created",
        "response.failed",
    ]
    assert parsed[-1].data["response"]["error"]["type"] == "overloaded_error"
    assert observed == [failure]
    assert source.close_calls == 1


@pytest.mark.asyncio
async def test_early_consumer_close_closes_canonical_source_once() -> None:
    source = _CloseTrackingEvents(text_event_stream("hello"))
    stream = iter_responses_sse_from_events(source, _snapshot())

    assert parse_sse_text(await anext(stream))[0].event == "response.created"
    assert isinstance(stream, AsyncCloseable)
    await stream.aclose()

    assert source.close_calls == 1


def test_presenter_rejects_events_after_terminal_completion() -> None:
    presenter = ResponsesEventPresenter(_snapshot())
    for event in text_event_stream("done"):
        presenter.present(event)

    with pytest.raises(RuntimeError, match="after response completion"):
        presenter.present(ResponseStarted("response_again", "provider-model"))
