import asyncio
import sqlite3
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from free_claude_code.application.chat import (
    ChatConflictError,
    ChatEventSubscriptionPort,
    ChatNotFoundError,
    ChatOperationAcknowledgement,
    ChatOperationKind,
    ChatReasoning,
    ChatService,
    ChatSession,
    ChatUnavailableError,
    GenerationStatus,
)
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.ports import ProviderPort, RequestRuntimeLease
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.interprocess_lock import InterprocessFileLock
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.runtime.chat_sqlite import SQLiteChatStore


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_fails", [False, True])
async def test_cancelled_startup_drains_retirement_worker_before_releasing_lock(
    tmp_path, monkeypatch, repair_fails
):
    store = SQLiteChatStore(tmp_path / "chat.db", tmp_path / "chat.lock")
    await store.start()
    original_run = store._run_sync
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocked_run(operation):
        entered.set()
        assert release.wait(5)
        try:
            if repair_fails:
                raise OSError("injected repair failure after cancellation")
            return original_run(operation)
        finally:
            finished.set()

    monkeypatch.setattr(store, "_run_sync", blocked_run)
    service = ChatService(FakeRuntime(FakeChatProvider()), store)
    task = asyncio.create_task(service.start())
    replacement_lock = InterprocessFileLock(tmp_path / "chat.lock")
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()
        assert not finished.is_set()
        assert not replacement_lock.acquire()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert replacement_lock.acquire()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        replacement_lock.release()
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("supports_thinking", [True, False, None])
async def test_retired_chat_selection_startup_and_stale_patch(
    tmp_path, monkeypatch, supports_thinking
):
    runtime = FakeRuntime(FakeChatProvider())
    monkeypatch.setattr(
        runtime,
        "cached_model_info",
        lambda provider_id, model_id: ProviderModelInfo(
            model_id, supports_thinking=supports_thinking
        ),
    )
    store = SQLiteChatStore(tmp_path / "chat.db", tmp_path / "chat.lock")
    await store.start()
    old = await store.create_session(
        session_id="aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa",
        model="github_models/old",
        reasoning=ChatReasoning.HIGH,
    )
    await store.close()
    service = ChatService(runtime, store)
    await service.start()
    try:
        assert service.availability() == (True, None)
        repaired = await service.get_session(old.id)
        assert repaired.model == runtime.settings.model
        assert repaired.revision == old.revision + 1
        expected_reasoning = (
            ChatReasoning.OFF if supports_thinking is False else ChatReasoning.HIGH
        )
        assert repaired.reasoning is expected_reasoning
        with pytest.raises(ChatConflictError):
            await service.update_session(
                old.id,
                expected_revision=old.revision,
                title=None,
                model="github_models/old",
                reasoning=None,
            )
        assert await service.get_session(old.id) == repaired
        # Simulate a stale tab submitting an old selection while thinking is enabled.
        enabled = await store.update_session(
            old.id,
            expected_revision=repaired.revision,
            title=None,
            model=None,
            reasoning=ChatReasoning.HIGH,
        )
        patched = await service.update_session(
            old.id,
            expected_revision=enabled.revision,
            title=None,
            model="github_models/old",
            reasoning=None,
        )
        assert patched.model == runtime.settings.model
        assert patched.reasoning is expected_reasoning
        assert runtime.leases == []
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [True, False])
async def test_chat_retirement_failure_releases_store_lock(
    tmp_path, monkeypatch, cancel
):
    store = SQLiteChatStore(tmp_path / "chat.db", tmp_path / "chat.lock")
    service = ChatService(FakeRuntime(FakeChatProvider()), store)

    async def fail_repair(**kwargs):
        if cancel:
            raise asyncio.CancelledError()
        raise OSError("injected repair failure")

    monkeypatch.setattr(store, "repair_retired_model_selections", fail_repair)
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await service.start()
    else:
        await service.start()
    assert not service.availability()[0]
    replacement = SQLiteChatStore(tmp_path / "chat.db", tmp_path / "chat.lock")
    await replacement.start()
    await replacement.close()
    await service.close()


@pytest.mark.asyncio
async def test_chat_retirement_failed_cleanup_is_retried_before_start(
    tmp_path, monkeypatch
):
    store = SQLiteChatStore(tmp_path / "chat.db", tmp_path / "chat.lock")
    service = ChatService(FakeRuntime(FakeChatProvider()), store)
    original_repair = store.repair_retired_model_selections
    original_close = store.close

    async def fail_repair(**kwargs):
        raise OSError("injected repair failure")

    async def fail_close():
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(store, "repair_retired_model_selections", fail_repair)
    monkeypatch.setattr(store, "close", fail_close)
    await service.start()
    assert not service.availability()[0]
    monkeypatch.setattr(store, "repair_retired_model_selections", original_repair)
    monkeypatch.setattr(store, "close", original_close)
    await service.start()
    try:
        assert service.availability() == (True, None)
        await service.create_session()
    finally:
        await service.close()


class FakeChatProvider:
    def __init__(
        self,
        *,
        block_after_delta: bool = False,
        truncate_summary: bool = False,
    ) -> None:
        self.block_after_delta = block_after_delta
        self.truncate_summary = truncate_summary
        self.started = asyncio.Event()
        self.closed = 0
        self.requests: list[MessagesRequest] = []

    def preflight_messages(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> None:
        del reasoning
        self.requests.append(request)

    def preflight_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        reasoning: ReasoningPolicy,
    ) -> None:
        raise AssertionError((request, reasoning))

    async def stream_messages(
        self,
        request: MessagesRequest,
        *,
        input_tokens: int,
        request_id: str,
        response_model: str,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        del input_tokens, request_id, response_model, reasoning
        text = "summary" if str(request.system).startswith("Summarize") else "answer"
        frames = [
            format_sse_event(
                "message_start",
                {"type": "message_start", "message": {"content": []}},
            ),
            format_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            ),
            format_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "thought"},
                },
            ),
            format_sse_event(
                "content_block_stop", {"type": "content_block_stop", "index": 0}
            ),
            format_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            format_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": text},
                },
            ),
        ]
        try:
            wire = "".join(frames)
            midpoint = len(wire) // 2
            yield wire[:midpoint]
            yield wire[midpoint:]
            self.started.set()
            if self.truncate_summary and text == "summary":
                return
            if self.block_after_delta:
                await asyncio.Event().wait()
            yield format_sse_event(
                "content_block_stop", {"type": "content_block_stop", "index": 1}
            )
            yield format_sse_event(
                "message_delta",
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            )
            yield format_sse_event("message_stop", {"type": "message_stop"})
        finally:
            self.closed += 1

    async def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        input_tokens: int,
        request_id: str,
        response_model: str,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        raise AssertionError(
            (request, input_tokens, request_id, response_model, reasoning)
        )
        yield ""


def _context_failure() -> ExecutionFailure:
    return ExecutionFailure(
        kind=FailureKind.CONTEXT_WINDOW_EXCEEDED,
        status_code=400,
        message="Provider input exceeds the model context window.",
        retryable=False,
    )


class ContextRecoveryProvider(FakeChatProvider):
    def __init__(self) -> None:
        super().__init__()
        self.generation_failures = 0
        self.generation_failure_after_segment = False
        self.generation_attempts = 0
        self.summary_batch_limit: int | None = None
        self.summary_failure: ExecutionFailure | None = None
        self.summary_batch_sizes: list[int] = []
        self.block_summary = False
        self.summary_started = asyncio.Event()

    async def stream_messages(
        self,
        request: MessagesRequest,
        *,
        input_tokens: int,
        request_id: str,
        response_model: str,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        is_summary = str(request.system).startswith("Summarize")
        if is_summary:
            source = request.messages[0].content
            if not isinstance(source, str):
                raise AssertionError("Compaction source must be text.")
            batch_size = source.count("USER\n")
            self.summary_batch_sizes.append(batch_size)
            self.summary_started.set()
            if self.block_summary:
                await asyncio.Event().wait()
            if self.summary_failure is not None:
                raise self.summary_failure
            if (
                self.summary_batch_limit is not None
                and batch_size > self.summary_batch_limit
            ):
                raise _context_failure()
        else:
            self.generation_attempts += 1
            if self.generation_failures > 0:
                self.generation_failures -= 1
                if self.generation_failure_after_segment:
                    yield format_sse_event(
                        "message_start",
                        {"type": "message_start", "message": {"content": []}},
                    )
                    yield format_sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                raise _context_failure()

        async for frame in super().stream_messages(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
        ):
            yield frame


class MixedFallbackFailureProvider(ContextRecoveryProvider):
    def __init__(self) -> None:
        super().__init__()
        self.fail_candidates = False

    async def stream_messages(
        self,
        request: MessagesRequest,
        *,
        input_tokens: int,
        request_id: str,
        response_model: str,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        if self.fail_candidates and not str(request.system).startswith("Summarize"):
            if request.model == "model":
                raise ExecutionFailure(
                    kind=FailureKind.PERMISSION,
                    status_code=403,
                    message="primary denied access",
                    retryable=False,
                )
            if request.model == "fallback":
                raise _context_failure()
            raise AssertionError(f"Unexpected fallback model: {request.model}")
        async for frame in super().stream_messages(
            request,
            input_tokens=input_tokens,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
        ):
            yield frame


class BackpressuredCompletionProvider(FakeChatProvider):
    async def stream_messages(
        self,
        request: MessagesRequest,
        *,
        input_tokens: int,
        request_id: str,
        response_model: str,
        reasoning: ReasoningPolicy,
    ) -> AsyncIterator[str]:
        del request, input_tokens, request_id, response_model, reasoning
        yield format_sse_event(
            "message_start",
            {"type": "message_start", "message": {"content": []}},
        )
        yield format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        for _ in range(125):
            yield format_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "x"},
                },
            )
        yield format_sse_event(
            "content_block_stop", {"type": "content_block_stop", "index": 0}
        )
        yield format_sse_event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        )
        yield format_sse_event("message_stop", {"type": "message_stop"})


class FakeLease:
    def __init__(
        self,
        settings: Settings,
        provider: FakeChatProvider,
        model_infos: tuple[ProviderModelInfo, ...],
    ) -> None:
        self._settings = settings
        self._provider = provider
        self.model_infos = model_infos
        self.released = 0

    @property
    def generation_id(self) -> int:
        return 1

    @property
    def settings(self) -> Settings:
        return self._settings

    def is_provider_cached(self, provider_id: str) -> bool:
        return provider_id == "groq"

    def resolve_provider(self, provider_id: str) -> ProviderPort:
        assert provider_id == "groq"
        return self._provider

    async def release(self) -> None:
        self.released += 1


class FakeRuntime:
    def __init__(
        self,
        provider: FakeChatProvider,
        *,
        context_window_tokens: int | None = 100_000,
    ) -> None:
        self.settings = Settings().model_copy(
            update={
                "model": "groq/model",
                "model_fallbacks": None,
                "provider_progress_timeout": 5.0,
            }
        )
        self.provider = provider
        self.context_window_tokens = context_window_tokens
        self.leases: list[FakeLease] = []

    async def acquire(
        self, *, include_model_infos: bool = False
    ) -> RequestRuntimeLease:
        lease = FakeLease(
            self.settings,
            self.provider,
            (
                ProviderModelInfo(
                    "groq/model",
                    supports_thinking=True,
                    context_window_tokens=self.context_window_tokens,
                    max_output_tokens=20_000,
                ),
            )
            if include_model_infos
            else (),
        )
        self.leases.append(lease)
        return lease

    def current_settings(self) -> Settings:
        return self.settings

    def cached_model_info(
        self, provider_id: str, model_id: str
    ) -> ProviderModelInfo | None:
        if (provider_id, model_id) == ("groq", "model"):
            return ProviderModelInfo(
                "model",
                supports_thinking=True,
                context_window_tokens=self.context_window_tokens,
                max_output_tokens=20_000,
            )
        return None

    def cached_prefixed_model_infos(self) -> tuple[ProviderModelInfo, ...]:
        return ()


async def _service(
    tmp_path: Path, provider: FakeChatProvider
) -> tuple[ChatService, FakeRuntime, SQLiteChatStore]:
    runtime = FakeRuntime(provider)
    store = SQLiteChatStore(tmp_path / "chat.db", tmp_path / "chat.lock")
    service = ChatService(runtime, store)
    await service.start()
    return service, runtime, store


_TURN_TERMINAL_EVENTS = frozenset(
    {"turn.completed", "turn.failed", "turn.stopped", "operation.failed"}
)
_COMPACTION_TERMINAL_EVENTS = frozenset(
    {
        "compaction.completed",
        "compaction.failed",
        "compaction.stopped",
        "operation.failed",
    }
)


@dataclass(slots=True)
class _ObservedOperation:
    acknowledgement: ChatOperationAcknowledgement
    subscription: ChatEventSubscriptionPort

    async def aclose(self) -> None:
        await self.subscription.aclose()


async def _observe_call[**P](
    service: ChatService,
    command: Callable[P, Awaitable[ChatOperationAcknowledgement]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> _ObservedOperation:
    subscription, _active = await service.subscribe()
    try:
        acknowledgement = await command(*args, **kwargs)
    except BaseException:
        await subscription.aclose()
        raise
    return _ObservedOperation(acknowledgement, subscription)


async def _drain(operation: _ObservedOperation) -> list[str]:
    events: list[str] = []
    terminal_events = (
        _COMPACTION_TERMINAL_EVENTS
        if operation.acknowledgement.kind is ChatOperationKind.COMPACT
        else _TURN_TERMINAL_EVENTS
    )
    try:
        async for event in operation.subscription:
            if event.data.get("operation_id") != operation.acknowledgement.operation_id:
                continue
            assert event.data["kind"] == operation.acknowledgement.kind.value
            events.append(event.event)
            if event.event in terminal_events:
                break
    finally:
        await operation.aclose()
    return events


async def _seed_completed_turns(
    service: ChatService,
    session: ChatSession,
    *,
    count: int,
    operation_offset: int = 1,
) -> ChatSession:
    current = session
    for index in range(operation_offset, operation_offset + count):
        operation = await _observe_call(
            service,
            service.send,
            current.id,
            expected_revision=current.revision,
            operation_id=f"00000000-0000-4000-8000-{index:012d}",
            text=f"turn {index}",
        )
        assert (await _drain(operation))[-1] == "turn.completed"
        current = await service.get_session(current.id)
    return current


@pytest.mark.asyncio
async def test_send_streams_and_persists_interleaved_segments(tmp_path: Path):
    provider = FakeChatProvider()
    service, runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="b21677f0-aa9a-4acb-b197-64d3dbd56536",
            text="hello",
        )
        events = await _drain(stream)

        transcript = await store.get_transcript(session.id)
        generation = transcript.turns[0].generation
        assert events[-1] == "turn.completed"
        assert [segment.text for segment in generation.segments] == [
            "thought",
            "answer",
        ]
        assert generation.actual_model == "groq/model"
        assert runtime.leases[0].released == 1
        assert provider.closed == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_completion_persistence_exhaustion_never_downgrades_to_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        statuses: list[GenerationStatus] = []
        original_finish_generation = store.finish_generation

        async def unavailable_completion(*args, **kwargs):
            status = kwargs["status"]
            statuses.append(status)
            if len(statuses) <= 2:
                raise ChatUnavailableError("Chat storage is temporarily unavailable.")
            return await original_finish_generation(*args, **kwargs)

        monkeypatch.setattr(store, "finish_generation", unavailable_completion)
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="f69c7c4d-9a46-40f0-b6ad-7f288a2c2dd7",
            text="finish successfully",
        )

        assert (await _drain(stream))[-1] == "turn.failed"
        assert statuses == [GenerationStatus.COMPLETED, GenerationStatus.COMPLETED]
        generation = (await store.get_transcript(session.id)).turns[0].generation
        assert generation.status is GenerationStatus.RUNNING
        assert service.availability() == (
            False,
            "Chat storage became unavailable. Restart FCC to repair Chat Sessions.",
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_regeneration_retries_an_ambiguous_terminal_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        initial = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="e8476211-c6a1-45c4-a463-b7d92bfcc7d8",
            text="answer once",
        )
        assert (await _drain(initial))[-1] == "turn.completed"
        before = await store.get_transcript(session.id)
        original_id = before.turns[0].generation.id

        attempts = 0
        original_finish_regeneration = store.finish_regeneration

        async def ambiguous_finish_regeneration(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            result = await original_finish_regeneration(*args, **kwargs)
            if attempts == 1:
                raise ChatUnavailableError("Chat commit result was unavailable.")
            return result

        monkeypatch.setattr(
            store,
            "finish_regeneration",
            ambiguous_finish_regeneration,
        )
        replacement = await _observe_call(
            service,
            service.regenerate,
            session.id,
            expected_revision=before.session.revision,
            operation_id="8a99c4a9-e4f5-4ebd-a8c9-982a9227a557",
        )

        assert (await _drain(replacement))[-1] == "turn.completed"
        after = await store.get_transcript(session.id)
        assert attempts == 2
        assert after.turns[0].generation.id != original_id
        assert after.turns[0].generation.status is GenerationStatus.COMPLETED
        assert service.availability()[0] is True
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_failed_regeneration_replaces_original_with_failed_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        initial = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="01355e92-06c4-4820-b80a-fc8e0dff2553",
            text="replace this answer",
        )
        assert (await _drain(initial))[-1] == "turn.completed"
        before = await store.get_transcript(session.id)
        original_id = before.turns[0].generation.id

        async def failing_stream(*args, **kwargs):
            del args, kwargs
            yield "".join(
                (
                    format_sse_event(
                        "message_start",
                        {"type": "message_start", "message": {"content": []}},
                    ),
                    format_sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        },
                    ),
                    format_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": "partial"},
                        },
                    ),
                    format_sse_event(
                        "error",
                        {
                            "type": "error",
                            "error": {"message": "provider failed"},
                        },
                    ),
                )
            )

        monkeypatch.setattr(provider, "stream_messages", failing_stream)
        replacement = await _observe_call(
            service,
            service.regenerate,
            session.id,
            expected_revision=before.session.revision,
            operation_id="810b8d15-adab-4df6-9084-3189ca534d09",
        )

        assert (await _drain(replacement))[-1] == "turn.failed"
        generation = (await store.get_transcript(session.id)).turns[0].generation
        assert generation.id != original_id
        assert generation.status is GenerationStatus.FAILED
        assert generation.error_message == "provider failed"
        assert generation.segments[-1].text == "partial"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stopped_regeneration_retries_an_ambiguous_discard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        initial = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="14e14878-842d-425a-952c-4e9f105760c0",
            text="keep the original answer",
        )
        assert (await _drain(initial))[-1] == "turn.completed"
        before = await store.get_transcript(session.id)
        original_generation = before.turns[0].generation

        discard_attempts = 0
        segment_writes = 0
        original_discard_generation = store.discard_generation
        original_replace_segments = store.replace_generation_segments

        async def ambiguous_discard(*args, **kwargs):
            nonlocal discard_attempts
            discard_attempts += 1
            await original_discard_generation(*args, **kwargs)
            if discard_attempts == 1:
                raise ChatUnavailableError("Chat commit result was unavailable.")

        async def observed_replace_segments(*args, **kwargs):
            nonlocal segment_writes
            segment_writes += 1
            return await original_replace_segments(*args, **kwargs)

        monkeypatch.setattr(store, "discard_generation", ambiguous_discard)
        monkeypatch.setattr(
            store, "replace_generation_segments", observed_replace_segments
        )
        provider.started.clear()
        provider.block_after_delta = True
        replacement = await _observe_call(
            service,
            service.regenerate,
            session.id,
            expected_revision=before.session.revision,
            operation_id="19cbb991-38ff-4cd2-a20e-f94783f78003",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        writes_before_stop = segment_writes

        assert await service.stop(
            session.id,
            operation_id="19cbb991-38ff-4cd2-a20e-f94783f78003",
        )
        assert (await _drain(replacement))[-1] == "turn.stopped"

        after = await store.get_transcript(session.id)
        assert discard_attempts == 2
        assert segment_writes == writes_before_stop
        assert after.turns[0].generation == original_generation
        assert service.availability()[0] is True
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_closing_subscription_does_not_cancel_active_operation(
    tmp_path: Path,
):
    provider = FakeChatProvider(block_after_delta=True)
    service, runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="9781df8c-aa92-422e-97d9-e9ea7f542b89",
            text="hello",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        await stream.aclose()

        assert runtime.leases[0].released == 0
        assert provider.closed == 0

        replacement_subscription, active_operations = await service.subscribe()
        assert [active.operation_id for active in active_operations] == [
            stream.acknowledgement.operation_id
        ]
        replacement_observer = _ObservedOperation(
            stream.acknowledgement, replacement_subscription
        )
        assert await service.stop(
            session.id,
            operation_id=stream.acknowledgement.operation_id,
        )
        events = await asyncio.wait_for(_drain(replacement_observer), timeout=1)

        generation = (await store.get_transcript(session.id)).turns[0].generation
        assert events[-1] == "turn.stopped"
        assert generation.status is GenerationStatus.STOPPED
        assert generation.segments[-1].text == "answer"
        assert runtime.leases[0].released == 1
        assert provider.closed == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stop_retries_transient_terminal_persistence_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider(block_after_delta=True)
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        operation_id = "3a1c071f-f558-4be3-a072-f9af230352ec"
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id=operation_id,
            text="survive one storage failure",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)

        attempts = 0
        original_finish_generation = store.finish_generation

        async def flaky_finish_generation(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ChatUnavailableError("Chat storage is temporarily unavailable.")
            return await original_finish_generation(*args, **kwargs)

        monkeypatch.setattr(store, "finish_generation", flaky_finish_generation)

        assert await service.stop(session.id, operation_id=operation_id) is True
        assert (await _drain(stream))[-1] == "turn.stopped"

        transcript = await store.get_transcript(session.id)
        assert attempts == 2
        assert transcript.turns[-1].generation.status is GenerationStatus.STOPPED
        snapshot_subscription, active_operations = await service.subscribe()
        await snapshot_subscription.aclose()
        assert active_operations == ()

        provider.block_after_delta = False
        replacement = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=transcript.session.revision,
            operation_id="f5190fdb-ce16-402d-a5a2-1633ba0cae55",
            text="keep going",
        )
        assert (await _drain(replacement))[-1] == "turn.completed"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stop_retries_transient_terminal_segment_flush(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider(block_after_delta=True)
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        operation_id = "242841f6-d826-48b5-9ed7-bf073b17840e"
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id=operation_id,
            text="preserve the final partial segment",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)

        attempts = 0
        original_replace_segments = store.replace_generation_segments

        async def flaky_replace_segments(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ChatUnavailableError("Chat storage is temporarily unavailable.")
            return await original_replace_segments(*args, **kwargs)

        monkeypatch.setattr(
            store, "replace_generation_segments", flaky_replace_segments
        )

        assert await service.stop(session.id, operation_id=operation_id) is True
        assert (await _drain(stream))[-1] == "turn.stopped"

        generation = (await store.get_transcript(session.id)).turns[-1].generation
        assert attempts == 2
        assert generation.status is GenerationStatus.STOPPED
        assert generation.segments[-1].text == "answer"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_terminal_persistence_failure_disables_chat_instead_of_lying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider(block_after_delta=True)
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        operation_id = "f744570e-0ec6-4f70-b74d-19de6bc60547"
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id=operation_id,
            text="surface a persistent storage failure",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)

        attempts = 0

        async def unavailable_finish_generation(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            raise ChatUnavailableError("Chat storage is unavailable.")

        monkeypatch.setattr(
            store,
            "finish_generation",
            unavailable_finish_generation,
        )

        assert await service.stop(session.id, operation_id=operation_id) is True
        assert (await _drain(stream))[-1] == "turn.failed"
        assert attempts == 2

        available, message = service.availability()
        assert available is False
        assert message == (
            "Chat storage became unavailable. Restart FCC to repair Chat Sessions."
        )
        with pytest.raises(ChatUnavailableError, match="Restart FCC"):
            await service.get_detail(session.id)

        generation = (await store.get_transcript(session.id)).turns[-1].generation
        assert generation.status is GenerationStatus.RUNNING
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_immediate_stop_before_operation_task_starts_releases_owner(
    tmp_path: Path,
):
    provider = FakeChatProvider()
    service, _runtime, _store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        operation_id = "2029b65e-f4d9-4b80-8c6a-563a32fb8b8f"
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id=operation_id,
            text="stop before start",
        )

        assert await service.stop(session.id, operation_id=operation_id) is True
        assert (await _drain(stream))[-1] == "turn.stopped"

        replacement = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="4b50506e-92bf-46ba-aa35-333dc0946e14",
            text="still usable",
        )
        assert (await _drain(replacement))[-1] == "turn.completed"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_ambiguous_generation_start_adopts_the_committed_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        original_begin_send = store.begin_send

        async def ambiguous_begin_send(*args, **kwargs):
            await original_begin_send(*args, **kwargs)
            raise ChatUnavailableError("Chat commit result was unavailable.")

        monkeypatch.setattr(store, "begin_send", ambiguous_begin_send)
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="cc47acc3-d7c2-46d0-a1b6-fe2acda191e4",
            text="recover the committed start",
        )

        assert (await _drain(stream))[-1] == "turn.completed"
        generation = (await store.get_transcript(session.id)).turns[0].generation
        assert generation.status is GenerationStatus.COMPLETED
        assert (
            await service.stop(
                session.id,
                operation_id="cc47acc3-d7c2-46d0-a1b6-fe2acda191e4",
            )
            is False
        )
        assert service.availability()[0] is True
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_disconnect_does_not_interrupt_failure_terminalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    release_flush = asyncio.Event()
    try:

        async def failing_stream(*args, **kwargs):
            del args, kwargs
            yield "".join(
                (
                    format_sse_event(
                        "message_start",
                        {"type": "message_start", "message": {"content": []}},
                    ),
                    format_sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        },
                    ),
                    format_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": "partial"},
                        },
                    ),
                    format_sse_event(
                        "error",
                        {
                            "type": "error",
                            "error": {"message": "provider failed"},
                        },
                    ),
                )
            )

        monkeypatch.setattr(provider, "stream_messages", failing_stream)
        entered_flush = asyncio.Event()
        original_replace_segments = store.replace_generation_segments

        async def blocked_replace_segments(generation_id, segments):
            entered_flush.set()
            await release_flush.wait()
            await original_replace_segments(generation_id, segments)

        monkeypatch.setattr(
            store,
            "replace_generation_segments",
            blocked_replace_segments,
        )
        session = await service.create_session()
        operation_id = "63523531-024d-47f2-a0ce-ff7104d4e3af"
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id=operation_id,
            text="fail before disconnect",
        )
        await asyncio.wait_for(entered_flush.wait(), timeout=1)

        await stream.aclose()
        replacement_subscription, _active = await service.subscribe()
        replacement_observer = _ObservedOperation(
            stream.acknowledgement, replacement_subscription
        )
        release_flush.set()

        assert (await _drain(replacement_observer))[-1] == "turn.failed"
        generation = (await store.get_transcript(session.id)).turns[0].generation
        assert generation.status is GenerationStatus.FAILED
        assert await service.stop(session.id, operation_id=operation_id) is False
        assert service.availability()[0] is True
    finally:
        release_flush.set()
        await service.close()


@pytest.mark.asyncio
async def test_second_operation_on_same_session_is_rejected(tmp_path: Path):
    provider = FakeChatProvider(block_after_delta=True)
    service, _runtime, _store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        first = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="3e363fbc-25ee-414e-a954-02d11990497f",
            text="first",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        running = await service.get_session(session.id)
        with pytest.raises(ChatConflictError, match="active operation"):
            await _observe_call(
                service,
                service.send,
                session.id,
                expected_revision=running.revision,
                operation_id="223ad9ac-e830-42ec-b73d-b1cbd81511c4",
                text="second",
            )
        await first.aclose()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_durable_mutations_publish_once_and_stale_mutation_publishes_nothing(
    tmp_path: Path,
):
    service, _runtime, _store = await _service(tmp_path, FakeChatProvider())
    subscription, active_operations = await service.subscribe()
    assert active_operations == ()
    events = subscription.__aiter__()
    try:
        session = await service.create_session()
        created = await anext(events)

        with pytest.raises(ChatConflictError, match="another tab"):
            await service.update_session(
                session.id,
                expected_revision=session.revision + 10,
                title="Stale title",
                model=None,
                reasoning=None,
            )

        updated = await service.update_session(
            session.id,
            expected_revision=session.revision,
            title="Durable title",
            model=None,
            reasoning=None,
        )
        update_event = await anext(events)
        preferences = await service.save_system_prompt("Shared prompt")
        preferences_event = await anext(events)
        await service.delete_session(
            session.id,
            expected_revision=updated.revision,
        )
        deleted = await anext(events)

        assert [
            (created.id, created.event),
            (update_event.id, update_event.event),
            (preferences_event.id, preferences_event.event),
            (deleted.id, deleted.event),
        ] == [
            (1, "session.created"),
            (2, "session.updated"),
            (3, "preferences.updated"),
            (4, "session.deleted"),
        ]
        assert update_event.data["revision"] == updated.revision
        assert preferences_event.data["updated_at"] == preferences.updated_at
    finally:
        await subscription.aclose()
        await service.close()


@pytest.mark.asyncio
async def test_delete_validates_stale_revision_before_cancelling_active_send(
    tmp_path: Path,
):
    provider = FakeChatProvider(block_after_delta=True)
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        operation_id = "3b4390e2-bbd0-499a-94c5-d7813b1d5f75"
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id=operation_id,
            text="keep running",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        assert (await service.get_session(session.id)).revision > session.revision

        with pytest.raises(ChatConflictError, match="another tab"):
            await service.delete_session(
                session.id,
                expected_revision=session.revision,
            )

        assert await service.stop(session.id, operation_id=operation_id) is True
        generation = (await store.get_transcript(session.id)).turns[0].generation
        assert generation.status is GenerationStatus.STOPPED
        await stream.aclose()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_delete_active_send_finishes_generation_before_removing_session(
    tmp_path: Path,
):
    provider = FakeChatProvider(block_after_delta=True)
    service, _runtime, _store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="0162e7e5-aabd-4ffb-ae50-ed864825ec71",
            text="delete me",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        running = await service.get_session(session.id)

        await service.delete_session(session.id, expected_revision=running.revision)

        with pytest.raises(ChatNotFoundError):
            await service.get_session(session.id)
        await stream.aclose()
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_cancellation_waits_for_generation_start_commit_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider(block_after_delta=True)
    service, _runtime, store = await _service(tmp_path, provider)
    blocker: sqlite3.Connection | None = None
    try:
        session = await service.create_session()
        initial_stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="7f4c7a3f-c06e-42c3-9887-f748dc5aa518",
            text="partial answer",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=5)
        assert await service.stop(
            session.id,
            operation_id="7f4c7a3f-c06e-42c3-9887-f748dc5aa518",
        )
        await initial_stream.aclose()

        entered = asyncio.Event()
        original_begin_retry = store.begin_retry

        async def observed_begin_retry(*args, **kwargs):
            entered.set()
            return await original_begin_retry(*args, **kwargs)

        monkeypatch.setattr(store, "begin_retry", observed_begin_retry)
        blocker = sqlite3.connect(tmp_path / "chat.db", isolation_level=None)
        blocker.execute("PRAGMA journal_mode = WAL")
        blocker.execute("BEGIN IMMEDIATE")

        current = await service.get_session(session.id)
        retry_stream = await _observe_call(
            service,
            service.retry,
            session.id,
            expected_revision=current.revision,
            operation_id="38dcb3e5-04b7-4ecf-bc8b-7c72550839b0",
        )
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.sleep(0.05)

        stop_task = asyncio.create_task(
            service.stop(
                session.id,
                operation_id=retry_stream.acknowledgement.operation_id,
            )
        )
        await asyncio.sleep(0.05)
        waited_for_commit = not stop_task.done()
        blocker.commit()
        blocker.close()
        blocker = None
        assert await asyncio.wait_for(stop_task, timeout=1) is True
        assert (await _drain(retry_stream))[-1] == "turn.stopped"

        generation = (await store.get_transcript(session.id)).turns[-1].generation
        assert waited_for_commit
        assert generation.status is GenerationStatus.STOPPED
    finally:
        if blocker is not None:
            blocker.rollback()
            blocker.close()
        await service.close()


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_terminal_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider(block_after_delta=True)
    service, _runtime, store = await _service(tmp_path, provider)
    blocker: sqlite3.Connection | None = None
    try:
        session = await service.create_session()
        operation_id = "f02c0da7-1ec9-4433-bef3-dc760753e451"
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id=operation_id,
            text="preserve partial output",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)

        entered_cleanup = asyncio.Event()
        original_replace_segments = store.replace_generation_segments

        async def observed_replace_segments(generation_id, segments):
            entered_cleanup.set()
            await original_replace_segments(generation_id, segments)

        monkeypatch.setattr(
            store,
            "replace_generation_segments",
            observed_replace_segments,
        )
        blocker = sqlite3.connect(tmp_path / "chat.db", isolation_level=None)
        blocker.execute("PRAGMA journal_mode = WAL")
        blocker.execute("BEGIN IMMEDIATE")

        first_stop_task = asyncio.create_task(
            service.stop(session.id, operation_id=operation_id)
        )
        await asyncio.wait_for(entered_cleanup.wait(), timeout=1)
        second_stop_task = asyncio.create_task(
            service.stop(session.id, operation_id=operation_id)
        )
        await asyncio.sleep(0.05)

        assert not first_stop_task.done()
        assert not second_stop_task.done()
        blocker.commit()
        blocker.close()
        blocker = None
        assert await asyncio.wait_for(first_stop_task, timeout=1) is True
        assert await asyncio.wait_for(second_stop_task, timeout=1) is True
        assert (await _drain(stream))[-1] == "turn.stopped"

        generation = (await store.get_transcript(session.id)).turns[0].generation
        assert generation.status is GenerationStatus.STOPPED
        assert await service.stop(session.id, operation_id=operation_id) is False
    finally:
        if blocker is not None:
            blocker.rollback()
            blocker.close()
        await service.close()


@pytest.mark.asyncio
async def test_stop_at_generation_commit_preserves_completed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        entered_commit = asyncio.Event()
        release_commit = asyncio.Event()
        original_finish_generation = store.finish_generation

        async def observed_finish_generation(*args, **kwargs):
            entered_commit.set()
            await release_commit.wait()
            return await original_finish_generation(*args, **kwargs)

        monkeypatch.setattr(store, "finish_generation", observed_finish_generation)
        session = await service.create_session()
        operation_id = "03ba5dce-bd47-452c-a878-229d4df65944"
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id=operation_id,
            text="finish this answer",
        )
        await asyncio.wait_for(entered_commit.wait(), timeout=1)

        stop_task = asyncio.create_task(
            service.stop(session.id, operation_id=operation_id)
        )
        assert await asyncio.wait_for(stop_task, timeout=1) is False
        release_commit.set()

        events = await _drain(stream)
        generation = (await store.get_transcript(session.id)).turns[0].generation
        assert events[-1] == "turn.completed"
        assert generation.status is GenerationStatus.COMPLETED
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_snapshot_never_advertises_unpublished_terminal_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    service, _runtime, _store = await _service(tmp_path, FakeChatProvider())
    release_terminal = asyncio.Event()
    late_subscription: ChatEventSubscriptionPort | None = None
    fresh_subscription: ChatEventSubscriptionPort | None = None
    try:
        entered_terminal = asyncio.Event()
        original_release_active = service._release_active

        async def observed_release_active(active):
            entered_terminal.set()
            await release_terminal.wait()
            return await original_release_active(active)

        monkeypatch.setattr(service, "_release_active", observed_release_active)
        session = await service.create_session()
        operation_id = "025295bc-d2f5-4208-9dab-ed507471fb33"
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id=operation_id,
            text="publish this terminal event",
        )
        await asyncio.wait_for(entered_terminal.wait(), timeout=1)

        late_subscription, active = await service.subscribe()
        assert len(active) == 1
        assert active[0].operation_id == operation_id
        snapshot_sequence = active[0].operation_sequence

        release_terminal.set()
        late_events = late_subscription.__aiter__()
        terminal = await asyncio.wait_for(anext(late_events), timeout=1)
        assert terminal.event == "turn.completed"
        assert terminal.data["operation_sequence"] == snapshot_sequence + 1
        assert (await _drain(stream))[-1] == "turn.completed"

        fresh_subscription, current_active = await service.subscribe()
        assert current_active == ()
    finally:
        release_terminal.set()
        if late_subscription is not None:
            await late_subscription.aclose()
        if fresh_subscription is not None:
            await fresh_subscription.aclose()
        await service.close()


@pytest.mark.asyncio
async def test_stop_after_completed_regeneration_cannot_discard_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    service, _runtime, store = await _service(tmp_path, FakeChatProvider())
    release_terminal = asyncio.Event()
    try:
        session = await service.create_session()
        initial = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="25362cff-95ae-4051-9d15-36317753eb29",
            text="replace this answer",
        )
        await _drain(initial)
        before = await store.get_transcript(session.id)

        entered_terminal = asyncio.Event()
        original_release_active = service._release_active

        async def observed_release_active(active):
            entered_terminal.set()
            await release_terminal.wait()
            return await original_release_active(active)

        discarded = 0
        original_discard_generation = store.discard_generation

        async def observed_discard_generation(*args, **kwargs):
            nonlocal discarded
            discarded += 1
            return await original_discard_generation(*args, **kwargs)

        monkeypatch.setattr(service, "_release_active", observed_release_active)
        monkeypatch.setattr(store, "discard_generation", observed_discard_generation)
        operation_id = "a2f69564-3eb5-4805-8578-2be42d710e5a"
        replacement = await _observe_call(
            service,
            service.regenerate,
            session.id,
            expected_revision=before.session.revision,
            operation_id=operation_id,
        )
        await asyncio.wait_for(entered_terminal.wait(), timeout=1)

        assert await service.stop(session.id, operation_id=operation_id) is False
        release_terminal.set()
        assert (await _drain(replacement))[-1] == "turn.completed"

        after = await store.get_transcript(session.id)
        assert discarded == 0
        assert after.turns[-1].generation.status is GenerationStatus.COMPLETED
        assert service.availability()[0] is True
    finally:
        release_terminal.set()
        await service.close()


@pytest.mark.asyncio
async def test_delete_waits_for_terminal_settlement_before_removing_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    service, _runtime, store = await _service(tmp_path, FakeChatProvider())
    release_commit = asyncio.Event()
    try:
        entered_commit = asyncio.Event()
        original_finish_generation = store.finish_generation

        async def observed_finish_generation(*args, **kwargs):
            entered_commit.set()
            await release_commit.wait()
            return await original_finish_generation(*args, **kwargs)

        monkeypatch.setattr(store, "finish_generation", observed_finish_generation)
        session = await service.create_session()
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="77a5fc8a-6043-49f9-a291-b0545bd47ed8",
            text="finish before deletion",
        )
        await asyncio.wait_for(entered_commit.wait(), timeout=1)
        running = await service.get_session(session.id)

        delete_task = asyncio.create_task(
            service.delete_session(session.id, expected_revision=running.revision)
        )
        await asyncio.sleep(0.05)
        assert not delete_task.done()
        release_commit.set()
        await asyncio.wait_for(delete_task, timeout=1)

        with pytest.raises(ChatNotFoundError):
            await service.get_session(session.id)
        await stream.aclose()
    finally:
        release_commit.set()
        await service.close()


@pytest.mark.asyncio
async def test_cancelled_delete_request_still_finishes_accepted_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider(block_after_delta=True)
    service, _runtime, store = await _service(tmp_path, provider)
    release_settlement = asyncio.Event()
    subscription: ChatEventSubscriptionPort | None = None
    try:
        entered_settlement = asyncio.Event()
        original_finish_generation = store.finish_generation

        async def observed_finish_generation(*args, **kwargs):
            entered_settlement.set()
            await release_settlement.wait()
            return await original_finish_generation(*args, **kwargs)

        monkeypatch.setattr(store, "finish_generation", observed_finish_generation)
        session = await service.create_session()
        subscription, _active = await service.subscribe()
        operation_id = "cd352568-fefd-4324-8062-9dadc4efbba6"
        await service.send(
            session.id,
            expected_revision=session.revision,
            operation_id=operation_id,
            text="delete this chat",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        running = await service.get_session(session.id)

        request_task = asyncio.create_task(
            service.delete_session(session.id, expected_revision=running.revision)
        )
        await asyncio.wait_for(entered_settlement.wait(), timeout=1)
        request_task.cancel()
        await asyncio.sleep(0.05)
        assert not request_task.done()

        release_settlement.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(request_task, timeout=1)
        with pytest.raises(ChatNotFoundError):
            await service.get_session(session.id)

        events = subscription.__aiter__()
        while True:
            event = await asyncio.wait_for(anext(events), timeout=1)
            if event.event == "session.deleted":
                break
        fresh_subscription, active = await service.subscribe()
        try:
            assert active == ()
        finally:
            await fresh_subscription.aclose()
    finally:
        release_settlement.set()
        if subscription is not None:
            await subscription.aclose()
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("drain_feeds", [False, True])
async def test_close_waits_for_terminal_settlement(
    tmp_path: Path, monkeypatch, drain_feeds
):
    service, _runtime, store = await _service(tmp_path, FakeChatProvider())
    release_commit = asyncio.Event()
    service_closed = False
    reopened: SQLiteChatStore | None = None
    try:
        entered_commit = asyncio.Event()
        original_finish_generation = store.finish_generation

        async def observed_finish_generation(*args, **kwargs):
            entered_commit.set()
            await release_commit.wait()
            return await original_finish_generation(*args, **kwargs)

        monkeypatch.setattr(store, "finish_generation", observed_finish_generation)
        session = await service.create_session()
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="aa6d0128-de80-43ff-ab8c-aa1f7c4bf5ba",
            text="finish before shutdown",
        )
        await asyncio.wait_for(entered_commit.wait(), timeout=1)

        if drain_feeds:
            service.begin_shutdown()
            service.begin_shutdown()
            assert service.availability()[0] is False
            with pytest.raises(ChatUnavailableError):
                await service.subscribe()
            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(
                    anext(stream.subscription.__aiter__()), timeout=1
                )

        close_task = asyncio.create_task(service.close())
        await asyncio.sleep(0.05)
        assert not close_task.done()
        release_commit.set()
        await asyncio.wait_for(close_task, timeout=1)
        service_closed = True
        await stream.aclose()

        reopened = SQLiteChatStore(tmp_path / "chat.db", tmp_path / "chat.lock")
        await reopened.start()
        transcript = await reopened.get_transcript(session.id)
        assert transcript.turns[-1].generation.status is GenerationStatus.COMPLETED
    finally:
        release_commit.set()
        if reopened is not None:
            await reopened.close()
        if not service_closed:
            await service.close()


@pytest.mark.asyncio
async def test_cancelled_stop_waiter_does_not_cancel_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider(block_after_delta=True)
    service, _runtime, store = await _service(tmp_path, provider)
    release_settlement = asyncio.Event()
    try:
        entered_settlement = asyncio.Event()
        original_finish_generation = store.finish_generation

        async def observed_finish_generation(*args, **kwargs):
            entered_settlement.set()
            await release_settlement.wait()
            return await original_finish_generation(*args, **kwargs)

        monkeypatch.setattr(store, "finish_generation", observed_finish_generation)
        session = await service.create_session()
        operation_id = "13d97a77-aef0-4f2c-8415-1525ab70c427"
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id=operation_id,
            text="stop independently of this request",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)

        stop_task = asyncio.create_task(
            service.stop(session.id, operation_id=operation_id)
        )
        await asyncio.wait_for(entered_settlement.wait(), timeout=1)
        stop_task.cancel()
        await asyncio.sleep(0.05)
        assert not stop_task.done()

        release_settlement.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(stop_task, timeout=1)
        assert (await _drain(stream))[-1] == "turn.stopped"
        transcript = await store.get_transcript(session.id)
        assert transcript.turns[-1].generation.status is GenerationStatus.STOPPED
    finally:
        release_settlement.set()
        await service.close()


@pytest.mark.asyncio
async def test_terminal_event_precedes_best_effort_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    service, _runtime, store = await _service(tmp_path, FakeChatProvider())
    release_summary = asyncio.Event()
    fresh_subscription: ChatEventSubscriptionPort | None = None
    try:
        entered_summary = asyncio.Event()
        blocked_summary = False
        original_get_session_summary = store.get_session_summary

        async def observed_get_session_summary(session_id: str):
            nonlocal blocked_summary
            transcript = await store.get_transcript(session_id)
            completed = bool(
                transcript.turns
                and transcript.turns[-1].generation.status is GenerationStatus.COMPLETED
            )
            if completed and not blocked_summary:
                blocked_summary = True
                entered_summary.set()
                await release_summary.wait()
                raise ChatUnavailableError("Summary read failed.")
            return await original_get_session_summary(session_id)

        monkeypatch.setattr(
            store,
            "get_session_summary",
            observed_get_session_summary,
        )
        session = await service.create_session()
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="526be960-a16f-4d83-8f02-d51568003b60",
            text="publish before the summary",
        )

        assert (await _drain(stream))[-1] == "turn.completed"
        await asyncio.wait_for(entered_summary.wait(), timeout=1)
        fresh_subscription, active = await service.subscribe()
        assert active == ()

        current = await service.get_session(session.id)
        next_stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=current.revision,
            operation_id="ee7985a6-adf8-454f-9785-53d13c4eef8d",
            text="start while the old summary waits",
        )
        release_summary.set()
        assert (await _drain(next_stream))[-1] == "turn.completed"
        assert service.availability()[0] is True
    finally:
        release_summary.set()
        if fresh_subscription is not None:
            await fresh_subscription.aclose()
        await service.close()


@pytest.mark.asyncio
async def test_detail_snapshot_keeps_operation_owner_visible_through_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    release_commit = asyncio.Event()
    release_snapshot = asyncio.Event()
    try:
        entered_commit = asyncio.Event()
        committed = asyncio.Event()
        original_finish_generation = store.finish_generation

        async def observed_finish_generation(*args, **kwargs):
            entered_commit.set()
            await release_commit.wait()
            result = await original_finish_generation(*args, **kwargs)
            committed.set()
            return result

        monkeypatch.setattr(store, "finish_generation", observed_finish_generation)
        session = await service.create_session()
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="9e34cdcb-3d56-44f2-8921-a5d43cb0ed20",
            text="complete while detail loads",
        )
        await asyncio.wait_for(entered_commit.wait(), timeout=1)

        entered_snapshot = asyncio.Event()
        original_get_transcript = store.get_transcript

        async def observed_get_transcript(session_id: str):
            entered_snapshot.set()
            await release_snapshot.wait()
            return await original_get_transcript(session_id)

        monkeypatch.setattr(store, "get_transcript", observed_get_transcript)
        detail_task = asyncio.create_task(service.get_detail(session.id))
        await asyncio.wait_for(entered_snapshot.wait(), timeout=1)

        release_commit.set()
        await asyncio.wait_for(committed.wait(), timeout=1)
        release_snapshot.set()
        detail = await asyncio.wait_for(detail_task, timeout=1)
        await _drain(stream)

        assert not hasattr(detail, "active_operation")
        assert detail.session.revision > session.revision
    finally:
        release_commit.set()
        release_snapshot.set()
        await service.close()


@pytest.mark.asyncio
async def test_terminal_event_releases_owner_while_an_older_detail_read_waits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    release_commit = asyncio.Event()
    release_snapshot = asyncio.Event()
    try:
        entered_commit = asyncio.Event()
        original_finish_generation = store.finish_generation

        async def observed_finish_generation(*args, **kwargs):
            entered_commit.set()
            await release_commit.wait()
            return await original_finish_generation(*args, **kwargs)

        monkeypatch.setattr(store, "finish_generation", observed_finish_generation)
        session = await service.create_session()
        operation_id = "7f4dcfd1-bc69-4df1-9737-f434d357133a"
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id=operation_id,
            text="finish while detail owns the lifecycle lock",
        )
        await asyncio.wait_for(entered_commit.wait(), timeout=1)

        entered_snapshot = asyncio.Event()
        original_get_transcript = store.get_transcript

        async def observed_get_transcript(session_id: str):
            entered_snapshot.set()
            await release_snapshot.wait()
            return await original_get_transcript(session_id)

        monkeypatch.setattr(store, "get_transcript", observed_get_transcript)
        detail_task = asyncio.create_task(service.get_detail(session.id))
        await asyncio.wait_for(entered_snapshot.wait(), timeout=1)

        release_commit.set()
        assert (await asyncio.wait_for(_drain(stream), timeout=1))[-1] == (
            "turn.completed"
        )
        release_snapshot.set()
        older_detail = await asyncio.wait_for(detail_task, timeout=1)
        assert not hasattr(older_detail, "active_operation")

        current_detail = await service.get_detail(session.id)
        assert not hasattr(current_detail, "active_operation")

        current = await service.get_session(session.id)
        next_stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=current.revision,
            operation_id="87f5dd83-ee78-48d8-82f5-b65cd91034fc",
            text="the next operation can start",
        )
        await _drain(next_stream)
    finally:
        release_commit.set()
        release_snapshot.set()
        await service.close()


@pytest.mark.asyncio
async def test_disconnect_after_durable_completion_cannot_downgrade_status(
    tmp_path: Path,
):
    provider = BackpressuredCompletionProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="80354cee-d3b1-4760-963f-9cc7a1558ffc",
            text="fill the event queue",
        )

        async def wait_for_completion() -> None:
            while True:
                transcript = await store.get_transcript(session.id)
                if (
                    transcript.turns
                    and transcript.turns[0].generation.status
                    is GenerationStatus.COMPLETED
                ):
                    return
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_completion(), timeout=1)
        await stream.aclose()

        generation = (await store.get_transcript(session.id)).turns[0].generation
        assert generation.status is GenerationStatus.COMPLETED
        assert generation.stop_reason == "end_turn"
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation_kind", ["send", "regenerate"])
async def test_context_failure_compacts_once_and_retries_same_operation(
    tmp_path: Path,
    operation_kind: str,
) -> None:
    provider = ContextRecoveryProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await _seed_completed_turns(
            service,
            await service.create_session(),
            count=2,
        )
        prior_generation_id = (
            (await store.get_transcript(session.id)).turns[-1].generation.id
        )
        provider.generation_attempts = 0
        provider.summary_batch_sizes.clear()
        provider.generation_failures = 1

        if operation_kind == "send":
            operation = await _observe_call(
                service,
                service.send,
                session.id,
                expected_revision=session.revision,
                operation_id="00000000-0000-4000-8000-000000000100",
                text="recover this send",
            )
        else:
            operation = await _observe_call(
                service,
                service.regenerate,
                session.id,
                expected_revision=session.revision,
                operation_id="00000000-0000-4000-8000-000000000101",
            )
        events = await _drain(operation)

        transcript = await store.get_transcript(session.id)
        generation = transcript.turns[-1].generation
        assert events.count("turn.started") == 1
        assert events.count("compaction.started") == 1
        assert events.count("compaction.completed") == 1
        assert events[-1] == "turn.completed"
        assert "turn.failed" not in events
        assert provider.generation_attempts == 2
        assert provider.summary_batch_sizes == [1]
        assert transcript.compaction is not None
        assert generation.status is GenerationStatus.COMPLETED
        if operation_kind == "send":
            assert len(transcript.turns) == 3
        else:
            assert len(transcript.turns) == 2
            assert generation.id != prior_generation_id
            generation_requests = [
                request
                for request in provider.requests
                if not str(request.system).startswith("Summarize")
            ]
            retried_messages = generation_requests[-1].messages
            assert all(message.role != "assistant" for message in retried_messages)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_mixed_fallback_failures_do_not_trigger_context_recovery(
    tmp_path: Path,
) -> None:
    provider = MixedFallbackFailureProvider()
    service, runtime, store = await _service(tmp_path, provider)
    try:
        session = await _seed_completed_turns(
            service,
            await service.create_session(),
            count=2,
        )
        runtime.settings = runtime.settings.model_copy(
            update={"model_fallbacks": ("groq/fallback",)}
        )
        provider.fail_candidates = True
        provider.summary_batch_sizes.clear()

        operation = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="00000000-0000-4000-8000-000000000105",
            text="do not compact for mixed failures",
        )
        events = await _drain(operation)

        transcript = await store.get_transcript(session.id)
        generation = transcript.turns[-1].generation
        assert events.count("compaction.started") == 0
        assert events[-1] == "turn.failed"
        assert transcript.compaction is None
        assert provider.summary_batch_sizes == []
        assert generation.status is GenerationStatus.FAILED
        assert generation.error_code == FailureKind.CONTEXT_WINDOW_EXCEEDED.value
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_retry_context_failure_reuses_failed_generation_after_compaction(
    tmp_path: Path,
) -> None:
    provider = ContextRecoveryProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await _seed_completed_turns(
            service,
            await service.create_session(),
            count=1,
        )
        provider.generation_failures = 1
        provider.generation_failure_after_segment = True
        failed = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="00000000-0000-4000-8000-000000000110",
            text="fail after output starts",
        )
        assert (await _drain(failed))[-1] == "turn.failed"
        transcript = await store.get_transcript(session.id)
        failed_generation_id = transcript.turns[-1].generation.id
        assert transcript.compaction is None

        session = transcript.session
        provider.generation_attempts = 0
        provider.summary_batch_sizes.clear()
        provider.generation_failure_after_segment = False
        provider.generation_failures = 1
        retry = await _observe_call(
            service,
            service.retry,
            session.id,
            expected_revision=session.revision,
            operation_id="00000000-0000-4000-8000-000000000111",
        )
        events = await _drain(retry)

        transcript = await store.get_transcript(session.id)
        generation = transcript.turns[-1].generation
        assert events.count("turn.started") == 1
        assert events.count("compaction.started") == 1
        assert events.count("compaction.completed") == 1
        assert events[-1] == "turn.completed"
        assert generation.id == failed_generation_id
        assert generation.status is GenerationStatus.COMPLETED
        assert provider.generation_attempts == 2
        assert provider.summary_batch_sizes == [1]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_second_context_failure_finishes_as_one_failed_generation(
    tmp_path: Path,
) -> None:
    provider = ContextRecoveryProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await _seed_completed_turns(
            service,
            await service.create_session(),
            count=2,
        )
        provider.generation_attempts = 0
        provider.generation_failures = 2
        operation = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="00000000-0000-4000-8000-000000000120",
            text="still too large after compaction",
        )
        events = await _drain(operation)

        transcript = await store.get_transcript(session.id)
        generation = transcript.turns[-1].generation
        assert events.count("turn.started") == 1
        assert events.count("compaction.started") == 1
        assert events.count("compaction.completed") == 1
        assert events.count("turn.failed") == 1
        assert provider.generation_attempts == 2
        assert generation.status is GenerationStatus.FAILED
        assert generation.error_code == FailureKind.CONTEXT_WINDOW_EXCEEDED.value
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_uncompactable_context_failure_keeps_canonical_error(
    tmp_path: Path,
) -> None:
    provider = ContextRecoveryProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        provider.generation_failures = 1
        operation = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="00000000-0000-4000-8000-000000000125",
            text="one oversized turn",
        )
        events = await _drain(operation)

        transcript = await store.get_transcript(session.id)
        generation = transcript.turns[-1].generation
        assert events.count("compaction.started") == 0
        assert events[-1] == "turn.failed"
        assert transcript.compaction is None
        assert generation.error_code == FailureKind.CONTEXT_WINDOW_EXCEEDED.value
        assert generation.error_message == (
            "Provider input exceeds the model context window."
        )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stop_during_reactive_compaction_stops_the_original_generation(
    tmp_path: Path,
) -> None:
    provider = ContextRecoveryProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await _seed_completed_turns(
            service,
            await service.create_session(),
            count=2,
        )
        provider.generation_failures = 1
        provider.block_summary = True
        provider.summary_started.clear()
        operation = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="00000000-0000-4000-8000-000000000130",
            text="stop while compacting",
        )
        await asyncio.wait_for(provider.summary_started.wait(), timeout=1)

        assert await service.stop(
            session.id,
            operation_id=operation.acknowledgement.operation_id,
        )
        events = await _drain(operation)

        transcript = await store.get_transcript(session.id)
        assert events.count("compaction.started") == 1
        assert events[-1] == "turn.stopped"
        assert transcript.compaction is None
        assert transcript.turns[-1].generation.status is GenerationStatus.STOPPED
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_unknown_context_compaction_halves_only_whole_exchange_batches(
    tmp_path: Path,
) -> None:
    provider = ContextRecoveryProvider()
    runtime = FakeRuntime(provider, context_window_tokens=None)
    store = SQLiteChatStore(tmp_path / "chat.db", tmp_path / "chat.lock")
    service = ChatService(runtime, store)
    await service.start()
    try:
        session = await _seed_completed_turns(
            service,
            await service.create_session(),
            count=8,
        )
        provider.summary_batch_limit = 2
        provider.summary_batch_sizes.clear()
        compact = await _observe_call(
            service,
            service.compact,
            session.id,
            expected_revision=session.revision,
            operation_id="00000000-0000-4000-8000-000000000140",
        )

        assert (await _drain(compact))[-1] == "compaction.completed"
        assert provider.summary_batch_sizes == [4, 2, 2]
        transcript = await store.get_transcript(session.id)
        assert transcript.compaction is not None
        assert transcript.compaction.covered_through_sequence == 4
        assert len(transcript.turns) == 8
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_unknown_context_single_exchange_failure_is_actionable(
    tmp_path: Path,
) -> None:
    provider = ContextRecoveryProvider()
    runtime = FakeRuntime(provider, context_window_tokens=None)
    store = SQLiteChatStore(tmp_path / "chat.db", tmp_path / "chat.lock")
    service = ChatService(runtime, store)
    await service.start()
    try:
        session = await _seed_completed_turns(
            service,
            await service.create_session(),
            count=2,
        )
        provider.summary_batch_limit = 0
        provider.summary_batch_sizes.clear()
        compact = await _observe_call(
            service,
            service.compact,
            session.id,
            expected_revision=session.revision,
            operation_id="00000000-0000-4000-8000-000000000150",
        )

        assert (await _drain(compact))[-1] == "compaction.failed"
        assert provider.summary_batch_sizes == [1]
        assert (await store.get_transcript(session.id)).compaction is None
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_unknown_context_non_context_summary_failure_is_not_resized(
    tmp_path: Path,
) -> None:
    provider = ContextRecoveryProvider()
    runtime = FakeRuntime(provider, context_window_tokens=None)
    store = SQLiteChatStore(tmp_path / "chat.db", tmp_path / "chat.lock")
    service = ChatService(runtime, store)
    await service.start()
    try:
        session = await _seed_completed_turns(
            service,
            await service.create_session(),
            count=6,
        )
        provider.summary_failure = ExecutionFailure(
            kind=FailureKind.UPSTREAM,
            status_code=502,
            message="provider failed",
            retryable=False,
        )
        provider.summary_batch_sizes.clear()
        compact = await _observe_call(
            service,
            service.compact,
            session.id,
            expected_revision=session.revision,
            operation_id="00000000-0000-4000-8000-000000000160",
        )

        assert (await _drain(compact))[-1] == "compaction.failed"
        assert provider.summary_batch_sizes == [2]
        assert (await store.get_transcript(session.id)).compaction is None
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stop_during_compaction_commit_waits_and_reports_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    blocker: sqlite3.Connection | None = None
    try:
        session = await service.create_session()
        for index, operation_id in enumerate(
            (
                "08ea4712-3732-4626-9590-ac78cd273982",
                "35d15a22-88e4-476f-aec2-9c0cdfb5cb87",
            )
        ):
            stream = await _observe_call(
                service,
                service.send,
                session.id,
                expected_revision=session.revision,
                operation_id=operation_id,
                text=f"turn {index}",
            )
            await _drain(stream)
            session = await service.get_session(session.id)

        entered_commit = asyncio.Event()
        original_upsert_compaction = store.upsert_compaction

        async def observed_upsert_compaction(*args, **kwargs):
            entered_commit.set()
            return await original_upsert_compaction(*args, **kwargs)

        monkeypatch.setattr(store, "upsert_compaction", observed_upsert_compaction)
        blocker = sqlite3.connect(tmp_path / "chat.db", isolation_level=None)
        blocker.execute("PRAGMA journal_mode = WAL")
        blocker.execute("BEGIN IMMEDIATE")

        operation_id = "b15f0721-acd3-482f-812d-28d2d3cc568b"
        compact = await _observe_call(
            service,
            service.compact,
            session.id,
            expected_revision=session.revision,
            operation_id=operation_id,
        )
        await asyncio.wait_for(entered_commit.wait(), timeout=1)
        stop_task = asyncio.create_task(
            service.stop(session.id, operation_id=operation_id)
        )
        assert await asyncio.wait_for(stop_task, timeout=1) is False

        blocker.commit()
        blocker.close()
        blocker = None
        events = await _drain(compact)

        transcript = await store.get_transcript(session.id)
        assert events[-1] == "compaction.completed"
        assert transcript.compaction is not None
        assert transcript.compaction.covered_through_sequence == 1
    finally:
        if blocker is not None:
            blocker.rollback()
            blocker.close()
        await service.close()


@pytest.mark.asyncio
async def test_compaction_retries_an_ambiguous_commit_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        for operation_id, text in (
            ("110ce497-5d5f-4659-b2c7-7ae4bf362900", "first"),
            ("64de0c33-c956-4ac2-b172-9f9fc5caf13b", "second"),
        ):
            stream = await _observe_call(
                service,
                service.send,
                session.id,
                expected_revision=session.revision,
                operation_id=operation_id,
                text=text,
            )
            assert (await _drain(stream))[-1] == "turn.completed"
            session = await service.get_session(session.id)

        revision_before_compaction = session.revision
        attempts = 0
        original_upsert_compaction = store.upsert_compaction

        async def ambiguous_upsert_compaction(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            result = await original_upsert_compaction(*args, **kwargs)
            if attempts == 1:
                raise ChatUnavailableError("Chat commit result was unavailable.")
            return result

        monkeypatch.setattr(store, "upsert_compaction", ambiguous_upsert_compaction)
        compact = await _observe_call(
            service,
            service.compact,
            session.id,
            expected_revision=session.revision,
            operation_id="861cba8a-fca4-47eb-84cb-de1a7059714c",
        )

        assert (await _drain(compact))[-1] == "compaction.completed"
        transcript = await store.get_transcript(session.id)
        assert attempts == 2
        assert transcript.compaction is not None
        assert transcript.session.revision == revision_before_compaction + 1
        assert service.availability()[0] is True
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_manual_compaction_keeps_full_transcript_and_adds_checkpoint(
    tmp_path: Path,
):
    provider = FakeChatProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        first = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="e90f819b-a835-4c1f-80d7-6e7c17c429a7",
            text="first",
        )
        await _drain(first)
        session = await service.get_session(session.id)
        second = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="8de64c2a-34c8-4b30-a880-37d3678c62de",
            text="second",
        )
        await _drain(second)
        session = await service.get_session(session.id)
        compact = await _observe_call(
            service,
            service.compact,
            session.id,
            expected_revision=session.revision,
            operation_id="b5c9bc6f-3b74-4218-bac5-17b6813f443a",
        )
        events = await _drain(compact)

        transcript = await store.get_transcript(session.id)
        assert events[-1] == "compaction.completed"
        assert len(transcript.turns) == 2
        assert transcript.compaction is not None
        assert transcript.compaction.covered_through_sequence == 1
        assert transcript.compaction.summary == "summary"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_manual_compaction_rejects_incomplete_summary_stream(tmp_path: Path):
    provider = FakeChatProvider(truncate_summary=True)
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        for operation_id, text in (
            ("560c7a0f-c074-489b-8b90-e7e031577716", "first"),
            ("d77427c1-8417-4ab5-a380-94860c778db8", "second"),
        ):
            stream = await _observe_call(
                service,
                service.send,
                session.id,
                expected_revision=session.revision,
                operation_id=operation_id,
                text=text,
            )
            await _drain(stream)
            session = await service.get_session(session.id)

        compact = await _observe_call(
            service,
            service.compact,
            session.id,
            expected_revision=session.revision,
            operation_id="8895d9ae-c896-4af2-be44-d6328e1da736",
        )
        events = await _drain(compact)

        transcript = await store.get_transcript(session.id)
        assert events[-1] == "compaction.failed"
        assert transcript.compaction is None
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_send_auto_compacts_without_removing_original_turns(tmp_path: Path):
    provider = FakeChatProvider()
    runtime = FakeRuntime(provider, context_window_tokens=40_000)
    store = SQLiteChatStore(tmp_path / "chat.db", tmp_path / "chat.lock")
    service = ChatService(runtime, store)
    await service.start()
    try:
        session = await service.create_session()
        operation_ids = (
            "4e542b5b-8386-46d2-8643-67a523c216f0",
            "86293f1e-2899-47b9-8590-27450fc00989",
            "dd945983-b49c-4749-83b3-3051d3998bc2",
            "3a4f8108-2cf4-45c1-b412-56a8551e7f8b",
        )
        for operation_id in operation_ids:
            stream = await _observe_call(
                service,
                service.send,
                session.id,
                expected_revision=session.revision,
                operation_id=operation_id,
                text="token " * 5_000,
            )
            await _drain(stream)
            session = await service.get_session(session.id)

        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id="195faf19-ea0a-4633-ad81-a09734f5e17c",
            text="token " * 5_000,
        )
        events = await _drain(stream)

        transcript = await store.get_transcript(session.id)
        assert "compaction.started" in events
        assert "compaction.completed" in events
        assert len(transcript.turns) == 5
        assert transcript.compaction is not None
        assert transcript.compaction.covered_through_sequence >= 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stop_during_auto_compaction_commit_keeps_checkpoint_without_new_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    provider = FakeChatProvider()
    runtime = FakeRuntime(provider, context_window_tokens=40_000)
    store = SQLiteChatStore(tmp_path / "chat.db", tmp_path / "chat.lock")
    service = ChatService(runtime, store)
    await service.start()
    blocker: sqlite3.Connection | None = None
    try:
        session = await service.create_session()
        for operation_id in (
            "d3bb405e-245f-4dbf-bbdd-b72508926367",
            "d4fec5fe-a75a-4752-8c33-f51fd105774f",
            "79cd9bd9-df33-4ed4-8a5c-5e34770a30f7",
            "71ac3c34-7be1-4c26-b69e-f6f7193fa353",
        ):
            stream = await _observe_call(
                service,
                service.send,
                session.id,
                expected_revision=session.revision,
                operation_id=operation_id,
                text="token " * 5_000,
            )
            await _drain(stream)
            session = await service.get_session(session.id)

        entered_commit = asyncio.Event()
        original_upsert_compaction = store.upsert_compaction

        async def observed_upsert_compaction(*args, **kwargs):
            entered_commit.set()
            return await original_upsert_compaction(*args, **kwargs)

        monkeypatch.setattr(store, "upsert_compaction", observed_upsert_compaction)
        blocker = sqlite3.connect(tmp_path / "chat.db", isolation_level=None)
        blocker.execute("PRAGMA journal_mode = WAL")
        blocker.execute("BEGIN IMMEDIATE")

        operation_id = "443b12d9-8989-4bb5-a609-c9c2c0e2ecf5"
        stream = await _observe_call(
            service,
            service.send,
            session.id,
            expected_revision=session.revision,
            operation_id=operation_id,
            text="token " * 5_000,
        )
        await asyncio.wait_for(entered_commit.wait(), timeout=1)
        stop_task = asyncio.create_task(
            service.stop(session.id, operation_id=operation_id)
        )
        await asyncio.sleep(0.05)
        assert not stop_task.done()

        blocker.commit()
        blocker.close()
        blocker = None
        assert await asyncio.wait_for(stop_task, timeout=1) is True
        events = await _drain(stream)

        transcript = await store.get_transcript(session.id)
        assert events[-2:] == ["compaction.completed", "turn.stopped"]
        assert transcript.compaction is not None
        assert len(transcript.turns) == 4
    finally:
        if blocker is not None:
            blocker.rollback()
            blocker.close()
        await service.close()


@pytest.mark.asyncio
async def test_storage_start_failure_disables_only_chat(tmp_path: Path):
    database_path = tmp_path / "chat.db"
    database_path.mkdir()
    store = SQLiteChatStore(database_path, tmp_path / "chat.lock")
    service = ChatService(FakeRuntime(FakeChatProvider()), store)

    await service.start()
    try:
        available, message = service.availability()
        assert available is False
        assert message == "Chat storage is unavailable."
        with pytest.raises(ChatUnavailableError, match="storage is unavailable"):
            await service.create_session()
    finally:
        await service.close()
