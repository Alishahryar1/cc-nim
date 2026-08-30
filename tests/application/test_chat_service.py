import asyncio
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from free_claude_code.application.chat import (
    ChatConflictError,
    ChatNotFoundError,
    ChatService,
    ChatUnavailableError,
    GenerationStatus,
)
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.ports import ProviderPort, RequestRuntimeLease
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import MessagesRequest
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.runtime.chat_sqlite import SQLiteChatStore


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
    def __init__(self, settings: Settings, provider: FakeChatProvider) -> None:
        self._settings = settings
        self._provider = provider
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
        context_window_tokens: int = 100_000,
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

    async def acquire(self) -> RequestRuntimeLease:
        lease = FakeLease(self.settings, self.provider)
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


async def _drain(stream) -> list[str]:
    events = [event.event async for event in stream]
    await stream.aclose()
    return events


@pytest.mark.asyncio
async def test_send_streams_and_persists_interleaved_segments(tmp_path: Path):
    provider = FakeChatProvider()
    service, runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        stream = await service.send(
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
async def test_closing_initiating_stream_cancels_and_persists_partial_answer(
    tmp_path: Path,
):
    provider = FakeChatProvider(block_after_delta=True)
    service, runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        stream = await service.send(
            session.id,
            expected_revision=session.revision,
            operation_id="9781df8c-aa92-422e-97d9-e9ea7f542b89",
            text="hello",
        )
        collector = asyncio.create_task(_drain(stream))
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        await stream.aclose()
        events = await asyncio.wait_for(collector, timeout=1)

        generation = (await store.get_transcript(session.id)).turns[0].generation
        assert events[-1] == "turn.stopped"
        assert generation.status is GenerationStatus.STOPPED
        assert generation.segments[-1].text == "answer"
        assert runtime.leases[0].released == 1
        assert provider.closed == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_second_operation_on_same_session_is_rejected(tmp_path: Path):
    provider = FakeChatProvider(block_after_delta=True)
    service, _runtime, _store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        first = await service.send(
            session.id,
            expected_revision=session.revision,
            operation_id="3e363fbc-25ee-414e-a954-02d11990497f",
            text="first",
        )
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        running = await service.get_session(session.id)
        with pytest.raises(ChatConflictError, match="active operation"):
            await service.send(
                session.id,
                expected_revision=running.revision,
                operation_id="223ad9ac-e830-42ec-b73d-b1cbd81511c4",
                text="second",
            )
        await first.aclose()
    finally:
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
        stream = await service.send(
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
        stream = await service.send(
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
        initial_stream = await service.send(
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
        retry_stream = await service.retry(
            session.id,
            expected_revision=current.revision,
            operation_id="38dcb3e5-04b7-4ecf-bc8b-7c72550839b0",
        )
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.sleep(0.05)

        close_task = asyncio.create_task(retry_stream.aclose())
        await asyncio.sleep(0.05)
        waited_for_commit = not close_task.done()
        blocker.commit()
        blocker.close()
        blocker = None
        await asyncio.wait_for(close_task, timeout=1)

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
        stream = await service.send(
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

        close_task = asyncio.create_task(stream.aclose())
        await asyncio.wait_for(entered_cleanup.wait(), timeout=1)
        stop_task = asyncio.create_task(
            service.stop(session.id, operation_id=operation_id)
        )
        await asyncio.sleep(0.05)

        assert not close_task.done()
        assert not stop_task.done()
        blocker.commit()
        blocker.close()
        blocker = None
        await asyncio.wait_for(close_task, timeout=1)
        assert await asyncio.wait_for(stop_task, timeout=1) is True

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
        stream = await service.send(
            session.id,
            expected_revision=session.revision,
            operation_id=operation_id,
            text="finish this answer",
        )
        await asyncio.wait_for(entered_commit.wait(), timeout=1)

        stop_task = asyncio.create_task(
            service.stop(session.id, operation_id=operation_id)
        )
        await asyncio.sleep(0.05)
        assert not stop_task.done()
        release_commit.set()

        assert await asyncio.wait_for(stop_task, timeout=1) is True
        events = await _drain(stream)
        generation = (await store.get_transcript(session.id)).turns[0].generation
        assert events[-1] == "turn.completed"
        assert generation.status is GenerationStatus.COMPLETED
    finally:
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
        stream = await service.send(
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

        assert detail.active_operation is True
        assert detail.session.revision > session.revision
    finally:
        release_commit.set()
        release_snapshot.set()
        await service.close()


@pytest.mark.asyncio
async def test_cancellation_while_release_waits_cannot_strand_operation_owner(
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
        stream = await service.send(
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
        events = stream.__aiter__()
        while (
            await asyncio.wait_for(anext(events), timeout=1)
        ).event != "turn.completed":
            pass

        close_task = asyncio.create_task(stream.aclose())
        await asyncio.sleep(0)
        release_snapshot.set()
        await asyncio.wait_for(detail_task, timeout=1)
        await asyncio.wait_for(close_task, timeout=1)

        current = await service.get_session(session.id)
        next_stream = await service.send(
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
        stream = await service.send(
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
            stream = await service.send(
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
        compact = await service.compact(
            session.id,
            expected_revision=session.revision,
            operation_id=operation_id,
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
async def test_manual_compaction_keeps_full_transcript_and_adds_checkpoint(
    tmp_path: Path,
):
    provider = FakeChatProvider()
    service, _runtime, store = await _service(tmp_path, provider)
    try:
        session = await service.create_session()
        first = await service.send(
            session.id,
            expected_revision=session.revision,
            operation_id="e90f819b-a835-4c1f-80d7-6e7c17c429a7",
            text="first",
        )
        await _drain(first)
        session = await service.get_session(session.id)
        second = await service.send(
            session.id,
            expected_revision=session.revision,
            operation_id="8de64c2a-34c8-4b30-a880-37d3678c62de",
            text="second",
        )
        await _drain(second)
        session = await service.get_session(session.id)
        compact = await service.compact(
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
            stream = await service.send(
                session.id,
                expected_revision=session.revision,
                operation_id=operation_id,
                text=text,
            )
            await _drain(stream)
            session = await service.get_session(session.id)

        compact = await service.compact(
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
        )
        for operation_id in operation_ids:
            stream = await service.send(
                session.id,
                expected_revision=session.revision,
                operation_id=operation_id,
                text="token " * 5_000,
            )
            await _drain(stream)
            session = await service.get_session(session.id)

        stream = await service.send(
            session.id,
            expected_revision=session.revision,
            operation_id="195faf19-ea0a-4633-ad81-a09734f5e17c",
            text="token " * 5_000,
        )
        events = await _drain(stream)

        transcript = await store.get_transcript(session.id)
        assert "compaction.started" in events
        assert "compaction.completed" in events
        assert len(transcript.turns) == 4
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
        ):
            stream = await service.send(
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
        stream = await service.send(
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
        assert len(transcript.turns) == 3
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
