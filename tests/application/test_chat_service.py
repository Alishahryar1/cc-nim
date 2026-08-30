import asyncio
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
    def __init__(self, *, block_after_delta: bool = False) -> None:
        self.block_after_delta = block_after_delta
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
