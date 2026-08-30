"""Application lifecycle and commands for durable local Chat Sessions."""

import asyncio
import contextlib
import sys
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum

from loguru import logger

from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.ports import RequestRuntimeLease, RequestRuntimePort
from free_claude_code.application.routing import ProviderModelTarget
from free_claude_code.core.anthropic import (
    ContentBlockText,
    MessagesResponse,
    aggregate_anthropic_sse_to_message,
)
from free_claude_code.core.anthropic.streaming import AnthropicSSEDecoder
from free_claude_code.core.anthropic.tokens import estimate_text_tokens
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.trace import close_stream_input

from .context import (
    ChatContextBuilder,
    PreparedChatRequest,
    compaction_target_tokens,
)
from .models import (
    DEFAULT_CHAT_SYSTEM_PROMPT,
    ChatCompaction,
    ChatConflictError,
    ChatContextEstimate,
    ChatModelOption,
    ChatPreferences,
    ChatReasoning,
    ChatSegment,
    ChatSession,
    ChatSessionDetail,
    ChatSessionPage,
    ChatStreamEvent,
    ChatTranscript,
    ChatTurn,
    ChatUnavailableError,
    ChatValidationError,
    GenerationStatus,
    SegmentKind,
)
from .ports import ChatOperationStream, ChatStorePort

_EVENT_QUEUE_SIZE = 128
_PERSIST_INTERVAL_SECONDS = 0.25
_PERSIST_CHARACTER_THRESHOLD = 4_096
_TURN_PAGE_LIMIT = 50
_SESSION_PAGE_LIMIT = 25
_STORAGE_RESTART_MESSAGE = (
    "Chat storage became unavailable. Restart FCC to repair Chat Sessions."
)


class _OperationKind(StrEnum):
    SEND = "send"
    RETRY = "retry"
    REGENERATE = "regenerate"
    COMPACT = "compact"


class _CancellationReason(StrEnum):
    STOPPED = "stopped"
    INTERRUPTED = "interrupted"
    DELETED = "deleted"


class _TerminalPersistenceError(ChatUnavailableError):
    """A bounded terminal write failed and requires startup repair."""


@dataclass(slots=True)
class _ActiveOperation:
    session_id: str
    operation_id: str
    kind: _OperationKind
    queue: asyncio.Queue[ChatStreamEvent | None] = field(
        default_factory=lambda: asyncio.Queue(maxsize=_EVENT_QUEUE_SIZE)
    )
    task: asyncio.Task[None] | None = None
    cancellation_reason: _CancellationReason | None = None
    generation_id: str | None = None
    regeneration: bool = False
    segments: list[ChatSegment] = field(default_factory=list)
    event_sequence: int = 0
    actual_model: str | None = None
    terminal_emitted: bool = False


class _OperationStream:
    """Initiating browser view of one application-owned operation."""

    def __init__(self, service: ChatService, active: _ActiveOperation) -> None:
        self._service = service
        self._active = active
        self.operation_id = active.operation_id
        self._closed = False

    def __aiter__(self) -> AsyncIterator[ChatStreamEvent]:
        return self._events()

    async def _events(self) -> AsyncIterator[ChatStreamEvent]:
        while True:
            event = await self._active.queue.get()
            if event is None:
                return
            yield event

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        task = self._active.task
        if task is not None and not task.done():
            await self._service._cancel_active(
                self._active,
                reason=_CancellationReason.STOPPED,
            )


class ChatService:
    """Single owner of Chat session commands and active provider work."""

    def __init__(self, runtime: RequestRuntimePort, store: ChatStorePort) -> None:
        self._runtime = runtime
        self._store = store
        self._context = ChatContextBuilder(runtime)
        self._active: dict[str, _ActiveOperation] = {}
        self._deleting: set[str] = set()
        self._active_lock = asyncio.Lock()
        self._started = False
        self._accepting = False
        self._unavailable_message: str | None = "Chat Sessions is starting."

    async def start(self) -> None:
        if self._started:
            return
        try:
            await self._store.start()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._unavailable_message = (
                str(exc)
                if isinstance(exc, ChatUnavailableError)
                else "Chat storage could not be opened."
            )
            logger.warning(
                "Chat Sessions unavailable: exc_type={}",
                type(exc).__name__,
            )
            return
        self._started = True
        self._accepting = True
        self._unavailable_message = None

    async def close(self) -> None:
        self._accepting = False
        async with self._active_lock:
            active = tuple(self._active.values())
        await asyncio.gather(
            *(
                self._cancel_active(
                    operation,
                    reason=_CancellationReason.INTERRUPTED,
                )
                for operation in active
            )
        )
        await self._store.close()
        self._started = False
        self._unavailable_message = "Chat Sessions is stopped."

    def availability(self) -> tuple[bool, str | None]:
        return self._started and self._accepting, self._unavailable_message

    def models(self) -> tuple[ChatModelOption, ...]:
        return self._context.models()

    async def preferences(self) -> ChatPreferences:
        self._require_available()
        return await self._store.load_preferences()

    async def save_system_prompt(self, value: str) -> ChatPreferences:
        self._require_available()
        return await self._store.save_system_prompt(value)

    async def reset_system_prompt(self) -> ChatPreferences:
        self._require_available()
        return await self._store.save_system_prompt(DEFAULT_CHAT_SYSTEM_PROMPT)

    async def create_session(self) -> ChatSession:
        self._require_available()
        preferences = await self._store.load_preferences()
        options = self.models()
        if not options:
            raise ChatUnavailableError(
                "No configured models are available for Chat Sessions."
            )
        available = {option.model_ref: option for option in options}
        configured_default = self._runtime.current_settings().model
        model = preferences.last_model
        if model not in available:
            model = (
                configured_default
                if configured_default in available
                else options[0].model_ref
            )
        reasoning = preferences.last_reasoning
        if available[model].supports_reasoning is False:
            reasoning = ChatReasoning.OFF
        return await self._store.create_session(
            session_id=str(uuid.uuid4()),
            model=model,
            reasoning=reasoning,
        )

    async def list_sessions(
        self,
        *,
        query: str,
        cursor: tuple[int, str] | None,
        limit: int,
    ) -> ChatSessionPage:
        self._require_available()
        return await self._store.list_sessions(
            query=query,
            cursor=cursor,
            limit=max(1, min(_SESSION_PAGE_LIMIT, limit)),
        )

    async def get_session(self, session_id: str) -> ChatSession:
        self._require_available()
        return await self._store.get_session(_canonical_uuid(session_id, "session"))

    async def get_detail(self, session_id: str) -> ChatSessionDetail:
        self._require_available()
        session_id = _canonical_uuid(session_id, "session")
        async with self._active_lock:
            active_operation = session_id in self._active
            transcript = await self._store.get_transcript(session_id)
            prompt = (await self._store.load_preferences()).system_prompt
            context: ChatContextEstimate | None
            context_error: str | None = None
            try:
                context = self._context.prepare(
                    transcript,
                    system_prompt=prompt,
                ).estimate
            except ChatValidationError as exc:
                context = None
                context_error = str(exc)
            has_more = len(transcript.turns) > _TURN_PAGE_LIMIT
            turns = transcript.turns[-_TURN_PAGE_LIMIT:]
            next_before = turns[0].sequence if has_more and turns else None
            return ChatSessionDetail(
                session=transcript.session,
                turns=turns,
                next_before=next_before,
                compaction=transcript.compaction,
                context=context,
                context_error=context_error,
                active_operation=active_operation,
            )

    async def update_session(
        self,
        session_id: str,
        *,
        expected_revision: int,
        title: str | None,
        model: str | None,
        reasoning: ChatReasoning | None,
    ) -> ChatSession:
        self._require_available()
        session_id = _canonical_uuid(session_id, "session")
        if title is not None:
            title = " ".join(title.split())
            if not title:
                raise ChatValidationError("Chat title cannot be empty.")
            if len(title) > 200:
                raise ChatValidationError("Chat title cannot exceed 200 characters.")
        if model is not None:
            self._context.model(model)
        async with self._active_lock:
            if session_id in self._deleting:
                raise ChatConflictError("This chat is being deleted.")
            if (
                model is not None or reasoning is not None
            ) and session_id in self._active:
                raise ChatConflictError(
                    "Model and thinking cannot change while this chat is running."
                )
            return await self._store.update_session(
                session_id,
                expected_revision=expected_revision,
                title=title,
                model=model,
                reasoning=reasoning,
            )

    async def delete_session(self, session_id: str, *, expected_revision: int) -> None:
        self._require_available()
        session_id = _canonical_uuid(session_id, "session")
        async with self._active_lock:
            if session_id in self._deleting:
                raise ChatConflictError("This chat is already being deleted.")
            self._deleting.add(session_id)
            active = self._active.get(session_id)
        try:
            current = await self._store.get_session(session_id)
            _expect_revision(current, expected_revision)
            if active is not None:
                await self._cancel_active(active, reason=_CancellationReason.DELETED)
                current = await self._store.get_session(session_id)
            await self._store.delete_session(
                session_id,
                expected_revision=current.revision,
            )
        finally:
            async with self._active_lock:
                self._deleting.discard(session_id)

    async def get_turn_page(
        self,
        session_id: str,
        *,
        before_sequence: int | None,
        limit: int,
    ) -> tuple[tuple[ChatTurn, ...], int | None, ChatCompaction | None]:
        self._require_available()
        if before_sequence is not None and before_sequence <= 0:
            raise ChatValidationError("Turn cursor must be positive.")
        return await self._store.get_turn_page(
            _canonical_uuid(session_id, "session"),
            before_sequence=before_sequence,
            limit=max(1, min(_TURN_PAGE_LIMIT, limit)),
        )

    async def estimate(self, session_id: str, *, draft: str) -> ChatContextEstimate:
        self._require_available()
        transcript = await self._store.get_transcript(
            _canonical_uuid(session_id, "session")
        )
        prompt = (await self._store.load_preferences()).system_prompt
        return self._context.prepare(
            transcript,
            system_prompt=prompt,
            draft=draft,
        ).estimate

    async def send(
        self,
        session_id: str,
        *,
        expected_revision: int,
        operation_id: str,
        text: str,
    ) -> ChatOperationStream:
        self._require_available()
        session_id = _canonical_uuid(session_id, "session")
        operation_id = _canonical_uuid(operation_id, "operation")
        if not text.strip():
            raise ChatValidationError("Write a message before sending.")
        transcript = await self._store.get_transcript(session_id)
        _expect_revision(transcript.session, expected_revision)
        prompt = (await self._store.load_preferences()).system_prompt
        self._context.prepare(transcript, system_prompt=prompt, draft=text)
        return await self._start_operation(
            session_id,
            operation_id=operation_id,
            kind=_OperationKind.SEND,
            action=lambda active: self._run_send(
                active,
                expected_revision=expected_revision,
                text=text,
            ),
        )

    async def retry(
        self,
        session_id: str,
        *,
        expected_revision: int,
        operation_id: str,
    ) -> ChatOperationStream:
        self._require_available()
        session_id = _canonical_uuid(session_id, "session")
        operation_id = _canonical_uuid(operation_id, "operation")
        transcript = await self._store.get_transcript(session_id)
        _expect_revision(transcript.session, expected_revision)
        latest = _latest_turn(transcript)
        if latest.generation.status not in {
            GenerationStatus.STOPPED,
            GenerationStatus.INTERRUPTED,
            GenerationStatus.FAILED,
        }:
            raise ChatConflictError("Only the latest unfinished answer can be retried.")
        prompt = (await self._store.load_preferences()).system_prompt
        self._context.prepare(
            transcript,
            system_prompt=prompt,
            exclude_generation_id=latest.generation.id,
        )
        return await self._start_operation(
            session_id,
            operation_id=operation_id,
            kind=_OperationKind.RETRY,
            action=lambda active: self._run_retry(
                active,
                expected_revision=expected_revision,
            ),
        )

    async def regenerate(
        self,
        session_id: str,
        *,
        expected_revision: int,
        operation_id: str,
    ) -> ChatOperationStream:
        self._require_available()
        session_id = _canonical_uuid(session_id, "session")
        operation_id = _canonical_uuid(operation_id, "operation")
        transcript = await self._store.get_transcript(session_id)
        _expect_revision(transcript.session, expected_revision)
        latest = _latest_turn(transcript)
        if latest.generation.status is not GenerationStatus.COMPLETED:
            raise ChatConflictError("Only the latest completed answer can regenerate.")
        prompt = (await self._store.load_preferences()).system_prompt
        self._context.prepare(
            transcript,
            system_prompt=prompt,
            exclude_generation_id=latest.generation.id,
        )
        return await self._start_operation(
            session_id,
            operation_id=operation_id,
            kind=_OperationKind.REGENERATE,
            action=lambda active: self._run_regenerate(
                active,
                expected_revision=expected_revision,
            ),
        )

    async def compact(
        self,
        session_id: str,
        *,
        expected_revision: int,
        operation_id: str,
    ) -> ChatOperationStream:
        self._require_available()
        session_id = _canonical_uuid(session_id, "session")
        operation_id = _canonical_uuid(operation_id, "operation")
        transcript = await self._store.get_transcript(session_id)
        _expect_revision(transcript.session, expected_revision)
        if not self._context.can_compact(transcript):
            raise ChatValidationError(
                "There is not enough older conversation to compact."
            )
        return await self._start_operation(
            session_id,
            operation_id=operation_id,
            kind=_OperationKind.COMPACT,
            action=lambda active: self._run_manual_compaction(
                active,
                expected_revision=expected_revision,
            ),
        )

    async def stop(self, session_id: str, *, operation_id: str) -> bool:
        self._require_available()
        session_id = _canonical_uuid(session_id, "session")
        operation_id = _canonical_uuid(operation_id, "operation")
        async with self._active_lock:
            active = self._active.get(session_id)
        if active is None:
            return False
        if active.operation_id != operation_id:
            raise ChatConflictError("That operation is no longer active.")
        await self._cancel_active(active, reason=_CancellationReason.STOPPED)
        return True

    async def _start_operation(
        self,
        session_id: str,
        *,
        operation_id: str,
        kind: _OperationKind,
        action: Callable[[_ActiveOperation], Awaitable[None]],
    ) -> _OperationStream:
        async with self._active_lock:
            if not self._accepting:
                raise ChatUnavailableError("Chat Sessions is shutting down.")
            if session_id in self._deleting:
                raise ChatConflictError("This chat is being deleted.")
            if session_id in self._active:
                raise ChatConflictError("This chat already has an active operation.")
            active = _ActiveOperation(
                session_id=session_id,
                operation_id=operation_id,
                kind=kind,
            )
            self._active[session_id] = active
            active.task = asyncio.create_task(
                self._run_active(active, action),
                name=f"fcc-chat-{kind.value}-{operation_id}",
            )
        return _OperationStream(self, active)

    async def _run_active(
        self,
        active: _ActiveOperation,
        action: Callable[[_ActiveOperation], Awaitable[None]],
    ) -> None:
        try:
            await action(active)
        except asyncio.CancelledError:
            settlement = asyncio.create_task(
                self._handle_cancelled(active),
                name=f"fcc-chat-settle-cancelled-{active.operation_id}",
            )
            await _await_task_despite_cancellation(settlement)
        except Exception as exc:
            settlement = asyncio.create_task(
                self._handle_failure(active, exc),
                name=f"fcc-chat-settle-failed-{active.operation_id}",
            )
            await _await_task_despite_cancellation(settlement)
        finally:
            release_task = asyncio.create_task(
                self._release_active(active),
                name=f"fcc-chat-release-{active.operation_id}",
            )
            await _await_task_despite_cancellation(release_task)

    async def _run_send(
        self,
        active: _ActiveOperation,
        *,
        expected_revision: int,
        text: str,
    ) -> None:
        lease = await self._runtime.acquire(include_model_infos=True)
        try:
            builder = ChatContextBuilder(
                self._runtime,
                settings=lease.settings,
                model_infos=lease.model_infos,
            )
            prompt = (await self._store.load_preferences()).system_prompt
            transcript = await self._store.get_transcript(active.session_id)
            _expect_revision(transcript.session, expected_revision)
            prepared = builder.prepare(transcript, system_prompt=prompt, draft=text)
            if prepared.estimate.should_auto_compact:
                await self._compact_transcript(
                    active,
                    builder=builder,
                    lease=lease,
                    transcript=transcript,
                    prompt=prompt,
                    pending_draft=text,
                    excluded_generation_id=None,
                )
                transcript = await self._store.get_transcript(active.session_id)
                prepared = builder.prepare(
                    transcript,
                    system_prompt=prompt,
                    draft=text,
                )
            _require_request_fits(prepared.estimate)
            generation_id = str(uuid.uuid4())
            turn_id = str(uuid.uuid4())
            await _commit_generation_start(
                active,
                store=self._store,
                generation_id=generation_id,
                operation=self._store.begin_send(
                    active.session_id,
                    expected_revision=transcript.session.revision,
                    turn_id=turn_id,
                    generation_id=generation_id,
                    operation_id=active.operation_id,
                    user_text=text,
                    requested_model=transcript.session.model,
                    reasoning=transcript.session.reasoning,
                    effective_output_limit=prepared.routed.request.max_tokens or 1,
                ),
            )
            await self._emit(
                active,
                "turn.started",
                {
                    "turn_id": turn_id,
                    "generation_id": generation_id,
                    "revision": transcript.session.revision + 1,
                },
            )
            await self._execute_generation(active, prepared=prepared, lease=lease)
        finally:
            await lease.release()

    async def _run_retry(
        self,
        active: _ActiveOperation,
        *,
        expected_revision: int,
    ) -> None:
        lease = await self._runtime.acquire(include_model_infos=True)
        try:
            builder = ChatContextBuilder(
                self._runtime,
                settings=lease.settings,
                model_infos=lease.model_infos,
            )
            prompt = (await self._store.load_preferences()).system_prompt
            transcript = await self._store.get_transcript(active.session_id)
            _expect_revision(transcript.session, expected_revision)
            latest = _latest_turn(transcript)
            prepared = builder.prepare(
                transcript,
                system_prompt=prompt,
                exclude_generation_id=latest.generation.id,
            )
            if prepared.estimate.should_auto_compact:
                await self._compact_transcript(
                    active,
                    builder=builder,
                    lease=lease,
                    transcript=transcript,
                    prompt=prompt,
                    pending_draft=None,
                    excluded_generation_id=latest.generation.id,
                )
                transcript = await self._store.get_transcript(active.session_id)
                latest = _latest_turn(transcript)
                prepared = builder.prepare(
                    transcript,
                    system_prompt=prompt,
                    exclude_generation_id=latest.generation.id,
                )
            _require_request_fits(prepared.estimate)
            generation_id = latest.generation.id
            await _commit_generation_start(
                active,
                store=self._store,
                generation_id=generation_id,
                operation=self._store.begin_retry(
                    active.session_id,
                    expected_revision=transcript.session.revision,
                    requested_model=transcript.session.model,
                    reasoning=transcript.session.reasoning,
                    effective_output_limit=prepared.routed.request.max_tokens or 1,
                ),
            )
            await self._emit(
                active,
                "turn.started",
                {
                    "turn_id": latest.id,
                    "generation_id": generation_id,
                    "revision": transcript.session.revision + 1,
                },
            )
            await self._execute_generation(active, prepared=prepared, lease=lease)
        finally:
            await lease.release()

    async def _run_regenerate(
        self,
        active: _ActiveOperation,
        *,
        expected_revision: int,
    ) -> None:
        lease = await self._runtime.acquire(include_model_infos=True)
        try:
            builder = ChatContextBuilder(
                self._runtime,
                settings=lease.settings,
                model_infos=lease.model_infos,
            )
            prompt = (await self._store.load_preferences()).system_prompt
            transcript = await self._store.get_transcript(active.session_id)
            _expect_revision(transcript.session, expected_revision)
            latest = _latest_turn(transcript)
            prepared = builder.prepare(
                transcript,
                system_prompt=prompt,
                exclude_generation_id=latest.generation.id,
            )
            if prepared.estimate.should_auto_compact:
                await self._compact_transcript(
                    active,
                    builder=builder,
                    lease=lease,
                    transcript=transcript,
                    prompt=prompt,
                    pending_draft=None,
                    excluded_generation_id=latest.generation.id,
                )
                transcript = await self._store.get_transcript(active.session_id)
                latest = _latest_turn(transcript)
                prepared = builder.prepare(
                    transcript,
                    system_prompt=prompt,
                    exclude_generation_id=latest.generation.id,
                )
            _require_request_fits(prepared.estimate)
            generation_id = str(uuid.uuid4())
            await _commit_generation_start(
                active,
                store=self._store,
                generation_id=generation_id,
                regeneration=True,
                operation=self._store.begin_regenerate(
                    active.session_id,
                    expected_revision=transcript.session.revision,
                    generation_id=generation_id,
                    requested_model=transcript.session.model,
                    reasoning=transcript.session.reasoning,
                    effective_output_limit=prepared.routed.request.max_tokens or 1,
                ),
            )
            await self._emit(
                active,
                "turn.started",
                {
                    "turn_id": latest.id,
                    "generation_id": generation_id,
                    "revision": transcript.session.revision + 1,
                    "regeneration": True,
                },
            )
            await self._execute_generation(active, prepared=prepared, lease=lease)
        finally:
            await lease.release()

    async def _run_manual_compaction(
        self,
        active: _ActiveOperation,
        *,
        expected_revision: int,
    ) -> None:
        lease = await self._runtime.acquire(include_model_infos=True)
        try:
            builder = ChatContextBuilder(
                self._runtime,
                settings=lease.settings,
                model_infos=lease.model_infos,
            )
            transcript = await self._store.get_transcript(active.session_id)
            _expect_revision(transcript.session, expected_revision)
            prompt = (await self._store.load_preferences()).system_prompt
            await self._compact_transcript(
                active,
                builder=builder,
                lease=lease,
                transcript=transcript,
                prompt=prompt,
                pending_draft=None,
                excluded_generation_id=None,
            )
        finally:
            await lease.release()

    async def _execute_generation(
        self,
        active: _ActiveOperation,
        *,
        prepared: PreparedChatRequest,
        lease: RequestRuntimeLease,
    ) -> None:
        generation_id = active.generation_id
        if generation_id is None:
            raise RuntimeError("Chat generation was not initialized.")
        executor = ProviderExecutor(
            lease.resolve_provider,
            progress_timeout_seconds=lease.settings.provider_progress_timeout,
            generation_id=lease.generation_id,
            log_raw_payloads=lease.settings.log_raw_api_payloads,
        )

        async def selected(target_model: ProviderModelTarget) -> None:
            active.actual_model = target_model.provider_model_ref
            await self._store.set_generation_actual_model(
                generation_id, target_model.provider_model_ref
            )

        stream = executor.stream_messages(
            prepared.routed,
            raw_log_payload=prepared.routed.request.model_dump(mode="json"),
            request_id=active.operation_id,
            candidate_selected=selected,
        )
        decoder = AnthropicSSEDecoder()
        block_segments: dict[int, int] = {}
        saw_message_stop = False
        stop_reason: str | None = None
        last_flush = time.monotonic()
        pending_characters = 0
        try:
            async for chunk in stream:
                for event in decoder.feed(chunk):
                    (
                        delta_characters,
                        event_stop,
                        event_reason,
                    ) = await self._handle_sse_event(
                        active,
                        event_name=event.event,
                        data=event.data,
                        block_segments=block_segments,
                    )
                    pending_characters += delta_characters
                    saw_message_stop = saw_message_stop or event_stop
                    stop_reason = event_reason or stop_reason
                    now = time.monotonic()
                    if (
                        pending_characters >= _PERSIST_CHARACTER_THRESHOLD
                        or now - last_flush >= _PERSIST_INTERVAL_SECONDS
                    ):
                        await self._flush_segments(active)
                        last_flush = now
                        pending_characters = 0
            for event in decoder.finish():
                (
                    delta_characters,
                    event_stop,
                    event_reason,
                ) = await self._handle_sse_event(
                    active,
                    event_name=event.event,
                    data=event.data,
                    block_segments=block_segments,
                )
                pending_characters += delta_characters
                saw_message_stop = saw_message_stop or event_stop
                stop_reason = event_reason or stop_reason
        finally:
            await close_stream_input(
                stream,
                owner="chat_sessions",
                source="application",
                preserved_error=sys.exception(),
            )

        if not saw_message_stop:
            raise ChatValidationError("The provider stream ended before completing.")

        async def commit_completion() -> ChatSession:
            await self._flush_terminal_segments(active)
            if active.regeneration:
                return await self._finish_regeneration(
                    generation_id, stop_reason=stop_reason
                )
            return await self._finish_generation(
                generation_id,
                status=GenerationStatus.COMPLETED,
                stop_reason=stop_reason,
                error_code=None,
                error_message=None,
            )

        commit_task = asyncio.create_task(
            commit_completion(),
            name=f"fcc-chat-generation-commit-{generation_id}",
        )
        session, _cancellation = await _await_task_despite_cancellation(commit_task)
        self._emit_terminal_nowait(
            active,
            "turn.completed",
            {
                "generation_id": generation_id,
                "revision": session.revision,
                "actual_model": active.actual_model,
            },
        )

    async def _handle_sse_event(
        self,
        active: _ActiveOperation,
        *,
        event_name: str,
        data: dict[str, object],
        block_segments: dict[int, int],
    ) -> tuple[int, bool, str | None]:
        payload_type = _optional_string(data.get("type"))
        if event_name == "error" or payload_type == "error":
            error = data.get("error")
            message = "The provider returned an error."
            if isinstance(error, dict):
                candidate = error.get("message")
                if isinstance(candidate, str) and candidate.strip():
                    message = candidate
            raise ExecutionFailure(
                kind=FailureKind.UPSTREAM,
                status_code=502,
                message=message,
                retryable=False,
            )
        if payload_type == "content_block_start":
            index = _required_index(data)
            block = data.get("content_block")
            if not isinstance(block, dict):
                raise ChatValidationError(
                    "The provider emitted a malformed content block."
                )
            block_type = _optional_string(block.get("type"))
            if block_type == "redacted_thinking":
                return 0, False, None
            if block_type not in {"text", "thinking"}:
                raise ChatValidationError(
                    "The provider emitted an unsupported tool or content block."
                )
            kind = (
                SegmentKind.THINKING if block_type == "thinking" else SegmentKind.TEXT
            )
            ordinal = len(active.segments)
            active.segments.append(ChatSegment(ordinal=ordinal, kind=kind, text=""))
            block_segments[index] = ordinal
            await self._emit(
                active,
                "segment.started",
                {"ordinal": ordinal, "kind": kind.value},
            )
            eager_key = "thinking" if kind is SegmentKind.THINKING else "text"
            eager = _optional_string(block.get(eager_key)) or ""
            if eager:
                await self._append_delta(active, ordinal, eager)
            return len(eager), False, None
        if payload_type == "content_block_delta":
            index = _required_index(data)
            delta = data.get("delta")
            if not isinstance(delta, dict):
                raise ChatValidationError(
                    "The provider emitted a malformed content delta."
                )
            ordinal = block_segments.get(index)
            if ordinal is None:
                raise ChatValidationError(
                    "The provider emitted a delta for no open block."
                )
            delta_type = _optional_string(delta.get("type"))
            if delta_type == "signature_delta":
                return 0, False, None
            key = "thinking" if delta_type == "thinking_delta" else "text"
            if delta_type not in {"thinking_delta", "text_delta"}:
                raise ChatValidationError(
                    "The provider emitted an unsupported content delta."
                )
            text = _optional_string(delta.get(key)) or ""
            if text:
                await self._append_delta(active, ordinal, text)
            return len(text), False, None
        if payload_type == "content_block_stop":
            index = _required_index(data)
            ordinal = block_segments.pop(index, None)
            if ordinal is not None:
                await self._flush_segments(active)
                await self._emit(active, "segment.completed", {"ordinal": ordinal})
            return 0, False, None
        if payload_type == "message_delta":
            delta = data.get("delta")
            reason = None
            if isinstance(delta, dict):
                reason = _optional_string(delta.get("stop_reason"))
            return 0, False, reason
        if payload_type == "message_stop":
            return 0, True, None
        return 0, False, None

    async def _append_delta(
        self, active: _ActiveOperation, ordinal: int, delta: str
    ) -> None:
        current = active.segments[ordinal]
        active.segments[ordinal] = replace(current, text=f"{current.text}{delta}")
        await self._emit(
            active,
            "segment.delta",
            {"ordinal": ordinal, "kind": current.kind.value, "delta": delta},
        )

    async def _flush_segments(self, active: _ActiveOperation) -> None:
        if active.generation_id is None:
            return
        await self._store.replace_generation_segments(
            active.generation_id,
            tuple(active.segments),
        )

    async def _compact_transcript(
        self,
        active: _ActiveOperation,
        *,
        builder: ChatContextBuilder,
        lease: RequestRuntimeLease,
        transcript: ChatTranscript,
        prompt: str,
        pending_draft: str | None,
        excluded_generation_id: str | None,
    ) -> ChatCompaction:
        covered = (
            transcript.compaction.covered_through_sequence
            if transcript.compaction is not None
            else 0
        )
        remaining = tuple(turn for turn in transcript.turns if turn.sequence > covered)
        if len(remaining) <= 1:
            raise ChatValidationError(
                "There is not enough older conversation to compact."
            )
        await self._emit(active, "compaction.started", {})
        option = builder.model(transcript.session.model)
        output_tokens = builder.summary_output_tokens(option)
        retain_count = min(4, len(remaining) - 1)
        initial_count = len(remaining) - retain_count
        next_turns = remaining[:initial_count]
        summary = transcript.compaction.summary if transcript.compaction else None
        summary, actual_model = await self._summarize_turns(
            active,
            builder=builder,
            lease=lease,
            model_ref=transcript.session.model,
            existing_summary=summary,
            turns=next_turns,
            output_tokens=output_tokens,
        )
        covered = next_turns[-1].sequence

        while True:
            candidate = ChatCompaction(
                session_id=transcript.session.id,
                covered_through_sequence=covered,
                summary=summary,
                estimated_tokens=estimate_text_tokens(summary),
                requested_model=transcript.session.model,
                actual_model=actual_model,
                updated_at=0,
            )
            virtual = replace(transcript, compaction=candidate)
            estimate = builder.prepare(
                virtual,
                system_prompt=prompt,
                draft=pending_draft,
                exclude_generation_id=excluded_generation_id,
            ).estimate
            target = compaction_target_tokens(estimate)
            if target is None or estimate.estimated_input_tokens <= target:
                break
            later = tuple(turn for turn in remaining if turn.sequence > covered)
            if len(later) <= 1:
                raise ChatValidationError(
                    "The newest exchange cannot fit this model. Shorten the system "
                    "prompt, lower thinking, or choose a larger-context model."
                )
            summary, actual_model = await self._summarize_turns(
                active,
                builder=builder,
                lease=lease,
                model_ref=transcript.session.model,
                existing_summary=summary,
                turns=(later[0],),
                output_tokens=output_tokens,
            )
            covered = later[0].sequence

        async def commit_compaction() -> ChatCompaction:
            return await self._retry_terminal_persistence(
                lambda: self._store.upsert_compaction(
                    transcript.session.id,
                    covered_through_sequence=covered,
                    summary=summary,
                    estimated_tokens=estimate_text_tokens(summary),
                    requested_model=transcript.session.model,
                    actual_model=actual_model,
                ),
                label=(
                    f"compaction session_id={transcript.session.id} "
                    f"covered_through_sequence={covered}"
                ),
            )

        commit_task = asyncio.create_task(
            commit_compaction(),
            name=f"fcc-chat-compaction-commit-{transcript.session.id}",
        )
        compaction, cancellation = await _await_task_despite_cancellation(commit_task)
        completion: JsonObject = {
            "covered_through_sequence": covered,
            "revision": transcript.session.revision + 1,
        }
        if active.kind is _OperationKind.COMPACT:
            self._emit_terminal_nowait(active, "compaction.completed", completion)
        else:
            self._emit_nowait(active, "compaction.completed", completion)
            if cancellation is not None:
                raise cancellation
        return compaction

    async def _summarize_turns(
        self,
        active: _ActiveOperation,
        *,
        builder: ChatContextBuilder,
        lease: RequestRuntimeLease,
        model_ref: str,
        existing_summary: str | None,
        turns: tuple[ChatTurn, ...],
        output_tokens: int,
    ) -> tuple[str, str]:
        summary = existing_summary
        pending = list(turns)
        actual_model = model_ref
        while pending:
            take = len(pending)
            if builder.model(model_ref).context_window_tokens is not None:
                take = 0
                for candidate_count in range(1, len(pending) + 1):
                    source = builder.compaction_source(
                        summary,
                        tuple(pending[:candidate_count]),
                    )
                    if not builder.summary_source_fits(
                        model_ref=model_ref,
                        source=source,
                        output_tokens=output_tokens,
                    ):
                        break
                    take = candidate_count
                if take == 0:
                    take = 1
            source = builder.compaction_source(summary, tuple(pending[:take]))
            summary, actual_model = await self._execute_summary(
                active,
                builder=builder,
                lease=lease,
                model_ref=model_ref,
                source=source,
                output_tokens=output_tokens,
            )
            del pending[:take]
        if summary is None:
            raise ChatValidationError("Compaction produced no summary.")
        return summary, actual_model

    async def _execute_summary(
        self,
        active: _ActiveOperation,
        *,
        builder: ChatContextBuilder,
        lease: RequestRuntimeLease,
        model_ref: str,
        source: str,
        output_tokens: int,
    ) -> tuple[str, str]:
        routed = builder.prepare_summary(
            model_ref=model_ref,
            source=source,
            output_tokens=output_tokens,
        )
        executor = ProviderExecutor(
            lease.resolve_provider,
            progress_timeout_seconds=lease.settings.provider_progress_timeout,
            generation_id=lease.generation_id,
            log_raw_payloads=lease.settings.log_raw_api_payloads,
        )
        selected_model: str | None = None

        def selected(target_model: ProviderModelTarget) -> None:
            nonlocal selected_model
            selected_model = target_model.provider_model_ref

        stream = executor.stream_messages(
            routed,
            raw_log_payload=routed.request.model_dump(mode="json"),
            request_id=str(uuid.uuid4()),
            candidate_selected=selected,
        )
        try:
            body, error, complete = await aggregate_anthropic_sse_to_message(stream)
        finally:
            await close_stream_input(
                stream,
                owner="chat_compaction",
                source="application",
                preserved_error=sys.exception(),
            )
        if error is not None:
            message = error.get("message")
            raise ExecutionFailure(
                kind=FailureKind.UPSTREAM,
                status_code=502,
                message=message if isinstance(message, str) else "Compaction failed.",
                retryable=False,
            )
        if not complete:
            raise ChatValidationError("The provider stream ended before completing.")
        response = MessagesResponse.model_validate(body)
        summary = "\n".join(
            block.text
            for block in response.content
            if isinstance(block, ContentBlockText)
        ).strip()
        if not summary:
            raise ChatValidationError("The model returned an empty compaction summary.")
        if selected_model is None:
            raise ChatValidationError(
                "The provider did not identify the compaction model."
            )
        await self._emit(
            active,
            "compaction.progress",
            {"covered_exchange_count": 1},
        )
        return summary, selected_model

    async def _handle_cancelled(self, active: _ActiveOperation) -> None:
        reason = active.cancellation_reason or _CancellationReason.STOPPED
        if active.generation_id is not None:
            try:
                if active.regeneration:
                    await self._discard_regeneration(active.generation_id)
                else:
                    await self._flush_terminal_segments(active)
                    status = (
                        GenerationStatus.INTERRUPTED
                        if reason is _CancellationReason.INTERRUPTED
                        else GenerationStatus.STOPPED
                    )
                    await self._finish_generation(
                        active.generation_id,
                        status=status,
                        stop_reason=reason.value,
                        error_code=None,
                        error_message=None,
                    )
            except Exception as exc:
                if reason is _CancellationReason.INTERRUPTED:
                    logger.warning(
                        "Chat shutdown could not persist terminal state: "
                        "session_id={} operation_id={} exc_type={}",
                        active.session_id,
                        active.operation_id,
                        type(exc).__name__,
                    )
                    return
                self._disable_after_terminal_write_failure(active, exc)
                if reason is _CancellationReason.DELETED:
                    raise
                self._emit_terminal_nowait(
                    active,
                    "turn.failed",
                    {
                        "code": "chat_storage_unavailable",
                        "message": _STORAGE_RESTART_MESSAGE,
                    },
                )
                return
        if reason is _CancellationReason.DELETED:
            return
        event = (
            "compaction.stopped"
            if active.kind is _OperationKind.COMPACT
            else "turn.stopped"
        )
        self._emit_terminal_nowait(active, event, {"reason": reason.value})

    async def _handle_failure(self, active: _ActiveOperation, exc: Exception) -> None:
        if isinstance(exc, _TerminalPersistenceError):
            self._disable_after_terminal_write_failure(active, exc)
            event = (
                "compaction.failed"
                if active.kind is _OperationKind.COMPACT
                else "turn.failed"
            )
            self._emit_terminal_nowait(
                active,
                event,
                {
                    "code": "chat_storage_unavailable",
                    "message": _STORAGE_RESTART_MESSAGE,
                },
            )
            return
        code = "chat_error"
        message = "Chat operation failed."
        if isinstance(exc, ExecutionFailure):
            code = exc.kind.value
            message = exc.message
        elif isinstance(
            exc,
            (ChatValidationError, ChatConflictError, ChatUnavailableError),
        ):
            code = type(exc).__name__
            message = str(exc)
        else:
            logger.warning(
                "Chat operation failed: session_id={} operation_id={} exc_type={}",
                active.session_id,
                active.operation_id,
                type(exc).__name__,
            )
        if active.generation_id is not None:
            try:
                if active.regeneration:
                    await self._discard_regeneration(active.generation_id)
                else:
                    await self._flush_terminal_segments(active)
                    await self._finish_generation(
                        active.generation_id,
                        status=GenerationStatus.FAILED,
                        stop_reason=None,
                        error_code=code,
                        error_message=message,
                    )
            except Exception as persistence_exc:
                self._disable_after_terminal_write_failure(
                    active,
                    persistence_exc,
                )
                code = "chat_storage_unavailable"
                message = _STORAGE_RESTART_MESSAGE
        event = (
            "compaction.failed"
            if active.kind is _OperationKind.COMPACT
            else "turn.failed"
        )
        self._emit_terminal_nowait(active, event, {"code": code, "message": message})

    async def _flush_terminal_segments(self, active: _ActiveOperation) -> None:
        await self._retry_terminal_persistence(
            lambda: self._flush_segments(active),
            label=(
                f"segment flush session_id={active.session_id} "
                f"operation_id={active.operation_id}"
            ),
        )

    async def _finish_generation(
        self,
        generation_id: str,
        *,
        status: GenerationStatus,
        stop_reason: str | None,
        error_code: str | None,
        error_message: str | None,
    ) -> ChatSession:
        return await self._retry_terminal_persistence(
            lambda: self._store.finish_generation(
                generation_id,
                status=status,
                stop_reason=stop_reason,
                error_code=error_code,
                error_message=error_message,
            ),
            label=f"generation_id={generation_id} status={status.value}",
        )

    async def _finish_regeneration(
        self,
        generation_id: str,
        *,
        stop_reason: str | None,
    ) -> ChatSession:
        return await self._retry_terminal_persistence(
            lambda: self._store.finish_regeneration(
                generation_id,
                stop_reason=stop_reason,
            ),
            label=f"regeneration_id={generation_id} status=completed",
        )

    async def _discard_regeneration(self, generation_id: str) -> None:
        await self._retry_terminal_persistence(
            lambda: self._store.discard_generation(generation_id),
            label=f"regeneration_id={generation_id} status=discarded",
        )

    @staticmethod
    async def _retry_terminal_persistence[T](
        operation: Callable[[], Awaitable[T]],
        *,
        label: str,
    ) -> T:
        try:
            return await operation()
        except ChatUnavailableError:
            logger.warning(
                "Retrying transient Chat terminal persistence failure: {}",
                label,
            )
        try:
            return await operation()
        except ChatUnavailableError as exc:
            raise _TerminalPersistenceError(_STORAGE_RESTART_MESSAGE) from exc

    def _disable_after_terminal_write_failure(
        self,
        active: _ActiveOperation,
        exc: Exception,
    ) -> None:
        self._accepting = False
        self._unavailable_message = _STORAGE_RESTART_MESSAGE
        logger.error(
            "Chat Sessions disabled after terminal persistence failure: "
            "session_id={} operation_id={} exc_type={}",
            active.session_id,
            active.operation_id,
            type(exc).__name__,
        )

    async def _emit(
        self, active: _ActiveOperation, event: str, data: JsonObject
    ) -> None:
        active.event_sequence += 1
        payload: JsonObject = {
            "session_id": active.session_id,
            "operation_id": active.operation_id,
            **data,
        }
        await active.queue.put(
            ChatStreamEvent(
                event=event,
                sequence=active.event_sequence,
                data=payload,
            )
        )

    def _emit_terminal_nowait(
        self, active: _ActiveOperation, event: str, data: JsonObject
    ) -> None:
        if active.terminal_emitted:
            return
        self._emit_nowait(active, event, data)
        active.terminal_emitted = True

    def _emit_nowait(
        self, active: _ActiveOperation, event: str, data: JsonObject
    ) -> None:
        active.event_sequence += 1
        payload: JsonObject = {
            "session_id": active.session_id,
            "operation_id": active.operation_id,
            **data,
        }
        _make_queue_room(active.queue)
        active.queue.put_nowait(
            ChatStreamEvent(
                event=event,
                sequence=active.event_sequence,
                data=payload,
            )
        )

    async def _cancel_active(
        self,
        active: _ActiveOperation,
        *,
        reason: _CancellationReason,
    ) -> None:
        task = active.task
        if task is None or task.done():
            return
        if active.cancellation_reason is None:
            active.cancellation_reason = reason
            task.cancel()
        cancellation: asyncio.CancelledError | None = None
        try:
            _result, cancellation = await _await_task_despite_cancellation(task)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
        finally:
            await self._release_active(active)
        if cancellation is not None:
            raise cancellation

    async def _release_active(self, active: _ActiveOperation) -> None:
        async with self._active_lock:
            if self._active.get(active.session_id) is not active:
                return
            self._active.pop(active.session_id, None)
        _make_queue_room(active.queue)
        active.queue.put_nowait(None)

    def _require_available(self) -> None:
        if not self._started or not self._accepting:
            raise ChatUnavailableError(
                self._unavailable_message or "Chat Sessions is unavailable."
            )


async def _commit_generation_start(
    active: _ActiveOperation,
    *,
    store: ChatStorePort,
    generation_id: str,
    operation: Awaitable[object],
    regeneration: bool = False,
) -> None:
    """Publish durable generation ownership before restoring cancellation."""

    async def run() -> None:
        await operation

    task = asyncio.create_task(
        run(),
        name=f"fcc-chat-generation-start-{generation_id}",
    )
    cancellation = await _wait_task_despite_cancellation(task)
    try:
        task.result()
    except ChatUnavailableError:

        async def reconcile() -> bool:
            return await store.generation_start_committed(
                active.session_id,
                generation_id=generation_id,
                staged=regeneration,
            )

        reconciliation_task = asyncio.create_task(
            reconcile(),
            name=f"fcc-chat-generation-reconcile-{generation_id}",
        )
        try:
            (
                committed,
                reconciliation_cancellation,
            ) = await _await_task_despite_cancellation(reconciliation_task)
        except ChatUnavailableError as exc:
            raise _TerminalPersistenceError(_STORAGE_RESTART_MESSAGE) from exc
        cancellation = cancellation or reconciliation_cancellation
        if not committed:
            raise
    active.generation_id = generation_id
    active.regeneration = regeneration
    if cancellation is not None:
        raise cancellation


async def _wait_task_despite_cancellation[T](
    task: asyncio.Task[T],
) -> asyncio.CancelledError | None:
    """Wait for a child outcome while retaining cancellation for its caller."""

    current = asyncio.current_task()
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if current is not None and current.cancelling():
                cancellation = cancellation or exc
        except Exception:
            break
    return cancellation


async def _await_task_despite_cancellation[T](
    task: asyncio.Task[T],
) -> tuple[T, asyncio.CancelledError | None]:
    """Wait for an already-started task before restoring caller cancellation."""

    cancellation = await _wait_task_despite_cancellation(task)
    result = task.result()
    return result, cancellation


def _latest_turn(transcript: ChatTranscript) -> ChatTurn:
    if not transcript.turns:
        raise ChatConflictError("This chat has no answer to operate on.")
    return transcript.turns[-1]


def _expect_revision(session: ChatSession, expected: int) -> None:
    if expected <= 0 or session.revision != expected:
        raise ChatConflictError("This chat changed in another tab. Refresh it.")


def _require_request_fits(estimate: ChatContextEstimate) -> None:
    if (
        estimate.usable_input_tokens is not None
        and estimate.estimated_input_tokens > estimate.usable_input_tokens
    ):
        raise ChatValidationError(
            "This turn cannot fit the selected model. Shorten the message or system "
            "prompt, lower thinking, or choose a larger-context model."
        )


def _canonical_uuid(value: str, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ChatValidationError(f"Invalid {label} ID.") from exc
    if parsed.version != 4 or str(parsed) != value.lower():
        raise ChatValidationError(f"Invalid {label} ID.")
    return str(parsed)


def _required_index(data: dict[str, object]) -> int:
    value = data.get("index")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ChatValidationError("The provider emitted an invalid content index.")
    return value


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _make_queue_room(queue: asyncio.Queue[ChatStreamEvent | None]) -> None:
    if queue.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
