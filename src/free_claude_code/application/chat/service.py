"""Application lifecycle and commands for durable local Chat Sessions."""

import asyncio
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum

from loguru import logger

from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.ports import RequestRuntimeLease, RequestRuntimePort
from free_claude_code.application.routing import ProviderModelTarget
from free_claude_code.config.model_refs import (
    RETIRED_PROVIDER_IDS,
    is_retired_model_ref,
)
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
from .events import ChatEventPublisher
from .models import (
    DEFAULT_CHAT_SYSTEM_PROMPT,
    ChatActiveOperation,
    ChatCompaction,
    ChatConflictError,
    ChatContextEstimate,
    ChatModelOption,
    ChatOperationAcknowledgement,
    ChatOperationKind,
    ChatOperationPhase,
    ChatPreferences,
    ChatReasoning,
    ChatSegment,
    ChatSession,
    ChatSessionDetail,
    ChatSessionPage,
    ChatSessionSummary,
    ChatTranscript,
    ChatTurn,
    ChatUnavailableError,
    ChatValidationError,
    GenerationStatus,
    SegmentKind,
)
from .ports import ChatEventSubscriptionPort, ChatStorePort

_PERSIST_INTERVAL_SECONDS = 0.25
_PERSIST_CHARACTER_THRESHOLD = 4_096
_TURN_PAGE_LIMIT = 50
_SESSION_PAGE_LIMIT = 25
_STORAGE_RESTART_MESSAGE = (
    "Chat storage became unavailable. Restart FCC to repair Chat Sessions."
)


class _CancellationReason(StrEnum):
    STOPPED = "stopped"
    INTERRUPTED = "interrupted"
    DELETED = "deleted"


class _OperationLifecycle(StrEnum):
    RUNNING = "running"
    SETTLING = "settling"


@dataclass(frozen=True, slots=True)
class _TerminalEvent:
    event: str
    data: JsonObject
    publish_summary: bool


class _TerminalPersistenceError(ChatUnavailableError):
    """A bounded terminal write failed and requires startup repair."""


@dataclass(slots=True)
class _ActiveOperation:
    session_id: str
    operation_id: str
    kind: ChatOperationKind
    phase: ChatOperationPhase
    submitted_text: str | None = None
    work_task: asyncio.Task[None] | None = None
    supervisor_task: asyncio.Task[None] | None = None
    lifecycle: _OperationLifecycle = _OperationLifecycle.RUNNING
    cancellation_reason: _CancellationReason | None = None
    turn_id: str | None = None
    generation_id: str | None = None
    regeneration: bool = False
    segments: list[ChatSegment] = field(default_factory=list)
    event_sequence: int = 0
    actual_model: str | None = None
    terminal_event: _TerminalEvent | None = None


class ChatService:
    """Single owner of Chat session commands and active provider work."""

    def __init__(self, runtime: RequestRuntimePort, store: ChatStorePort) -> None:
        self._runtime = runtime
        self._store = store
        self._context = ChatContextBuilder(runtime)
        self._events = ChatEventPublisher()
        self._active: dict[str, _ActiveOperation] = {}
        self._deleting: set[str] = set()
        self._active_lock = asyncio.Lock()
        self._started = False
        self._accepting = False
        self._unavailable_message: str | None = "Chat Sessions is starting."

    async def start(self) -> None:
        if self._started:
            if self._accepting:
                return
            # A prior startup cleanup failed. Release its store before retrying.
            retry_cleanup = asyncio.create_task(self._close_startup_store())
            cancellation = await _wait_task_despite_cancellation(retry_cleanup)
            try:
                retry_cleanup.result()
            except Exception as exc:
                if cancellation is not None:
                    raise cancellation from exc
                raise
            if cancellation is not None:
                raise cancellation
        cancellation: asyncio.CancelledError | None = None
        try:
            await self._store.start()
            self._started = True
            default_model = self._runtime.current_settings().model
            option = self._context.model(default_model)
            repair_task = asyncio.create_task(
                self._store.repair_retired_model_selections(
                    retired_provider_ids=RETIRED_PROVIDER_IDS,
                    default_model=default_model,
                    disable_reasoning=option.supports_reasoning is False,
                ),
                name="fcc-chat-retirement-repair",
            )
            cancellation = await _wait_task_despite_cancellation(repair_task)
            repaired = repair_task.result()
            if cancellation is not None:
                raise cancellation
            if repaired:
                logger.info(
                    "Repaired retired model selections in {} Chat sessions", repaired
                )
        except BaseException as exc:
            if self._started:
                try:
                    cleanup_task = asyncio.create_task(
                        self._close_startup_store(), name="fcc-chat-start-cleanup"
                    )
                    cleanup_cancellation = await _wait_task_despite_cancellation(
                        cleanup_task
                    )
                    cancellation = cancellation or cleanup_cancellation
                    cleanup_task.result()
                except Exception as cleanup_exc:
                    logger.warning(
                        "Chat startup cleanup failed: exc_type={}",
                        type(cleanup_exc).__name__,
                    )
            if not isinstance(exc, Exception):
                raise
            if cancellation is not None:
                raise cancellation from exc
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

    async def _close_startup_store(self) -> None:
        await self._store.close()
        self._started = False

    def begin_shutdown(self) -> None:
        """Stop new Chat work and finish feeds before HTTP connections drain."""
        self._accepting = False
        self._unavailable_message = "Chat Sessions is stopping."
        self._events.disconnect_subscribers()

    async def close(self) -> None:
        self._accepting = False
        async with self._active_lock:
            active = tuple(self._active.values())
        await asyncio.gather(
            *(
                self._cancel_active(
                    operation,
                    reason=_CancellationReason.INTERRUPTED,
                    wait_if_settling=True,
                )
                for operation in active
            )
        )
        self._events.close()
        await self._store.close()
        self._started = False
        self._unavailable_message = "Chat Sessions is stopped."

    def availability(self) -> tuple[bool, str | None]:
        return self._started and self._accepting, self._unavailable_message

    def models(self) -> tuple[ChatModelOption, ...]:
        return self._context.models()

    async def subscribe(
        self,
    ) -> tuple[ChatEventSubscriptionPort, tuple[ChatActiveOperation, ...]]:
        self._require_available()
        subscription = self._events.subscribe()
        try:
            async with self._active_lock:
                active = tuple(_active_snapshot(item) for item in self._active.values())
        except BaseException:
            await subscription.aclose()
            raise
        return subscription, active

    async def preferences(self) -> ChatPreferences:
        self._require_available()
        return await self._store.load_preferences()

    async def save_system_prompt(self, value: str) -> ChatPreferences:
        self._require_available()
        preferences = await self._store.save_system_prompt(value)
        self._publish_preferences(preferences)
        return preferences

    async def reset_system_prompt(self) -> ChatPreferences:
        self._require_available()
        preferences = await self._store.save_system_prompt(DEFAULT_CHAT_SYSTEM_PROMPT)
        self._publish_preferences(preferences)
        return preferences

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
        session = await self._store.create_session(
            session_id=str(uuid.uuid4()),
            model=model,
            reasoning=reasoning,
        )
        await self._publish_session_summary(session.id, event="session.created")
        return session

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
            retired = is_retired_model_ref(model)
            if retired:
                model = self._runtime.current_settings().model
            option = self._context.model(model)
            if retired and option.supports_reasoning is False:
                reasoning = ChatReasoning.OFF
        async with self._active_lock:
            if session_id in self._deleting:
                raise ChatConflictError("This chat is being deleted.")
            if (
                model is not None or reasoning is not None
            ) and session_id in self._active:
                raise ChatConflictError(
                    "Model and thinking cannot change while this chat is running."
                )
            session = await self._store.update_session(
                session_id,
                expected_revision=expected_revision,
                title=title,
                model=model,
                reasoning=reasoning,
            )
        await self._publish_session_summary(session.id)
        return session

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

            async def finish_deletion() -> None:
                settled = current
                if active is not None:
                    await self._cancel_active(
                        active,
                        reason=_CancellationReason.DELETED,
                        wait_if_settling=True,
                    )
                    settled = await self._store.get_session(session_id)
                await self._store.delete_session(
                    session_id,
                    expected_revision=settled.revision,
                )
                self._events.publish("session.deleted", {"session_id": session_id})

            deletion_task = asyncio.create_task(
                finish_deletion(),
                name=f"fcc-chat-delete-{session_id}",
            )
            _result, cancellation = await _await_task_despite_cancellation(
                deletion_task
            )
            if cancellation is not None:
                raise cancellation
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
    ) -> ChatOperationAcknowledgement:
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
            kind=ChatOperationKind.SEND,
            submitted_text=text,
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
    ) -> ChatOperationAcknowledgement:
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
            kind=ChatOperationKind.RETRY,
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
    ) -> ChatOperationAcknowledgement:
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
            kind=ChatOperationKind.REGENERATE,
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
    ) -> ChatOperationAcknowledgement:
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
            kind=ChatOperationKind.COMPACT,
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
        return await self._cancel_active(
            active,
            reason=_CancellationReason.STOPPED,
            wait_if_settling=False,
        )

    async def _start_operation(
        self,
        session_id: str,
        *,
        operation_id: str,
        kind: ChatOperationKind,
        submitted_text: str | None = None,
        action: Callable[[_ActiveOperation], Awaitable[None]],
    ) -> ChatOperationAcknowledgement:
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
                phase=(
                    ChatOperationPhase.COMPACTING
                    if kind is ChatOperationKind.COMPACT
                    else ChatOperationPhase.GENERATING
                ),
                submitted_text=submitted_text,
            )

            async def run_work() -> None:
                await action(active)

            self._active[session_id] = active
            active.work_task = asyncio.create_task(
                run_work(),
                name=f"fcc-chat-{kind.value}-{operation_id}-work",
            )
            active.supervisor_task = asyncio.create_task(
                self._run_active(active),
                name=f"fcc-chat-{kind.value}-{operation_id}",
            )
            self._emit(
                active,
                "operation.started",
                {
                    "kind": kind.value,
                    "phase": active.phase.value,
                    "submitted_text": submitted_text,
                },
            )
        return ChatOperationAcknowledgement(
            session_id=session_id,
            operation_id=operation_id,
            kind=kind,
        )

    async def _run_active(
        self,
        active: _ActiveOperation,
    ) -> None:
        work_task = active.work_task
        if work_task is None:
            raise RuntimeError("Chat operation work was not initialized.")
        try:
            await work_task
        except asyncio.CancelledError:
            await self._claim_settlement(
                active,
                reason=active.cancellation_reason or _CancellationReason.STOPPED,
            )
            await self._handle_cancelled(active)
        except Exception as exc:
            await self._claim_settlement(active)
            if active.cancellation_reason is not None:
                await self._handle_cancelled(active)
            else:
                await self._handle_failure(active, exc)
        finally:
            publish_summary = await self._release_active(active)
            if publish_summary:
                await self._publish_session_summary(active.session_id)

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
            active.turn_id = turn_id
            await self._publish_session_summary(active.session_id)
            self._emit(
                active,
                "turn.started",
                {
                    "turn_id": turn_id,
                    "generation_id": generation_id,
                    "revision": transcript.session.revision + 1,
                },
            )
            await self._execute_generation_with_context_recovery(
                active,
                prepared=prepared,
                builder=builder,
                lease=lease,
                prompt=prompt,
                excluded_generation_id=generation_id,
            )
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
            active.turn_id = latest.id
            await self._publish_session_summary(active.session_id)
            self._emit(
                active,
                "turn.started",
                {
                    "turn_id": latest.id,
                    "generation_id": generation_id,
                    "revision": transcript.session.revision + 1,
                },
            )
            await self._execute_generation_with_context_recovery(
                active,
                prepared=prepared,
                builder=builder,
                lease=lease,
                prompt=prompt,
                excluded_generation_id=generation_id,
            )
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
            active.turn_id = latest.id
            await self._publish_session_summary(active.session_id)
            self._emit(
                active,
                "turn.started",
                {
                    "turn_id": latest.id,
                    "generation_id": generation_id,
                    "revision": transcript.session.revision + 1,
                    "regeneration": True,
                },
            )
            await self._execute_generation_with_context_recovery(
                active,
                prepared=prepared,
                builder=builder,
                lease=lease,
                prompt=prompt,
                excluded_generation_id=latest.generation.id,
            )
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
        candidate_failures: list[FailureKind] | None = None,
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

        def failed(failure: ExecutionFailure) -> None:
            if candidate_failures is not None:
                candidate_failures.append(failure.kind)

        stream = executor.stream_messages(
            prepared.routed,
            raw_log_payload=prepared.routed.request.model_dump(mode="json"),
            request_id=active.operation_id,
            candidate_selected=selected,
            candidate_failed=failed if candidate_failures is not None else None,
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

        if not await self._claim_settlement(active):
            raise asyncio.CancelledError

        async def commit_completion() -> ChatSession:
            await self._flush_terminal_segments(active)
            if active.regeneration:
                return await self._finish_regeneration(
                    generation_id,
                    status=GenerationStatus.COMPLETED,
                    stop_reason=stop_reason,
                    error_code=None,
                    error_message=None,
                )
            return await self._finish_generation(
                generation_id,
                status=GenerationStatus.COMPLETED,
                stop_reason=stop_reason,
                error_code=None,
                error_message=None,
            )

        session = await commit_completion()
        self._stage_terminal(
            active,
            "turn.completed",
            {
                "generation_id": generation_id,
                "revision": session.revision,
                "actual_model": active.actual_model,
            },
            publish_summary=True,
        )

    async def _execute_generation_with_context_recovery(
        self,
        active: _ActiveOperation,
        *,
        prepared: PreparedChatRequest,
        builder: ChatContextBuilder,
        lease: RequestRuntimeLease,
        prompt: str,
        excluded_generation_id: str,
    ) -> None:
        candidate_failures: list[FailureKind] = []
        try:
            await self._execute_generation(
                active,
                prepared=prepared,
                lease=lease,
                candidate_failures=candidate_failures,
            )
            return
        except ExecutionFailure as exc:
            if (
                exc.kind is not FailureKind.CONTEXT_WINDOW_EXCEEDED
                or active.segments
                or not candidate_failures
                or any(
                    kind is not FailureKind.CONTEXT_WINDOW_EXCEEDED
                    for kind in candidate_failures
                )
            ):
                raise
            context_failure = exc

        transcript = await self._store.get_transcript(active.session_id)
        if not builder.can_compact(transcript):
            raise context_failure
        await self._compact_transcript(
            active,
            builder=builder,
            lease=lease,
            transcript=transcript,
            prompt=prompt,
            pending_draft=None,
            excluded_generation_id=excluded_generation_id,
        )
        transcript = await self._store.get_transcript(active.session_id)
        prepared = builder.prepare(
            transcript,
            system_prompt=prompt,
            exclude_generation_id=excluded_generation_id,
        )
        _require_request_fits(prepared.estimate)
        await self._execute_generation(active, prepared=prepared, lease=lease)

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
            self._emit(
                active,
                "segment.started",
                {"ordinal": ordinal, "segment_kind": kind.value},
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
                self._emit(active, "segment.completed", {"ordinal": ordinal})
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
        self._emit(
            active,
            "segment.delta",
            {
                "ordinal": ordinal,
                "segment_kind": current.kind.value,
                "delta": delta,
            },
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
        active.phase = ChatOperationPhase.COMPACTING
        self._emit(active, "compaction.started", {"phase": active.phase.value})
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

        cancellation: asyncio.CancelledError | None = None
        if active.kind is ChatOperationKind.COMPACT:
            if not await self._claim_settlement(active):
                raise asyncio.CancelledError
            compaction = await commit_compaction()
        else:
            commit_task = asyncio.create_task(
                commit_compaction(),
                name=f"fcc-chat-compaction-commit-{transcript.session.id}",
            )
            compaction, cancellation = await _await_task_despite_cancellation(
                commit_task
            )
        completion: JsonObject = {
            "covered_through_sequence": covered,
            "revision": transcript.session.revision + 1,
        }
        if active.kind is ChatOperationKind.COMPACT:
            self._stage_terminal(
                active,
                "compaction.completed",
                completion,
                publish_summary=True,
            )
        else:
            active.phase = ChatOperationPhase.GENERATING
            completion["phase"] = active.phase.value
            self._emit(active, "compaction.completed", completion)
            if cancellation is not None:
                raise cancellation
            await self._publish_session_summary(active.session_id)
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
            while True:
                source = builder.compaction_source(summary, tuple(pending[:take]))
                try:
                    summary, actual_model = await self._execute_summary(
                        active,
                        builder=builder,
                        lease=lease,
                        model_ref=model_ref,
                        source=source,
                        output_tokens=output_tokens,
                    )
                    break
                except ExecutionFailure as exc:
                    if exc.kind is not FailureKind.CONTEXT_WINDOW_EXCEEDED:
                        raise
                    if take == 1:
                        raise ChatValidationError(
                            "The newest exchange cannot fit this model. Shorten the "
                            "system prompt, lower thinking, or choose a larger-context "
                            "model."
                        ) from exc
                    take = max(1, take // 2)
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
        self._emit(
            active,
            "compaction.progress",
            {"covered_exchange_count": 1},
        )
        return summary, selected_model

    async def _handle_cancelled(self, active: _ActiveOperation) -> None:
        reason = active.cancellation_reason or _CancellationReason.STOPPED
        terminal_data: JsonObject = {"reason": reason.value}
        publish_summary = False
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
                    session = await self._finish_generation(
                        active.generation_id,
                        status=status,
                        stop_reason=reason.value,
                        error_code=None,
                        error_message=None,
                    )
                    terminal_data["revision"] = session.revision
                    publish_summary = True
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
                self._stage_terminal(
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
            if active.kind is ChatOperationKind.COMPACT
            else "turn.stopped"
        )
        self._stage_terminal(
            active,
            event,
            terminal_data,
            publish_summary=publish_summary,
        )

    async def _handle_failure(self, active: _ActiveOperation, exc: Exception) -> None:
        if isinstance(exc, _TerminalPersistenceError):
            self._disable_after_terminal_write_failure(active, exc)
            event = (
                "compaction.failed"
                if active.kind is ChatOperationKind.COMPACT
                else (
                    "turn.failed"
                    if active.generation_id is not None
                    else "operation.failed"
                )
            )
            self._stage_terminal(
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
        terminal_data: JsonObject = {"code": code, "message": message}
        publish_summary = False
        if active.generation_id is not None:
            try:
                await self._flush_terminal_segments(active)
                if active.regeneration:
                    session = await self._finish_regeneration(
                        active.generation_id,
                        status=GenerationStatus.FAILED,
                        stop_reason=None,
                        error_code=code,
                        error_message=message,
                    )
                else:
                    session = await self._finish_generation(
                        active.generation_id,
                        status=GenerationStatus.FAILED,
                        stop_reason=None,
                        error_code=code,
                        error_message=message,
                    )
                terminal_data["revision"] = session.revision
                publish_summary = True
            except Exception as persistence_exc:
                self._disable_after_terminal_write_failure(
                    active,
                    persistence_exc,
                )
                code = "chat_storage_unavailable"
                message = _STORAGE_RESTART_MESSAGE
                terminal_data = {"code": code, "message": message}
        event = (
            "compaction.failed"
            if active.kind is ChatOperationKind.COMPACT
            else "turn.failed"
            if active.generation_id is not None
            else "operation.failed"
        )
        self._stage_terminal(
            active,
            event,
            terminal_data,
            publish_summary=publish_summary,
        )

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
        status: GenerationStatus,
        stop_reason: str | None,
        error_code: str | None,
        error_message: str | None,
    ) -> ChatSession:
        return await self._retry_terminal_persistence(
            lambda: self._store.finish_regeneration(
                generation_id,
                status=status,
                stop_reason=stop_reason,
                error_code=error_code,
                error_message=error_message,
            ),
            label=f"regeneration_id={generation_id} status={status.value}",
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

    def _emit(self, active: _ActiveOperation, event: str, data: JsonObject) -> None:
        active.event_sequence += 1
        payload: JsonObject = {
            "session_id": active.session_id,
            "operation_id": active.operation_id,
            "kind": active.kind.value,
            "operation_sequence": active.event_sequence,
            **data,
        }
        self._events.publish(event, payload)

    def _stage_terminal(
        self,
        active: _ActiveOperation,
        event: str,
        data: JsonObject,
        *,
        publish_summary: bool = False,
    ) -> None:
        if active.terminal_event is not None:
            return
        active.terminal_event = _TerminalEvent(
            event=event,
            data={**data},
            publish_summary=publish_summary,
        )

    async def _claim_settlement(
        self,
        active: _ActiveOperation,
        *,
        reason: _CancellationReason | None = None,
    ) -> bool:
        async with self._active_lock:
            if self._active.get(active.session_id) is not active:
                return False
            if active.lifecycle is _OperationLifecycle.SETTLING:
                return False
            active.lifecycle = _OperationLifecycle.SETTLING
            active.cancellation_reason = reason
            return True

    async def _cancel_active(
        self,
        active: _ActiveOperation,
        *,
        reason: _CancellationReason,
        wait_if_settling: bool,
    ) -> bool:
        async with self._active_lock:
            if self._active.get(active.session_id) is not active:
                return False
            accepted = False
            if active.lifecycle is _OperationLifecycle.RUNNING:
                active.lifecycle = _OperationLifecycle.SETTLING
                active.cancellation_reason = reason
                accepted = True
                work_task = active.work_task
                if work_task is not None and not work_task.done():
                    work_task.cancel()
            elif (
                reason is _CancellationReason.STOPPED
                and active.cancellation_reason is _CancellationReason.STOPPED
            ):
                accepted = True
            supervisor_task = active.supervisor_task
        if supervisor_task is not None and (accepted or wait_if_settling):
            cancellation = await _wait_task_despite_cancellation(supervisor_task)
            supervisor_task.result()
            if cancellation is not None:
                raise cancellation
        return accepted

    async def _release_active(self, active: _ActiveOperation) -> bool:
        publish_summary = False
        async with self._active_lock:
            if self._active.get(active.session_id) is not active:
                return False
            self._active.pop(active.session_id, None)
            terminal = active.terminal_event
            if terminal is not None:
                self._emit(active, terminal.event, terminal.data)
                publish_summary = terminal.publish_summary
        return publish_summary

    def _require_available(self) -> None:
        if not self._started or not self._accepting:
            raise ChatUnavailableError(
                self._unavailable_message or "Chat Sessions is unavailable."
            )

    def _publish_preferences(self, preferences: ChatPreferences) -> None:
        self._events.publish(
            "preferences.updated",
            {
                "system_prompt": preferences.system_prompt,
                "last_model": preferences.last_model,
                "last_reasoning": preferences.last_reasoning.value,
                "updated_at": preferences.updated_at,
            },
        )

    async def _publish_session_summary(
        self,
        session_id: str,
        *,
        event: str = "session.updated",
    ) -> None:
        try:
            summary = await self._store.get_session_summary(session_id)
            self._events.publish(event, _session_summary_event_data(summary))
        except Exception as exc:
            logger.warning(
                "Could not publish Chat session summary: session_id={} exc_type={}",
                session_id,
                type(exc).__name__,
            )
            return


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


def _active_snapshot(active: _ActiveOperation) -> ChatActiveOperation:
    return ChatActiveOperation(
        session_id=active.session_id,
        operation_id=active.operation_id,
        kind=active.kind,
        phase=active.phase,
        operation_sequence=active.event_sequence,
        submitted_text=active.submitted_text,
        turn_id=active.turn_id,
        generation_id=active.generation_id,
        regeneration=active.regeneration,
        actual_model=active.actual_model,
        segments=tuple(active.segments),
    )


def _session_summary_event_data(session: ChatSessionSummary) -> JsonObject:
    return {
        "session_id": session.id,
        "title": session.title,
        "model": session.model,
        "reasoning": session.reasoning.value,
        "revision": session.revision,
        "preview": session.preview[:240],
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }
