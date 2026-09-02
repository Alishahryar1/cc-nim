import asyncio
import sqlite3
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import closing
from pathlib import Path

import pytest

from free_claude_code.application.work import (
    CodexAppServerEvent,
    CodexAvailability,
    CodexConnectionLost,
    CodexControlCatalog,
    CodexInitialization,
    CodexNotification,
    CodexObjectPage,
    CodexRequestError,
    CodexServerRequest,
    CodexThreadHandle,
    CodexThreadSettings,
    CodexThreadSnapshot,
    CodexTurnHandle,
    CodexTurnSettings,
    CodexUnavailableError,
    WorkConflictError,
    WorkOperationKind,
    WorkOperationState,
    WorkService,
    WorkValidationError,
)
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.runtime.work_sqlite import SQLiteWorkStore


def _id() -> str:
    return str(uuid.uuid4())


async def _eventually(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.01)


class _FakeCodex:
    def __init__(self) -> None:
        self.connection_id = "connection-1"
        self.threads: dict[str, JsonObject] = {}
        self.turns: dict[str, list[JsonObject]] = {}
        self.turn_starts: list[tuple[str, str, CodexTurnSettings, str | None]] = []
        self.interrupts: list[tuple[str, str]] = []
        self.responses: list[tuple[str, int | str, JsonValue]] = []
        self.deleted: list[str] = []
        self._events: asyncio.Queue[CodexAppServerEvent | None] = asyncio.Queue()
        self._start_turn_gate: asyncio.Event | None = None
        self._list_gate: asyncio.Event | None = None
        self.start_turn_seen = asyncio.Event()
        self.list_seen = asyncio.Event()
        self.unavailable = False
        self.complete_before_turn_response = False
        self.advertise_model = True

    async def availability(self) -> CodexAvailability:
        return CodexAvailability(True, "codex", "codex-cli 0.152.0", None)

    async def initialize(self) -> CodexInitialization:
        return CodexInitialization(
            connection_id=self.connection_id,
            user_agent="fake-codex",
            codex_home="/fake/codex",
            platform_family="test",
            platform_os="test",
        )

    async def controls(self, *, cwd: str) -> CodexControlCatalog:
        del cwd
        return CodexControlCatalog(
            models=(
                (
                    {
                        "id": "model-1",
                        "supportedReasoningEfforts": [
                            {"reasoningEffort": "medium"},
                            {"reasoningEffort": "high"},
                        ],
                    },
                )
                if self.advertise_model
                else ()
            ),
            collaboration_modes=({"name": "Plan", "mode": "plan"},),
            permission_profiles=(
                {"id": ":workspace", "name": "Workspace", "allowed": True},
            ),
            config={"approval_policy": "on-request"},
        )

    async def start_thread(self, settings: CodexThreadSettings) -> CodexThreadHandle:
        thread_id = f"thread-{len(self.threads) + 1}"
        thread: JsonObject = {
            "id": thread_id,
            "name": "New Work Session",
            "preview": "",
            "recencyAt": 1,
        }
        self.threads[thread_id] = thread
        response: JsonObject = {
            "thread": dict(thread),
            "model": settings.model or "model-1",
            "reasoningEffort": "medium",
        }
        return CodexThreadHandle(self.connection_id, thread_id, response)

    async def resume_thread(
        self, thread_id: str, settings: CodexThreadSettings
    ) -> CodexThreadHandle:
        del settings
        thread = self._thread(thread_id)
        return CodexThreadHandle(
            self.connection_id,
            thread_id,
            {"thread": dict(thread)},
        )

    async def read_thread(self, thread_id: str) -> CodexThreadSnapshot:
        if self.unavailable:
            raise CodexUnavailableError("Codex is unavailable")
        return CodexThreadSnapshot(thread_id, dict(self._thread(thread_id)))

    async def list_threads_page(
        self, *, cursor: str | None, limit: int
    ) -> CodexObjectPage:
        del cursor, limit
        if self.unavailable:
            raise CodexUnavailableError("Codex is unavailable")
        self.list_seen.set()
        if self._list_gate is not None:
            await self._list_gate.wait()
        return CodexObjectPage(
            tuple(dict(item) for item in self.threads.values()), None, None
        )

    async def list_turns_page(
        self,
        *,
        thread_id: str,
        cursor: str | None,
        limit: int,
    ) -> CodexObjectPage:
        del cursor, limit
        self._thread(thread_id)
        return CodexObjectPage(tuple(self.turns.get(thread_id, ())), None, None)

    async def set_thread_name(self, *, thread_id: str, name: str) -> None:
        self._thread(thread_id)["name"] = name

    async def delete_thread(self, thread_id: str) -> None:
        self._thread(thread_id)
        self.deleted.append(thread_id)
        del self.threads[thread_id]

    async def start_turn(
        self,
        *,
        thread_id: str,
        text: str,
        settings: CodexTurnSettings,
        client_user_message_id: str | None = None,
    ) -> CodexTurnHandle:
        self._thread(thread_id)
        self.turn_starts.append((thread_id, text, settings, client_user_message_id))
        self.start_turn_seen.set()
        if self._start_turn_gate is not None:
            await self._start_turn_gate.wait()
        if self.complete_before_turn_response:
            await self.emit(
                CodexNotification(
                    self.connection_id,
                    "turn/started",
                    {"threadId": thread_id, "turn": {"id": "turn-1"}},
                )
            )
            await self.emit(
                CodexNotification(
                    self.connection_id,
                    "turn/completed",
                    {
                        "threadId": thread_id,
                        "turn": {"id": "turn-1", "status": "completed"},
                    },
                )
            )
            await asyncio.sleep(0)
        return CodexTurnHandle(
            self.connection_id,
            thread_id,
            "turn-1",
            {"turn": {"id": "turn-1"}},
        )

    async def interrupt_turn(self, *, thread_id: str, turn_id: str) -> None:
        self.interrupts.append((thread_id, turn_id))

    async def respond(
        self,
        *,
        connection_id: str,
        request_id: int | str,
        result: JsonValue,
    ) -> None:
        self.responses.append((connection_id, request_id, result))

    async def emit(self, event: CodexAppServerEvent) -> None:
        await self._events.put(event)

    async def close(self) -> None:
        await self._events.put(None)

    async def events(self) -> AsyncIterator[CodexAppServerEvent]:
        while (event := await self._events.get()) is not None:
            yield event

    def gate_next_turn(self) -> asyncio.Event:
        self._start_turn_gate = asyncio.Event()
        return self._start_turn_gate

    def gate_next_list(self) -> asyncio.Event:
        self._list_gate = asyncio.Event()
        return self._list_gate

    def _thread(self, thread_id: str) -> JsonObject:
        try:
            return self.threads[thread_id]
        except KeyError as exc:
            raise CodexRequestError(
                method="thread/read",
                code=-32001,
                message="thread not found",
            ) from exc


async def _service(tmp_path: Path) -> tuple[WorkService, _FakeCodex, SQLiteWorkStore]:
    codex = _FakeCodex()
    store = SQLiteWorkStore(tmp_path / "work.db", tmp_path / "work.lock")
    service = WorkService(codex, store)
    await service.start()
    return service, codex, store


async def _create(
    service: WorkService,
    store: SQLiteWorkStore,
    project: Path,
    *,
    operation_id: str | None = None,
) -> tuple[str, str]:
    operation_id = operation_id or _id()
    existing_ids = {record.thread_id for record in await store.list_sessions()}
    acknowledgement = await service.create_session(
        cwd=str(project),
        operation_id=operation_id,
    )
    assert acknowledgement.state is WorkOperationState.RESERVED
    await _eventually(lambda: _session_count(store) == len(existing_ids) + 1)
    record = next(
        record
        for record in await store.list_sessions()
        if record.thread_id not in existing_ids
    )
    return record.thread_id, operation_id


def _session_count(store: SQLiteWorkStore) -> int:
    database = store._database_path
    if not database.exists():
        return 0
    with closing(sqlite3.connect(database)) as connection:
        return connection.execute("SELECT count(*) FROM work_sessions").fetchone()[0]


def _operation_state(store: SQLiteWorkStore, operation_id: str) -> str | None:
    with closing(sqlite3.connect(store._database_path)) as connection:
        row = connection.execute(
            "SELECT state FROM work_operations WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
    return None if row is None else row[0]


@pytest.mark.asyncio
async def test_create_send_and_project_native_events_into_one_work_session(
    tmp_path: Path,
) -> None:
    service, codex, store = await _service(tmp_path)
    try:
        bootstrap = await service.bootstrap()
        assert bootstrap.available is True
        assert bootstrap.codex_version == "codex-cli 0.152.0"
        thread_id, _create_id = await _create(service, store, tmp_path)

        listed = await service.list_sessions(query="", cursor=None, limit=25)
        assert [item.thread_id for item in listed.sessions] == [thread_id]
        detail = await service.get_detail(thread_id)
        assert detail.summary.project_available is True
        assert detail.summary.session_available is True
        assert detail.controls["models"] == [
            {
                "id": "model-1",
                "supportedReasoningEfforts": [
                    {"reasoningEffort": "medium"},
                    {"reasoningEffort": "high"},
                ],
            }
        ]

        send_id = _id()
        accepted = await service.send(
            thread_id,
            expected_revision=1,
            operation_id=send_id,
            text="Inspect the repository",
        )
        assert accepted.kind is WorkOperationKind.SEND
        await _eventually(lambda: len(codex.turn_starts) == 1)
        assert codex.turn_starts[0][1] == "Inspect the repository"
        assert codex.turn_starts[0][3] == send_id

        await codex.emit(
            CodexNotification(
                codex.connection_id,
                "turn/started",
                {"threadId": thread_id, "turn": {"id": "turn-1"}},
            )
        )
        await codex.emit(
            CodexNotification(
                codex.connection_id,
                "item/agentMessage/delta",
                {
                    "threadId": thread_id,
                    "turnId": "turn-1",
                    "itemId": "message-1",
                    "delta": "Done",
                },
            )
        )
        await codex.emit(
            CodexNotification(
                codex.connection_id,
                "turn/completed",
                {
                    "threadId": thread_id,
                    "turn": {"id": "turn-1", "status": "completed"},
                },
            )
        )
        await _eventually(
            lambda: (
                service._native_status.get(thread_id) is not None
                and thread_id not in service._active_turns
            )
        )

        completed = await service.send(
            thread_id,
            expected_revision=1,
            operation_id=send_id,
            text="Inspect the repository",
        )
        assert completed.state is WorkOperationState.COMPLETED
        assert len(codex.turn_starts) == 1
        detail = await service.get_detail(thread_id)
        assert [(item.kind, item.text) for item in detail.live_items] == [
            ("agentMessage", "Done")
        ]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stop_before_turn_ack_is_latched_and_sent_once(tmp_path: Path) -> None:
    service, codex, store = await _service(tmp_path)
    try:
        thread_id, _create_id = await _create(service, store, tmp_path)
        release_turn = codex.gate_next_turn()
        await service.send(
            thread_id,
            expected_revision=1,
            operation_id=_id(),
            text="Wait",
        )
        await asyncio.wait_for(codex.start_turn_seen.wait(), timeout=2)
        stop_id = _id()
        stopped = await service.stop(thread_id, operation_id=stop_id)
        assert stopped.state is WorkOperationState.RESERVED
        assert codex.interrupts == []

        release_turn.set()
        await _eventually(lambda: codex.interrupts == [(thread_id, "turn-1")])
        await codex.emit(
            CodexNotification(
                codex.connection_id,
                "turn/completed",
                {
                    "threadId": thread_id,
                    "turn": {"id": "turn-1", "status": "interrupted"},
                },
            )
        )
        await _eventually(
            lambda: (
                _operation_state(store, stop_id) == WorkOperationState.COMPLETED.value
            )
        )
        repeated = await service.stop(thread_id, operation_id=stop_id)
        assert repeated.state is WorkOperationState.COMPLETED
        assert codex.interrupts == [(thread_id, "turn-1")]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_interaction_is_answered_once_and_connection_loss_interrupts_turn(
    tmp_path: Path,
) -> None:
    service, codex, store = await _service(tmp_path)
    try:
        thread_id, _create_id = await _create(service, store, tmp_path)
        await service.send(
            thread_id,
            expected_revision=1,
            operation_id=_id(),
            text="Run a command",
        )
        await _eventually(lambda: len(codex.turn_starts) == 1)
        request = CodexServerRequest(
            codex.connection_id,
            "approval-1",
            "item/commandExecution/requestApproval",
            {
                "threadId": thread_id,
                "turnId": "turn-1",
                "command": ["git", "status"],
                "availableDecisions": ["accept", "decline"],
            },
        )
        await codex.emit(request)
        await _eventually(lambda: len(service._interactions) == 1)
        interaction_id = next(iter(service._interactions))
        await service.respond(thread_id, interaction_id, value={"decision": "accept"})
        assert codex.responses[-1] == (
            codex.connection_id,
            "approval-1",
            {"decision": "accept"},
        )
        with pytest.raises(WorkConflictError, match="already answered"):
            await service.respond(
                thread_id,
                interaction_id,
                value={"decision": "decline"},
            )

        await codex.emit(
            CodexServerRequest(
                codex.connection_id,
                "question-1",
                "item/tool/requestUserInput",
                {
                    "threadId": thread_id,
                    "turnId": "turn-1",
                    "questions": [
                        {
                            "id": "choice",
                            "header": "Choice",
                            "question": "Pick one",
                            "options": [
                                {"label": "First", "description": "First option"},
                                {"label": "Second", "description": "Second option"},
                            ],
                        }
                    ],
                },
            )
        )
        await _eventually(lambda: len(service._interactions) == 2)
        question_id = next(
            key for key in service._interactions if key != interaction_id
        )
        with pytest.raises(WorkConflictError, match="already answered"):
            await service.respond(
                thread_id,
                interaction_id,
                value={"decision": "decline"},
            )
        with pytest.raises(WorkValidationError, match="offered by Codex"):
            await service.respond(
                thread_id,
                question_id,
                value={"answers": {"choice": ["Third"]}},
            )
        await service.respond(
            thread_id,
            question_id,
            value={"answers": {"choice": ["First"]}},
        )
        assert codex.responses[-1] == (
            codex.connection_id,
            "question-1",
            {"answers": {"choice": {"answers": ["First"]}}},
        )

        await codex.emit(CodexConnectionLost(codex.connection_id, "child stopped"))
        await _eventually(lambda: thread_id not in service._active_turns)
        assert service._native_status[thread_id].value == "disconnected"
        listed = await service.list_sessions(query="", cursor=None, limit=25)
        assert listed.sessions[0].status.value == "ready"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_missing_project_keeps_history_readable_but_blocks_new_turn(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    service, codex, store = await _service(tmp_path)
    try:
        thread_id, _create_id = await _create(service, store, project)
        codex.turns[thread_id] = [
            {
                "id": "turn-old",
                "items": [
                    {
                        "id": "answer-old",
                        "type": "agentMessage",
                        "status": "completed",
                        "text": "Persisted answer",
                    }
                ],
            }
        ]
        project.rmdir()

        detail = await service.get_detail(thread_id)
        assert detail.summary.project_available is False
        assert detail.controls == {
            "models": [],
            "collaboration_modes": [],
            "permission_profiles": [],
        }
        assert detail.turns.items[0].text == "Persisted answer"
        with pytest.raises(WorkConflictError, match="project folder is unavailable"):
            await service.send(
                thread_id,
                expected_revision=1,
                operation_id=_id(),
                text="Cannot run",
            )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_settings_and_native_rename_use_revisions_and_catalog_validation(
    tmp_path: Path,
) -> None:
    service, _codex, store = await _service(tmp_path)
    try:
        thread_id, _create_id = await _create(service, store, tmp_path)
        updated = await service.update_settings(
            thread_id,
            expected_revision=1,
            updates={"reasoning_effort": "high", "collaboration_mode": "Plan"},
        )
        assert updated.revision == 2
        assert updated.settings.reasoning_effort == "high"
        with pytest.raises(WorkConflictError, match="another tab"):
            await service.update_settings(
                thread_id,
                expected_revision=1,
                updates={"reasoning_effort": "medium"},
            )
        with pytest.raises(WorkValidationError, match="advertised by Codex"):
            await service.update_settings(
                thread_id,
                expected_revision=2,
                updates={"model": "not-advertised"},
            )

        renamed = await service.rename(
            thread_id,
            expected_revision=2,
            name="  Renamed   Work  ",
        )
        repeated = await service.rename(
            thread_id,
            expected_revision=2,
            name="Renamed Work",
        )
        assert renamed.revision == 3
        assert repeated == renamed
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_one_active_turn_per_session_does_not_serialize_other_sessions(
    tmp_path: Path,
) -> None:
    first_project = tmp_path / "first"
    second_project = tmp_path / "second"
    first_project.mkdir()
    second_project.mkdir()
    service, codex, store = await _service(tmp_path)
    release_turns = codex.gate_next_turn()
    try:
        first, _first_create = await _create(service, store, first_project)
        second, _second_create = await _create(service, store, second_project)
        first_send = await service.send(
            first,
            expected_revision=1,
            operation_id=_id(),
            text="First",
        )
        second_send = await service.send(
            second,
            expected_revision=1,
            operation_id=_id(),
            text="Second",
        )
        await _eventually(lambda: len(codex.turn_starts) == 2)

        assert first_send.state is WorkOperationState.RESERVED
        assert second_send.state is WorkOperationState.RESERVED
        assert {entry[0] for entry in codex.turn_starts} == {first, second}
        with pytest.raises(WorkConflictError, match="already has an active turn"):
            await service.send(
                first,
                expected_revision=1,
                operation_id=_id(),
                text="Conflicting",
            )
    finally:
        release_turns.set()
        await service.close()


@pytest.mark.asyncio
async def test_removed_catalog_selection_blocks_the_next_turn(
    tmp_path: Path,
) -> None:
    service, codex, store = await _service(tmp_path)
    try:
        thread_id, _create_id = await _create(service, store, tmp_path)
        codex.advertise_model = False
        operation_id = _id()
        await service.send(
            thread_id,
            expected_revision=1,
            operation_id=operation_id,
            text="Do not silently substitute a model",
        )
        await _eventually(
            lambda: (
                _operation_state(store, operation_id) == WorkOperationState.FAILED.value
            )
        )

        assert codex.turn_starts == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_delete_removes_native_and_registry_membership(tmp_path: Path) -> None:
    service, codex, store = await _service(tmp_path)
    try:
        thread_id, _create_id = await _create(service, store, tmp_path)
        delete_id = _id()
        await service.delete(thread_id, operation_id=delete_id)
        await _eventually(lambda: thread_id not in service._records)

        repeated = await service.delete(thread_id, operation_id=delete_id)

        assert codex.deleted == [thread_id]
        assert repeated.state is WorkOperationState.COMPLETED
        assert await store.list_sessions() == ()
        with closing(sqlite3.connect(tmp_path / "work.db")) as connection:
            rows = connection.execute(
                "SELECT operation_id, state FROM work_operations"
            ).fetchall()
        assert rows == [(delete_id, WorkOperationState.COMPLETED.value)]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_native_scan_does_not_mark_a_concurrent_create_missing(
    tmp_path: Path,
) -> None:
    service, codex, store = await _service(tmp_path)
    release_list = codex.gate_next_list()
    try:
        listing = asyncio.create_task(
            service.list_sessions(query="", cursor=None, limit=25)
        )
        await asyncio.wait_for(codex.list_seen.wait(), timeout=2)
        thread_id, _create_id = await _create(service, store, tmp_path)
        release_list.set()

        page = await asyncio.wait_for(listing, timeout=2)
        assert page.sessions[0].thread_id == thread_id
        assert page.sessions[0].session_available is True
    finally:
        release_list.set()
        await service.close()


@pytest.mark.asyncio
async def test_unavailable_codex_keeps_registered_sessions_visible(
    tmp_path: Path,
) -> None:
    service, codex, store = await _service(tmp_path)
    try:
        thread_id, _create_id = await _create(service, store, tmp_path)
        codex.unavailable = True

        listed = await service.list_sessions(query="", cursor=None, limit=25)
        detail = await service.get_detail(thread_id)
        assert listed.sessions[0].thread_id == thread_id
        assert listed.sessions[0].session_available is True
        assert listed.sessions[0].status.value == "disconnected"
        assert detail.summary.status.value == "disconnected"
        assert detail.controls == {}

        codex.unavailable = False
        recovered = await service.get_detail(thread_id)
        assert recovered.summary.status.value == "ready"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_terminal_turn_event_cannot_be_regressed_by_late_start_ack(
    tmp_path: Path,
) -> None:
    service, codex, store = await _service(tmp_path)
    try:
        thread_id, _create_id = await _create(service, store, tmp_path)
        codex.complete_before_turn_response = True
        operation_id = _id()
        await service.send(
            thread_id,
            expected_revision=1,
            operation_id=operation_id,
            text="Complete immediately",
        )
        await _eventually(
            lambda: (
                _operation_state(store, operation_id)
                == WorkOperationState.COMPLETED.value
            )
        )

        await asyncio.sleep(0.05)
        repeated = await service.send(
            thread_id,
            expected_revision=1,
            operation_id=operation_id,
            text="Complete immediately",
        )
        assert repeated.state is WorkOperationState.COMPLETED
        assert len(codex.turn_starts) == 1
    finally:
        await service.close()
