import asyncio
import sqlite3
import threading
import uuid
from contextlib import closing
from pathlib import Path

import pytest

from free_claude_code.application.chat import (
    DEFAULT_CHAT_SYSTEM_PROMPT,
    ChatConflictError,
    ChatReasoning,
    ChatSegment,
    ChatUnavailableError,
    GenerationStatus,
    SegmentKind,
)
from free_claude_code.runtime.chat_sqlite import SQLiteChatStore


@pytest.mark.asyncio
@pytest.mark.parametrize("disable_reasoning", [True, False])
@pytest.mark.parametrize("healthy_preference", [True, False])
async def test_retirement_repairs_current_selections_only(
    tmp_path, disable_reasoning, healthy_preference
):
    store = _store(tmp_path)
    await store.start()
    try:
        retired = await store.create_session(
            session_id=_id(),
            model="github_models/vendor/old",
            reasoning=ChatReasoning.HIGH,
        )
        generation_id = _id()
        await store.begin_send(
            retired.id,
            expected_revision=retired.revision,
            turn_id=_id(),
            generation_id=generation_id,
            operation_id=_id(),
            user_text="Keep history",
            requested_model=retired.model,
            reasoning=retired.reasoning,
            effective_output_limit=100,
        )
        await store.set_generation_actual_model(generation_id, retired.model)
        await store.finish_generation(
            generation_id,
            status=GenerationStatus.COMPLETED,
            stop_reason="end_turn",
            error_code=None,
            error_message=None,
        )
        await store.upsert_compaction(
            retired.id,
            covered_through_sequence=1,
            summary="Keep summary",
            estimated_tokens=5,
            requested_model=retired.model,
            actual_model=retired.model,
        )
        before = await store.get_transcript(retired.id)
        untouched = [
            await store.create_session(
                session_id=_id(), model=model, reasoning=ChatReasoning.MEDIUM
            )
            for model in ("githubXmodels/old", "github_models/", "groq/healthy")
        ]
        if not healthy_preference:
            temporary = await store.create_session(
                session_id=_id(), model=retired.model, reasoning=ChatReasoning.HIGH
            )
            await store.delete_session(
                temporary.id, expected_revision=temporary.revision
            )
        preferences = await store.load_preferences()
        assert (
            await store.repair_retired_model_selections(
                retired_provider_ids=frozenset({"github_models"}),
                default_model="groq/default",
                disable_reasoning=disable_reasoning,
            )
            == 1
        )
        after = await store.get_transcript(retired.id)
        assert after.session.model == "groq/default"
        assert after.session.revision == before.session.revision + 1
        assert after.session.reasoning is (
            ChatReasoning.OFF if disable_reasoning else ChatReasoning.HIGH
        )
        assert after.session.title == before.session.title
        assert after.turns == before.turns
        assert after.compaction == before.compaction
        for session in untouched:
            assert await store.get_session(session.id) == session
        updated_preferences = await store.load_preferences()
        if healthy_preference:
            assert updated_preferences == preferences
        else:
            assert updated_preferences.last_model == "groq/default"
            assert updated_preferences.last_reasoning is (
                ChatReasoning.OFF if disable_reasoning else ChatReasoning.HIGH
            )
        assert (
            await store.repair_retired_model_selections(
                retired_provider_ids=frozenset({"github_models"}),
                default_model="groq/default",
                disable_reasoning=disable_reasoning,
            )
            == 0
        )
        assert await store.get_transcript(retired.id) == after
        assert await store.load_preferences() == updated_preferences
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_retirement_repairs_global_preference_without_sessions(tmp_path):
    store = _store(tmp_path)
    await store.start()
    try:
        session = await store.create_session(
            session_id=_id(), model="github_models/old", reasoning=ChatReasoning.HIGH
        )
        await store.delete_session(session.id, expected_revision=session.revision)
        assert (
            await store.repair_retired_model_selections(
                retired_provider_ids=frozenset({"github_models"}),
                default_model="groq/default",
                disable_reasoning=True,
            )
            == 0
        )
        preferences = await store.load_preferences()
        assert preferences.last_model == "groq/default"
        assert preferences.last_reasoning is ChatReasoning.OFF
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_retirement_rolls_back_sessions_when_preference_update_fails(tmp_path):
    store = _store(tmp_path)
    await store.start()
    try:
        session = await store.create_session(
            session_id=_id(), model="github_models/old", reasoning=ChatReasoning.HIGH
        )
        before = await store.load_preferences()
        with closing(sqlite3.connect(tmp_path / "chat.db")) as connection:
            connection.execute(
                "CREATE TRIGGER fail_preference BEFORE UPDATE ON chat_settings BEGIN SELECT RAISE(ABORT, 'injected'); END"
            )
            connection.commit()
        with pytest.raises(ChatUnavailableError):
            await store.repair_retired_model_selections(
                retired_provider_ids=frozenset({"github_models"}),
                default_model="groq/default",
                disable_reasoning=True,
            )
        assert await store.get_session(session.id) == session
        assert await store.load_preferences() == before
    finally:
        await store.close()


def _id() -> str:
    return str(uuid.uuid4())


def _store(tmp_path: Path) -> SQLiteChatStore:
    return SQLiteChatStore(tmp_path / "chat.db", tmp_path / "chat.lock")


@pytest.mark.asyncio
async def test_store_creates_schema_preferences_and_searchable_sessions(tmp_path: Path):
    store = _store(tmp_path)
    await store.start()
    try:
        preferences = await store.load_preferences()
        assert preferences.system_prompt == DEFAULT_CHAT_SYSTEM_PROMPT
        assert preferences.last_reasoning is ChatReasoning.MEDIUM

        first = await store.create_session(
            session_id=_id(), model="groq/first", reasoning=ChatReasoning.MEDIUM
        )
        second = await store.create_session(
            session_id=_id(), model="open_router/second", reasoning=ChatReasoning.OFF
        )
        renamed = await store.update_session(
            first.id,
            expected_revision=first.revision,
            title="Café launch",
            model=None,
            reasoning=None,
        )

        page = await store.list_sessions(query="CAFÉ", cursor=None, limit=25)
        assert [session.id for session in page.sessions] == [renamed.id]
        assert await store.get_session_summary(renamed.id) == page.sessions[0]
        assert (await store.load_preferences()).last_model == second.model
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_store_rejects_stale_session_revision_atomically(tmp_path: Path):
    store = _store(tmp_path)
    await store.start()
    try:
        session = await store.create_session(
            session_id=_id(), model="groq/model", reasoning=ChatReasoning.OFF
        )
        updated = await store.update_session(
            session.id,
            expected_revision=session.revision,
            title="Current",
            model=None,
            reasoning=None,
        )
        with pytest.raises(ChatConflictError, match="another tab"):
            await store.update_session(
                session.id,
                expected_revision=session.revision,
                title="Stale",
                model=None,
                reasoning=None,
            )
        with pytest.raises(ChatConflictError, match="another tab"):
            await store.delete_session(
                session.id,
                expected_revision=session.revision,
            )
        assert (await store.get_session(session.id)).title == updated.title
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_user_title_new_chat_is_not_replaced_on_first_turn(tmp_path: Path):
    store = _store(tmp_path)
    await store.start()
    try:
        session = await store.create_session(
            session_id=_id(), model="groq/model", reasoning=ChatReasoning.OFF
        )
        renamed = await store.update_session(
            session.id,
            expected_revision=session.revision,
            title="New chat",
            model=None,
            reasoning=None,
        )

        await store.begin_send(
            session.id,
            expected_revision=renamed.revision,
            turn_id=_id(),
            generation_id=_id(),
            operation_id=_id(),
            user_text="First question",
            requested_model=session.model,
            reasoning=session.reasoning,
            effective_output_limit=1024,
        )

        assert (await store.get_session(session.id)).title == "New chat"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_store_persists_generation_segments_and_actual_fallback(tmp_path: Path):
    store = _store(tmp_path)
    await store.start()
    try:
        session = await store.create_session(
            session_id=_id(), model="groq/requested", reasoning=ChatReasoning.HIGH
        )
        generation_id = _id()
        operation_id = _id()
        turn = await store.begin_send(
            session.id,
            expected_revision=session.revision,
            turn_id=_id(),
            generation_id=generation_id,
            operation_id=operation_id,
            user_text="Explain this",
            requested_model=session.model,
            reasoning=session.reasoning,
            effective_output_limit=4096,
        )
        assert turn.operation_id == operation_id
        assert await store.generation_start_committed(
            session.id,
            generation_id=generation_id,
            staged=False,
        )
        assert not await store.generation_start_committed(
            session.id,
            generation_id=generation_id,
            staged=True,
        )
        await store.set_generation_actual_model(generation_id, "open_router/fallback")
        await store.replace_generation_segments(
            generation_id,
            (
                ChatSegment(0, SegmentKind.THINKING, "considering"),
                ChatSegment(1, SegmentKind.TEXT, "answer"),
            ),
        )
        completed = await store.finish_generation(
            generation_id,
            status=GenerationStatus.COMPLETED,
            stop_reason="end_turn",
            error_code=None,
            error_message=None,
        )
        repeated = await store.finish_generation(
            generation_id,
            status=GenerationStatus.STOPPED,
            stop_reason="stopped",
            error_code=None,
            error_message=None,
        )

        stored = (await store.get_transcript(session.id)).turns[0]
        assert repeated.revision == completed.revision
        assert stored.generation.status is GenerationStatus.COMPLETED
        assert stored.generation.stop_reason == "end_turn"
        assert stored.id == turn.id
        assert stored.generation.actual_model == "open_router/fallback"
        assert [segment.text for segment in stored.generation.segments] == [
            "considering",
            "answer",
        ]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_transcript_reads_one_revision_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    await store.start()
    release_read = threading.Event()
    session_read = threading.Event()
    try:
        session = await store.create_session(
            session_id=_id(), model="groq/model", reasoning=ChatReasoning.OFF
        )
        generation_id = _id()
        await store.begin_send(
            session.id,
            expected_revision=session.revision,
            turn_id=_id(),
            generation_id=generation_id,
            operation_id=_id(),
            user_text="Keep one coherent snapshot",
            requested_model=session.model,
            reasoning=session.reasoning,
            effective_output_limit=4_096,
        )
        original_get_session = store._get_session
        gate_next_read = True

        def gated_get_session(
            connection: sqlite3.Connection,
            session_id: str,
        ):
            nonlocal gate_next_read
            current = original_get_session(connection, session_id)
            if gate_next_read:
                gate_next_read = False
                session_read.set()
                if not release_read.wait(timeout=5):
                    raise AssertionError(
                        "Timed out waiting to finish the transcript read."
                    )
            return current

        monkeypatch.setattr(store, "_get_session", gated_get_session)
        transcript_task = asyncio.create_task(store.get_transcript(session.id))
        assert await asyncio.to_thread(session_read.wait, 1)

        completed = await store.finish_generation(
            generation_id,
            status=GenerationStatus.COMPLETED,
            stop_reason="end_turn",
            error_code=None,
            error_message=None,
        )
        release_read.set()
        transcript = await asyncio.wait_for(transcript_task, timeout=1)

        assert completed.revision == transcript.session.revision + 1
        assert transcript.turns[0].generation.status is GenerationStatus.RUNNING
    finally:
        release_read.set()
        await store.close()


@pytest.mark.asyncio
async def test_retry_reuses_generation_and_replaces_partial_output(tmp_path: Path):
    store = _store(tmp_path)
    await store.start()
    try:
        session = await store.create_session(
            session_id=_id(), model="groq/model", reasoning=ChatReasoning.OFF
        )
        generation_id = _id()
        await store.begin_send(
            session.id,
            expected_revision=session.revision,
            turn_id=_id(),
            generation_id=generation_id,
            operation_id=_id(),
            user_text="hello",
            requested_model=session.model,
            reasoning=session.reasoning,
            effective_output_limit=1024,
        )
        await store.replace_generation_segments(
            generation_id, (ChatSegment(0, SegmentKind.TEXT, "partial"),)
        )
        session = await store.finish_generation(
            generation_id,
            status=GenerationStatus.STOPPED,
            stop_reason="stopped",
            error_code=None,
            error_message=None,
        )
        retried = await store.begin_retry(
            session.id,
            expected_revision=session.revision,
            requested_model="open_router/new",
            reasoning=ChatReasoning.LOW,
            effective_output_limit=2048,
        )

        assert retried.id == generation_id
        assert retried.status is GenerationStatus.RUNNING
        assert retried.segments == ()
        assert retried.requested_model == "open_router/new"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_regeneration_keeps_visible_answer_until_atomic_swap(tmp_path: Path):
    store = _store(tmp_path)
    await store.start()
    try:
        session = await store.create_session(
            session_id=_id(), model="groq/model", reasoning=ChatReasoning.OFF
        )
        original_id = _id()
        await store.begin_send(
            session.id,
            expected_revision=session.revision,
            turn_id=_id(),
            generation_id=original_id,
            operation_id=_id(),
            user_text="hello",
            requested_model=session.model,
            reasoning=session.reasoning,
            effective_output_limit=1024,
        )
        await store.replace_generation_segments(
            original_id, (ChatSegment(0, SegmentKind.TEXT, "original"),)
        )
        session = await store.finish_generation(
            original_id,
            status=GenerationStatus.COMPLETED,
            stop_reason="end_turn",
            error_code=None,
            error_message=None,
        )
        replacement_id = _id()
        _turn, replacement = await store.begin_regenerate(
            session.id,
            expected_revision=session.revision,
            generation_id=replacement_id,
            requested_model=session.model,
            reasoning=session.reasoning,
            effective_output_limit=1024,
        )
        assert replacement.id == replacement_id
        assert await store.generation_start_committed(
            session.id,
            generation_id=replacement_id,
            staged=True,
        )
        assert not await store.generation_start_committed(
            session.id,
            generation_id=replacement_id,
            staged=False,
        )
        assert (await store.get_transcript(session.id)).turns[
            0
        ].generation.id == original_id

        await store.replace_generation_segments(
            replacement_id, (ChatSegment(0, SegmentKind.TEXT, "replacement"),)
        )
        with pytest.raises(ChatConflictError, match="Staged"):
            await store.finish_generation(
                replacement_id,
                status=GenerationStatus.COMPLETED,
                stop_reason="end_turn",
                error_code=None,
                error_message=None,
            )
        await store.finish_regeneration(
            replacement_id,
            status=GenerationStatus.COMPLETED,
            stop_reason="end_turn",
            error_code=None,
            error_message=None,
        )
        visible = (await store.get_transcript(session.id)).turns[0].generation
        assert visible.id == replacement_id
        assert visible.status is GenerationStatus.COMPLETED
        assert visible.stop_reason == "end_turn"
        assert visible.segments[0].text == "replacement"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_failed_regeneration_atomically_replaces_visible_answer(tmp_path: Path):
    store = _store(tmp_path)
    await store.start()
    try:
        session = await store.create_session(
            session_id=_id(), model="groq/model", reasoning=ChatReasoning.OFF
        )
        original_id = _id()
        await store.begin_send(
            session.id,
            expected_revision=session.revision,
            turn_id=_id(),
            generation_id=original_id,
            operation_id=_id(),
            user_text="hello",
            requested_model=session.model,
            reasoning=session.reasoning,
            effective_output_limit=1024,
        )
        session = await store.finish_generation(
            original_id,
            status=GenerationStatus.COMPLETED,
            stop_reason="end_turn",
            error_code=None,
            error_message=None,
        )
        replacement_id = _id()
        await store.begin_regenerate(
            session.id,
            expected_revision=session.revision,
            generation_id=replacement_id,
            requested_model=session.model,
            reasoning=session.reasoning,
            effective_output_limit=1024,
        )
        await store.replace_generation_segments(
            replacement_id, (ChatSegment(0, SegmentKind.TEXT, "partial"),)
        )

        finished = await store.finish_regeneration(
            replacement_id,
            status=GenerationStatus.FAILED,
            stop_reason=None,
            error_code="provider_error",
            error_message="provider failed",
        )
        repeated = await store.finish_regeneration(
            replacement_id,
            status=GenerationStatus.FAILED,
            stop_reason=None,
            error_code="provider_error",
            error_message="provider failed",
        )

        visible = (await store.get_transcript(session.id)).turns[0].generation
        assert repeated.revision == finished.revision
        assert visible.id == replacement_id
        assert visible.status is GenerationStatus.FAILED
        assert visible.error_code == "provider_error"
        assert visible.error_message == "provider failed"
        assert visible.segments[0].text == "partial"
        with closing(sqlite3.connect(tmp_path / "chat.db")) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM chat_generations WHERE turn_id = "
                "(SELECT turn_id FROM chat_generations WHERE id = ?)",
                (replacement_id,),
            ).fetchone() == (1,)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_startup_recovers_visible_and_discards_staged_running_generations(
    tmp_path: Path,
):
    store = _store(tmp_path)
    await store.start()
    session = await store.create_session(
        session_id=_id(), model="groq/model", reasoning=ChatReasoning.OFF
    )
    original_id = _id()
    await store.begin_send(
        session.id,
        expected_revision=session.revision,
        turn_id=_id(),
        generation_id=original_id,
        operation_id=_id(),
        user_text="hello",
        requested_model=session.model,
        reasoning=session.reasoning,
        effective_output_limit=1024,
    )
    await store.close()

    reopened = _store(tmp_path)
    await reopened.start()
    try:
        generation = (await reopened.get_transcript(session.id)).turns[0].generation
        assert generation.status is GenerationStatus.INTERRUPTED
        assert generation.stop_reason == "server_restart"
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_startup_discards_uncommitted_regeneration_and_keeps_visible_answer(
    tmp_path: Path,
):
    store = _store(tmp_path)
    await store.start()
    session = await store.create_session(
        session_id=_id(), model="groq/model", reasoning=ChatReasoning.OFF
    )
    original_id = _id()
    await store.begin_send(
        session.id,
        expected_revision=session.revision,
        turn_id=_id(),
        generation_id=original_id,
        operation_id=_id(),
        user_text="hello",
        requested_model=session.model,
        reasoning=session.reasoning,
        effective_output_limit=1024,
    )
    await store.replace_generation_segments(
        original_id, (ChatSegment(0, SegmentKind.TEXT, "original"),)
    )
    session = await store.finish_generation(
        original_id,
        status=GenerationStatus.COMPLETED,
        stop_reason="end_turn",
        error_code=None,
        error_message=None,
    )
    staged_id = _id()
    await store.begin_regenerate(
        session.id,
        expected_revision=session.revision,
        generation_id=staged_id,
        requested_model=session.model,
        reasoning=session.reasoning,
        effective_output_limit=1024,
    )
    await store.close()

    reopened = _store(tmp_path)
    await reopened.start()
    try:
        visible = (await reopened.get_transcript(session.id)).turns[0].generation
        assert visible.id == original_id
        with closing(sqlite3.connect(tmp_path / "chat.db")) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM chat_generations WHERE id = ?", (staged_id,)
            ).fetchone() == (0,)
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_startup_discards_terminal_staged_regeneration(tmp_path: Path):
    store = _store(tmp_path)
    await store.start()
    session = await store.create_session(
        session_id=_id(), model="groq/model", reasoning=ChatReasoning.OFF
    )
    original_id = _id()
    await store.begin_send(
        session.id,
        expected_revision=session.revision,
        turn_id=_id(),
        generation_id=original_id,
        operation_id=_id(),
        user_text="hello",
        requested_model=session.model,
        reasoning=session.reasoning,
        effective_output_limit=1024,
    )
    session = await store.finish_generation(
        original_id,
        status=GenerationStatus.COMPLETED,
        stop_reason="end_turn",
        error_code=None,
        error_message=None,
    )
    staged_id = _id()
    await store.begin_regenerate(
        session.id,
        expected_revision=session.revision,
        generation_id=staged_id,
        requested_model=session.model,
        reasoning=session.reasoning,
        effective_output_limit=1024,
    )
    await store.close()

    with closing(sqlite3.connect(tmp_path / "chat.db")) as connection:
        connection.execute(
            "UPDATE chat_generations SET status = 'completed' WHERE id = ?",
            (staged_id,),
        )
        connection.commit()

    reopened = _store(tmp_path)
    await reopened.start()
    try:
        with closing(sqlite3.connect(tmp_path / "chat.db")) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM chat_generations WHERE id = ?", (staged_id,)
            ).fetchone() == (0,)
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_failed_initial_schema_creation_does_not_claim_schema_version(
    tmp_path: Path,
):
    database = tmp_path / "chat.db"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("CREATE TABLE chat_settings (id INTEGER PRIMARY KEY)")

    store = _store(tmp_path)
    with pytest.raises(ChatUnavailableError, match="storage is unavailable"):
        await store.start()

    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)


@pytest.mark.asyncio
async def test_newer_schema_and_lock_contention_disable_only_chat(tmp_path: Path):
    database = tmp_path / "newer.db"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("PRAGMA user_version = 99")
    newer = SQLiteChatStore(database, tmp_path / "newer.lock")
    with pytest.raises(ChatUnavailableError, match="newer FCC version"):
        await newer.start()

    first = _store(tmp_path / "locked")
    second = _store(tmp_path / "locked")
    await first.start()
    try:
        with pytest.raises(ChatUnavailableError, match="another FCC server"):
            await second.start()
    finally:
        await first.close()
