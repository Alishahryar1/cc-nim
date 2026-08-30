import sqlite3
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
async def test_store_persists_generation_segments_and_actual_fallback(tmp_path: Path):
    store = _store(tmp_path)
    await store.start()
    try:
        session = await store.create_session(
            session_id=_id(), model="groq/requested", reasoning=ChatReasoning.HIGH
        )
        generation_id = _id()
        turn = await store.begin_send(
            session.id,
            expected_revision=session.revision,
            turn_id=_id(),
            generation_id=generation_id,
            user_text="Explain this",
            requested_model=session.model,
            reasoning=session.reasoning,
            effective_output_limit=4096,
        )
        await store.set_generation_actual_model(generation_id, "open_router/fallback")
        await store.replace_generation_segments(
            generation_id,
            (
                ChatSegment(0, SegmentKind.THINKING, "considering"),
                ChatSegment(1, SegmentKind.TEXT, "answer"),
            ),
        )
        await store.finish_generation(
            generation_id,
            status=GenerationStatus.COMPLETED,
            stop_reason="end_turn",
            error_code=None,
            error_message=None,
        )

        stored = (await store.get_transcript(session.id)).turns[0]
        assert stored.id == turn.id
        assert stored.generation.actual_model == "open_router/fallback"
        assert [segment.text for segment in stored.generation.segments] == [
            "considering",
            "answer",
        ]
    finally:
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
        assert (await store.get_transcript(session.id)).turns[
            0
        ].generation.id == original_id

        await store.replace_generation_segments(
            replacement_id, (ChatSegment(0, SegmentKind.TEXT, "replacement"),)
        )
        await store.finish_generation(
            replacement_id,
            status=GenerationStatus.COMPLETED,
            stop_reason="end_turn",
            error_code=None,
            error_message=None,
        )
        await store.complete_regeneration(replacement_id)
        visible = (await store.get_transcript(session.id)).turns[0].generation
        assert visible.id == replacement_id
        assert visible.segments[0].text == "replacement"
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
