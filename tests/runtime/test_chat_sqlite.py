import asyncio
import sqlite3
import threading
import uuid
from contextlib import closing
from pathlib import Path

import pytest

from free_claude_code.application.chat import (
    DEFAULT_CHAT_SYSTEM_PROMPT,
    ChatAttachmentFileInfo,
    ChatAttachmentKind,
    ChatConflictError,
    ChatPayloadTooLargeError,
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
async def test_store_migrates_existing_v1_database_to_attachment_schema(
    tmp_path: Path,
):
    store = _store(tmp_path)
    await store.start()
    await store.close()
    with closing(sqlite3.connect(tmp_path / "chat.db")) as connection:
        connection.execute("DROP TABLE chat_attachments")
        connection.execute("PRAGMA user_version = 1")
        connection.commit()

    reopened = _store(tmp_path)
    await reopened.start()
    try:
        with closing(sqlite3.connect(tmp_path / "chat.db")) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()
            table = connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'chat_attachments'"
            ).fetchone()
        assert version == (2,)
        assert table == ("chat_attachments",)
    finally:
        await reopened.close()


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


@pytest.mark.parametrize("user_text", ["", " \n\t "])
@pytest.mark.asyncio
async def test_attachment_only_turn_uses_first_filename_as_session_preview(
    tmp_path: Path,
    user_text: str,
):
    store = _store(tmp_path)
    await store.start()
    try:
        session = await store.create_session(
            session_id=_id(), model="groq/model", reasoning=ChatReasoning.OFF
        )
        attachment = await store.add_staged_attachment(
            session.id,
            attachment_id=_id(),
            filename="evidence.txt",
            file_info=ChatAttachmentFileInfo(
                kind=ChatAttachmentKind.TEXT,
                media_type="text/plain",
                byte_size=8,
                extracted_characters=8,
            ),
        )
        await store.begin_send(
            session.id,
            expected_revision=session.revision,
            turn_id=_id(),
            generation_id=_id(),
            operation_id=_id(),
            user_text=user_text,
            requested_model=session.model,
            reasoning=session.reasoning,
            effective_output_limit=1_024,
            attachment_ids=(attachment.id,),
        )

        page = await store.list_sessions(query="", cursor=None, limit=25)

        assert page.sessions[0].preview == "evidence.txt"
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
async def test_send_atomically_binds_exact_staged_attachments_in_request_order(
    tmp_path: Path,
):
    store = _store(tmp_path)
    await store.start()
    try:
        session = await store.create_session(
            session_id=_id(), model="groq/model", reasoning=ChatReasoning.OFF
        )
        info = ChatAttachmentFileInfo(
            kind=ChatAttachmentKind.TEXT,
            media_type="text/plain",
            byte_size=5,
            extracted_characters=5,
        )
        first_id, second_id, left_id = _id(), _id(), _id()
        for attachment_id, filename in (
            (first_id, "first.txt"),
            (second_id, "second.txt"),
            (left_id, "later.txt"),
        ):
            await store.add_staged_attachment(
                session.id,
                attachment_id=attachment_id,
                filename=filename,
                file_info=info,
            )

        turn = await store.begin_send(
            session.id,
            expected_revision=session.revision,
            turn_id=_id(),
            generation_id=_id(),
            operation_id=_id(),
            user_text="",
            requested_model=session.model,
            reasoning=session.reasoning,
            effective_output_limit=1_024,
            attachment_ids=(second_id, first_id),
        )

        assert [attachment.id for attachment in turn.attachments] == [
            second_id,
            first_id,
        ]
        assert [
            attachment.id
            for attachment in await store.list_staged_attachments(session.id)
        ] == [left_id]
        persisted = (await store.get_transcript(session.id)).turns[0]
        assert [attachment.id for attachment in persisted.attachments] == [
            second_id,
            first_id,
        ]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_missing_selected_attachment_rolls_back_the_whole_turn(tmp_path: Path):
    store = _store(tmp_path)
    await store.start()
    try:
        session = await store.create_session(
            session_id=_id(), model="groq/model", reasoning=ChatReasoning.OFF
        )

        with pytest.raises(ChatConflictError, match="another tab"):
            await store.begin_send(
                session.id,
                expected_revision=session.revision,
                turn_id=_id(),
                generation_id=_id(),
                operation_id=_id(),
                user_text="hello",
                requested_model=session.model,
                reasoning=session.reasoning,
                effective_output_limit=1_024,
                attachment_ids=(_id(),),
            )

        assert (await store.get_transcript(session.id)).turns == ()
        assert (await store.get_session(session.id)).revision == session.revision
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_staging_enforces_count_and_combined_byte_limits(tmp_path: Path):
    store = _store(tmp_path)
    await store.start()
    try:
        session = await store.create_session(
            session_id=_id(), model="groq/model", reasoning=ChatReasoning.OFF
        )
        small = ChatAttachmentFileInfo(
            kind=ChatAttachmentKind.TEXT,
            media_type="text/plain",
            byte_size=1,
            extracted_characters=1,
        )
        for index in range(5):
            await store.add_staged_attachment(
                session.id,
                attachment_id=_id(),
                filename=f"{index}.txt",
                file_info=small,
            )
        with pytest.raises(ChatPayloadTooLargeError, match="at most five"):
            await store.add_staged_attachment(
                session.id,
                attachment_id=_id(),
                filename="six.txt",
                file_info=small,
            )

        other = await store.create_session(
            session_id=_id(), model="groq/model", reasoning=ChatReasoning.OFF
        )
        large = ChatAttachmentFileInfo(
            kind=ChatAttachmentKind.IMAGE,
            media_type="image/png",
            byte_size=9 * 1024 * 1024,
            extracted_characters=None,
        )
        for filename in ("one.png", "two.png"):
            await store.add_staged_attachment(
                other.id,
                attachment_id=_id(),
                filename=filename,
                file_info=large,
            )
        remainder = ChatAttachmentFileInfo(
            kind=ChatAttachmentKind.IMAGE,
            media_type="image/png",
            byte_size=8 * 1024 * 1024,
            extracted_characters=None,
        )
        with pytest.raises(ChatPayloadTooLargeError, match="25 MiB"):
            await store.add_staged_attachment(
                other.id,
                attachment_id=_id(),
                filename="three.png",
                file_info=remainder,
            )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_send_rejects_attachment_owned_by_another_session(tmp_path: Path):
    store = _store(tmp_path)
    await store.start()
    try:
        first = await store.create_session(
            session_id=_id(), model="groq/model", reasoning=ChatReasoning.OFF
        )
        second = await store.create_session(
            session_id=_id(), model="groq/model", reasoning=ChatReasoning.OFF
        )
        attachment = await store.add_staged_attachment(
            second.id,
            attachment_id=_id(),
            filename="private.txt",
            file_info=ChatAttachmentFileInfo(
                kind=ChatAttachmentKind.TEXT,
                media_type="text/plain",
                byte_size=4,
                extracted_characters=4,
            ),
        )

        with pytest.raises(ChatConflictError, match="another tab"):
            await store.begin_send(
                first.id,
                expected_revision=first.revision,
                turn_id=_id(),
                generation_id=_id(),
                operation_id=_id(),
                user_text="steal",
                requested_model=first.model,
                reasoning=first.reasoning,
                effective_output_limit=1_024,
                attachment_ids=(attachment.id,),
            )

        assert (await store.get_transcript(first.id)).turns == ()
        assert [item.id for item in await store.list_staged_attachments(second.id)] == [
            attachment.id
        ]
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
