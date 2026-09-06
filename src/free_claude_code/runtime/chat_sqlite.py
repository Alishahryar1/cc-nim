"""SQLite persistence owned exclusively by local Chat Sessions."""

import os
import sqlite3
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TypeVar

import anyio.to_thread

from free_claude_code.application.chat.models import (
    DEFAULT_CHAT_SYSTEM_PROMPT,
    ChatCompaction,
    ChatConflictError,
    ChatGeneration,
    ChatNotFoundError,
    ChatPreferences,
    ChatReasoning,
    ChatSegment,
    ChatSession,
    ChatSessionPage,
    ChatSessionSummary,
    ChatTranscript,
    ChatTurn,
    ChatUnavailableError,
    GenerationStatus,
    SegmentKind,
)
from free_claude_code.core.interprocess_lock import InterprocessFileLock

T = TypeVar("T")

_SCHEMA_VERSION = 1
_BUSY_TIMEOUT_MS = 5_000
_NEW_CHAT_TITLE = "New chat"
_SESSION_SUMMARY_SELECT = """
    SELECT s.*,
        COALESCE((
            SELECT t.user_text FROM chat_turns AS t
            WHERE t.session_id = s.id
            ORDER BY t.sequence DESC LIMIT 1
        ), '') AS preview
    FROM chat_sessions AS s
"""


class SQLiteChatStore:
    """Short per-operation SQLite transactions behind one process lock."""

    def __init__(self, database_path: Path, lock_path: Path) -> None:
        self._database_path = database_path
        self._lock = InterprocessFileLock(lock_path)
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        directory = self._database_path.parent
        await anyio.to_thread.run_sync(
            partial(directory.mkdir, parents=True, exist_ok=True)
        )
        _owner_only(directory, 0o700)
        acquired = await anyio.to_thread.run_sync(self._lock.acquire)
        if not acquired:
            raise ChatUnavailableError(
                "Chat Sessions is already open in another FCC server process."
            )
        try:
            await anyio.to_thread.run_sync(self._run_sync, self._migrate_and_repair)
            _owner_only(self._database_path, 0o600)
            for suffix in ("-wal", "-shm"):
                sidecar = Path(f"{self._database_path}{suffix}")
                if sidecar.exists():
                    _owner_only(sidecar, 0o600)
        except sqlite3.Error as exc:
            self._lock.release()
            raise ChatUnavailableError("Chat storage is unavailable.") from exc
        except BaseException:
            self._lock.release()
            raise
        self._started = True

    async def close(self) -> None:
        self._started = False
        await anyio.to_thread.run_sync(self._lock.release)

    async def load_preferences(self) -> ChatPreferences:
        return await self._run(self._load_preferences)

    async def repair_retired_model_selections(
        self,
        *,
        retired_provider_ids: frozenset[str],
        default_model: str,
        disable_reasoning: bool,
    ) -> int:
        """Repair current selections atomically, preserving historical model facts."""

        def operation(connection: sqlite3.Connection) -> int:
            connection.execute("BEGIN IMMEDIATE")
            repaired = 0
            now = _now_ms()
            for provider_id in sorted(retired_provider_ids):
                prefix = f"{provider_id}/"
                cursor = connection.execute(
                    """
                    UPDATE chat_sessions
                    SET model = ?, reasoning = CASE WHEN ? THEN 'off' ELSE reasoning END,
                        revision = revision + 1, updated_at = ?
                    WHERE substr(model, 1, ?) = ? AND length(trim(substr(model, ?))) > 0
                    """,
                    (
                        default_model,
                        disable_reasoning,
                        now,
                        len(prefix),
                        prefix,
                        len(prefix) + 1,
                    ),
                )
                repaired += cursor.rowcount
                connection.execute(
                    """
                    UPDATE chat_settings
                    SET last_model = ?,
                        last_reasoning = CASE WHEN ? THEN 'off' ELSE last_reasoning END,
                        updated_at = ?
                    WHERE substr(last_model, 1, ?) = ?
                        AND length(trim(substr(last_model, ?))) > 0
                    """,
                    (
                        default_model,
                        disable_reasoning,
                        now,
                        len(prefix),
                        prefix,
                        len(prefix) + 1,
                    ),
                )
            connection.commit()
            return repaired

        return await self._run(operation)

    async def save_system_prompt(self, system_prompt: str) -> ChatPreferences:
        def operation(connection: sqlite3.Connection) -> ChatPreferences:
            now = _now_ms()
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE chat_settings SET system_prompt = ?, updated_at = ? WHERE id = 1",
                (system_prompt, now),
            )
            connection.commit()
            return self._load_preferences(connection)

        return await self._run(operation)

    async def create_session(
        self, *, session_id: str, model: str, reasoning: ChatReasoning
    ) -> ChatSession:
        def operation(connection: sqlite3.Connection) -> ChatSession:
            now = _now_ms()
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO chat_sessions (
                    id, title, title_search, auto_title_pending, model, reasoning,
                    revision, created_at, updated_at
                ) VALUES (?, ?, ?, 1, ?, ?, 1, ?, ?)
                """,
                (
                    session_id,
                    _NEW_CHAT_TITLE,
                    _NEW_CHAT_TITLE.casefold(),
                    model,
                    reasoning.value,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE chat_settings
                SET last_model = ?, last_reasoning = ?, updated_at = ?
                WHERE id = 1
                """,
                (model, reasoning.value, now),
            )
            connection.commit()
            return self._get_session(connection, session_id)

        return await self._run(operation)

    async def list_sessions(
        self,
        *,
        query: str,
        cursor: tuple[int, str] | None,
        limit: int,
    ) -> ChatSessionPage:
        def operation(connection: sqlite3.Connection) -> ChatSessionPage:
            clauses: list[str] = []
            parameters: list[object] = []
            folded = query.strip().casefold()
            if folded:
                clauses.append("instr(s.title_search, ?) > 0")
                parameters.append(folded)
            if cursor is not None:
                clauses.append("(s.updated_at < ? OR (s.updated_at = ? AND s.id < ?))")
                parameters.extend((cursor[0], cursor[0], cursor[1]))
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = connection.execute(
                f"""
                {_SESSION_SUMMARY_SELECT}
                {where}
                ORDER BY s.updated_at DESC, s.id DESC
                LIMIT ?
                """,
                (*parameters, limit + 1),
            ).fetchall()
            has_more = len(rows) > limit
            selected = rows[:limit]
            sessions = tuple(_session_summary_from_row(row) for row in selected)
            next_cursor = None
            if has_more and selected:
                last = selected[-1]
                next_cursor = (_row_int(last, "updated_at"), _row_str(last, "id"))
            return ChatSessionPage(sessions=sessions, next_cursor=next_cursor)

        return await self._run(operation)

    async def get_session_summary(self, session_id: str) -> ChatSessionSummary:
        def operation(connection: sqlite3.Connection) -> ChatSessionSummary:
            row = connection.execute(
                f"{_SESSION_SUMMARY_SELECT} WHERE s.id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise ChatNotFoundError("Chat session not found.")
            return _session_summary_from_row(row)

        return await self._run(operation)

    async def get_session(self, session_id: str) -> ChatSession:
        return await self._run(
            lambda connection: self._get_session(connection, session_id)
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
        def operation(connection: sqlite3.Connection) -> ChatSession:
            connection.execute("BEGIN IMMEDIATE")
            current = self._get_session(connection, session_id)
            _expect_revision(current, expected_revision)
            next_title = current.title if title is None else title
            next_model = current.model if model is None else model
            next_reasoning = current.reasoning if reasoning is None else reasoning
            now = _now_ms()
            cursor = connection.execute(
                """
                UPDATE chat_sessions
                SET title = ?, title_search = ?,
                    auto_title_pending = CASE
                        WHEN ? IS NULL THEN auto_title_pending ELSE 0
                    END,
                    model = ?, reasoning = ?,
                    revision = revision + 1, updated_at = ?
                WHERE id = ? AND revision = ?
                """,
                (
                    next_title,
                    next_title.casefold(),
                    title,
                    next_model,
                    next_reasoning.value,
                    now,
                    session_id,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise ChatConflictError("This chat changed in another tab. Refresh it.")
            if model is not None or reasoning is not None:
                connection.execute(
                    """
                    UPDATE chat_settings
                    SET last_model = ?, last_reasoning = ?, updated_at = ?
                    WHERE id = 1
                    """,
                    (next_model, next_reasoning.value, now),
                )
            connection.commit()
            return self._get_session(connection, session_id)

        return await self._run(operation)

    async def delete_session(self, session_id: str, *, expected_revision: int) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "DELETE FROM chat_sessions WHERE id = ? AND revision = ?",
                (session_id, expected_revision),
            )
            if cursor.rowcount != 1:
                exists = connection.execute(
                    "SELECT 1 FROM chat_sessions WHERE id = ?", (session_id,)
                ).fetchone()
                connection.rollback()
                if exists is None:
                    raise ChatNotFoundError("Chat session not found.")
                raise ChatConflictError("This chat changed in another tab. Refresh it.")
            connection.commit()

        await self._run(operation)

    async def get_transcript(self, session_id: str) -> ChatTranscript:
        def operation(connection: sqlite3.Connection) -> ChatTranscript:
            connection.execute("BEGIN")
            try:
                session = self._get_session(connection, session_id)
                rows = connection.execute(
                    "SELECT * FROM chat_turns WHERE session_id = ? ORDER BY sequence",
                    (session_id,),
                ).fetchall()
                turns = tuple(self._turn_from_row(connection, row) for row in rows)
                transcript = ChatTranscript(
                    session=session,
                    turns=turns,
                    compaction=self._get_compaction(connection, session_id),
                )
                connection.commit()
                return transcript
            except BaseException:
                connection.rollback()
                raise

        return await self._run(operation)

    async def get_turn_page(
        self,
        session_id: str,
        *,
        before_sequence: int | None,
        limit: int,
    ) -> tuple[tuple[ChatTurn, ...], int | None, ChatCompaction | None]:
        def operation(
            connection: sqlite3.Connection,
        ) -> tuple[tuple[ChatTurn, ...], int | None, ChatCompaction | None]:
            self._get_session(connection, session_id)
            parameters: list[object] = [session_id]
            before = ""
            if before_sequence is not None:
                before = "AND sequence < ?"
                parameters.append(before_sequence)
            rows = connection.execute(
                f"""
                SELECT * FROM chat_turns
                WHERE session_id = ? {before}
                ORDER BY sequence DESC LIMIT ?
                """,
                (*parameters, limit + 1),
            ).fetchall()
            has_more = len(rows) > limit
            selected = rows[:limit]
            selected.reverse()
            turns = tuple(self._turn_from_row(connection, row) for row in selected)
            next_before = turns[0].sequence if has_more and turns else None
            return turns, next_before, self._get_compaction(connection, session_id)

        return await self._run(operation)

    async def generation_start_committed(
        self,
        session_id: str,
        *,
        generation_id: str,
        staged: bool,
    ) -> bool:
        def operation(connection: sqlite3.Connection) -> bool:
            row = connection.execute(
                """
                SELECT g.visible, g.status
                FROM chat_generations AS g
                JOIN chat_turns AS t ON t.id = g.turn_id
                WHERE g.id = ? AND t.session_id = ?
                """,
                (generation_id, session_id),
            ).fetchone()
            if row is None:
                return False
            expected_visibility = 0 if staged else 1
            return (
                _row_int(row, "visible") == expected_visibility
                and _row_str(row, "status") == GenerationStatus.RUNNING.value
            )

        return await self._run(operation)

    async def begin_send(
        self,
        session_id: str,
        *,
        expected_revision: int,
        turn_id: str,
        generation_id: str,
        operation_id: str,
        user_text: str,
        requested_model: str,
        reasoning: ChatReasoning,
        effective_output_limit: int,
    ) -> ChatTurn:
        def operation(connection: sqlite3.Connection) -> ChatTurn:
            connection.execute("BEGIN IMMEDIATE")
            session = self._get_session(connection, session_id)
            _expect_revision(session, expected_revision)
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence "
                "FROM chat_turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise ChatUnavailableError("Could not allocate the next chat turn.")
            sequence = _row_int(row, "next_sequence")
            auto_title_row = connection.execute(
                "SELECT auto_title_pending FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if auto_title_row is None:
                raise ChatNotFoundError("Chat session not found.")
            now = _now_ms()
            title = session.title
            if _row_int(auto_title_row, "auto_title_pending") == 1:
                title = _title_from_text(user_text)
            connection.execute(
                """
                INSERT INTO chat_turns (
                    id, session_id, operation_id, sequence, user_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (turn_id, session_id, operation_id, sequence, user_text, now),
            )
            connection.execute(
                """
                INSERT INTO chat_generations (
                    id, turn_id, visible, status, requested_model, actual_model,
                    reasoning, effective_output_limit, stop_reason, error_code,
                    error_message, started_at, finished_at
                ) VALUES (?, ?, 1, ?, ?, NULL, ?, ?, NULL, NULL, NULL, ?, NULL)
                """,
                (
                    generation_id,
                    turn_id,
                    GenerationStatus.RUNNING.value,
                    requested_model,
                    reasoning.value,
                    effective_output_limit,
                    now,
                ),
            )
            cursor = connection.execute(
                """
                UPDATE chat_sessions
                SET title = ?, title_search = ?, auto_title_pending = 0,
                    revision = revision + 1, updated_at = ?
                WHERE id = ? AND revision = ?
                """,
                (title, title.casefold(), now, session_id, expected_revision),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise ChatConflictError("This chat changed in another tab. Refresh it.")
            turn = self._turn_by_id(connection, turn_id)
            connection.commit()
            return turn

        return await self._run(operation)

    async def begin_retry(
        self,
        session_id: str,
        *,
        expected_revision: int,
        requested_model: str,
        reasoning: ChatReasoning,
        effective_output_limit: int,
    ) -> ChatGeneration:
        def operation(connection: sqlite3.Connection) -> ChatGeneration:
            connection.execute("BEGIN IMMEDIATE")
            session = self._get_session(connection, session_id)
            _expect_revision(session, expected_revision)
            row = self._latest_visible_generation_row(connection, session_id)
            if _row_str(row, "status") not in {
                GenerationStatus.STOPPED.value,
                GenerationStatus.INTERRUPTED.value,
                GenerationStatus.FAILED.value,
            }:
                raise ChatConflictError(
                    "Only the latest unfinished answer can be retried."
                )
            generation_id = _row_str(row, "id")
            now = _now_ms()
            connection.execute(
                "DELETE FROM chat_generation_segments WHERE generation_id = ?",
                (generation_id,),
            )
            connection.execute(
                """
                UPDATE chat_generations
                SET status = ?, requested_model = ?, actual_model = NULL,
                    reasoning = ?, effective_output_limit = ?, stop_reason = NULL,
                    error_code = NULL, error_message = NULL, started_at = ?,
                    finished_at = NULL
                WHERE id = ?
                """,
                (
                    GenerationStatus.RUNNING.value,
                    requested_model,
                    reasoning.value,
                    effective_output_limit,
                    now,
                    generation_id,
                ),
            )
            self._bump_session(connection, session_id, expected_revision, now)
            generation = self._generation_by_id(connection, generation_id)
            connection.commit()
            return generation

        return await self._run(operation)

    async def begin_regenerate(
        self,
        session_id: str,
        *,
        expected_revision: int,
        generation_id: str,
        requested_model: str,
        reasoning: ChatReasoning,
        effective_output_limit: int,
    ) -> tuple[ChatTurn, ChatGeneration]:
        def operation(
            connection: sqlite3.Connection,
        ) -> tuple[ChatTurn, ChatGeneration]:
            connection.execute("BEGIN IMMEDIATE")
            session = self._get_session(connection, session_id)
            _expect_revision(session, expected_revision)
            visible = self._latest_visible_generation_row(connection, session_id)
            if _row_str(visible, "status") != GenerationStatus.COMPLETED.value:
                raise ChatConflictError(
                    "Only the latest completed answer can regenerate."
                )
            turn_id = _row_str(visible, "turn_id")
            now = _now_ms()
            connection.execute(
                """
                INSERT INTO chat_generations (
                    id, turn_id, visible, status, requested_model, actual_model,
                    reasoning, effective_output_limit, stop_reason, error_code,
                    error_message, started_at, finished_at
                ) VALUES (?, ?, 0, ?, ?, NULL, ?, ?, NULL, NULL, NULL, ?, NULL)
                """,
                (
                    generation_id,
                    turn_id,
                    GenerationStatus.RUNNING.value,
                    requested_model,
                    reasoning.value,
                    effective_output_limit,
                    now,
                ),
            )
            self._bump_session(connection, session_id, expected_revision, now)
            turn = self._turn_by_id(connection, turn_id)
            generation = self._generation_by_id(connection, generation_id)
            connection.commit()
            return turn, generation

        return await self._run(operation)

    async def set_generation_actual_model(
        self, generation_id: str, actual_model: str
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE chat_generations SET actual_model = ? WHERE id = ?",
                (actual_model, generation_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise ChatNotFoundError("Chat generation not found.")
            connection.commit()

        await self._run(operation)

    async def replace_generation_segments(
        self, generation_id: str, segments: tuple[ChatSegment, ...]
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            if (
                connection.execute(
                    "SELECT 1 FROM chat_generations WHERE id = ?", (generation_id,)
                ).fetchone()
                is None
            ):
                connection.rollback()
                raise ChatNotFoundError("Chat generation not found.")
            connection.execute(
                "DELETE FROM chat_generation_segments WHERE generation_id = ?",
                (generation_id,),
            )
            connection.executemany(
                """
                INSERT INTO chat_generation_segments
                    (generation_id, ordinal, kind, text)
                VALUES (?, ?, ?, ?)
                """,
                (
                    (generation_id, segment.ordinal, segment.kind.value, segment.text)
                    for segment in segments
                ),
            )
            connection.commit()

        await self._run(operation)

    async def finish_generation(
        self,
        generation_id: str,
        *,
        status: GenerationStatus,
        stop_reason: str | None,
        error_code: str | None,
        error_message: str | None,
    ) -> ChatSession:
        if status is GenerationStatus.RUNNING:
            raise ValueError("A generation cannot finish in Running state.")

        def operation(connection: sqlite3.Connection) -> ChatSession:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT g.visible, g.status, t.session_id
                FROM chat_generations AS g
                JOIN chat_turns AS t ON t.id = g.turn_id
                WHERE g.id = ?
                """,
                (generation_id,),
            ).fetchone()
            if row is None:
                raise ChatNotFoundError("Chat generation not found.")
            session_id = _row_str(row, "session_id")
            visible = bool(_row_int(row, "visible"))
            if not visible:
                connection.rollback()
                raise ChatConflictError(
                    "Staged regenerations must finish through finish_regeneration."
                )
            if _row_str(row, "status") != GenerationStatus.RUNNING.value:
                connection.rollback()
                return self._get_session(connection, session_id)
            now = _now_ms()
            cursor = connection.execute(
                """
                UPDATE chat_generations
                SET status = ?, stop_reason = ?, error_code = ?, error_message = ?,
                    finished_at = ?
                WHERE id = ? AND status = ?
                """,
                (
                    status.value,
                    stop_reason,
                    error_code,
                    error_message,
                    now,
                    generation_id,
                    GenerationStatus.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return self._get_session(connection, session_id)
            connection.execute(
                """
                UPDATE chat_sessions
                SET revision = revision + 1, updated_at = ? WHERE id = ?
                """,
                (now, session_id),
            )
            connection.commit()
            return self._get_session(connection, session_id)

        return await self._run(operation)

    async def discard_generation(self, generation_id: str) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT visible FROM chat_generations WHERE id = ?",
                (generation_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return
            if _row_int(row, "visible") != 0:
                connection.rollback()
                raise ChatConflictError("Visible answers cannot be discarded.")
            connection.execute(
                "DELETE FROM chat_generations WHERE id = ?",
                (generation_id,),
            )
            connection.commit()

        await self._run(operation)

    async def finish_regeneration(
        self,
        generation_id: str,
        *,
        status: GenerationStatus,
        stop_reason: str | None,
        error_code: str | None,
        error_message: str | None,
    ) -> ChatSession:
        if status not in {GenerationStatus.COMPLETED, GenerationStatus.FAILED}:
            raise ValueError("A regeneration must finish as Completed or Failed.")

        def operation(connection: sqlite3.Connection) -> ChatSession:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT g.turn_id, g.visible, g.status, t.session_id
                FROM chat_generations AS g
                JOIN chat_turns AS t ON t.id = g.turn_id
                WHERE g.id = ?
                """,
                (generation_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise ChatConflictError("Staged regeneration is unavailable.")
            session_id = _row_str(row, "session_id")
            if bool(_row_int(row, "visible")):
                if _row_str(row, "status") == status.value:
                    connection.rollback()
                    return self._get_session(connection, session_id)
                connection.rollback()
                raise ChatConflictError("Staged regeneration is unavailable.")
            if _row_str(row, "status") != GenerationStatus.RUNNING.value:
                connection.rollback()
                raise ChatConflictError("Staged regeneration is not running.")
            turn_id = _row_str(row, "turn_id")
            now = _now_ms()
            connection.execute(
                "UPDATE chat_generations SET visible = 0 WHERE turn_id = ? AND visible = 1",
                (turn_id,),
            )
            connection.execute(
                """
                UPDATE chat_generations
                SET visible = 1, status = ?, stop_reason = ?, error_code = ?,
                    error_message = ?, finished_at = ?
                WHERE id = ? AND visible = 0 AND status = ?
                """,
                (
                    status.value,
                    stop_reason,
                    error_code,
                    error_message,
                    now,
                    generation_id,
                    GenerationStatus.RUNNING.value,
                ),
            )
            connection.execute(
                "DELETE FROM chat_generations WHERE turn_id = ? AND id != ?",
                (turn_id, generation_id),
            )
            connection.execute(
                """
                UPDATE chat_sessions
                SET revision = revision + 1, updated_at = ? WHERE id = ?
                """,
                (now, session_id),
            )
            connection.commit()
            return self._get_session(connection, session_id)

        return await self._run(operation)

    async def upsert_compaction(
        self,
        session_id: str,
        *,
        covered_through_sequence: int,
        summary: str,
        estimated_tokens: int,
        requested_model: str,
        actual_model: str,
    ) -> ChatCompaction:
        def operation(connection: sqlite3.Connection) -> ChatCompaction:
            connection.execute("BEGIN IMMEDIATE")
            self._get_session(connection, session_id)
            current = self._get_compaction(connection, session_id)
            if current is not None and (
                current.covered_through_sequence == covered_through_sequence
                and current.summary == summary
                and current.estimated_tokens == estimated_tokens
                and current.requested_model == requested_model
                and current.actual_model == actual_model
            ):
                connection.rollback()
                return current
            now = _now_ms()
            connection.execute(
                """
                INSERT INTO chat_compactions (
                    session_id, covered_through_sequence, summary,
                    estimated_tokens, requested_model, actual_model, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    covered_through_sequence = excluded.covered_through_sequence,
                    summary = excluded.summary,
                    estimated_tokens = excluded.estimated_tokens,
                    requested_model = excluded.requested_model,
                    actual_model = excluded.actual_model,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    covered_through_sequence,
                    summary,
                    estimated_tokens,
                    requested_model,
                    actual_model,
                    now,
                ),
            )
            connection.execute(
                """
                UPDATE chat_sessions
                SET revision = revision + 1, updated_at = ? WHERE id = ?
                """,
                (now, session_id),
            )
            compaction = self._get_compaction(connection, session_id)
            if compaction is None:
                connection.rollback()
                raise ChatUnavailableError("Could not persist chat compaction.")
            connection.commit()
            return compaction

        return await self._run(operation)

    async def _run(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        if not self._started:
            raise ChatUnavailableError("Chat storage is not available.")
        try:
            return await anyio.to_thread.run_sync(lambda: self._run_sync(operation))
        except sqlite3.Error as exc:
            raise ChatUnavailableError("Chat storage is unavailable.") from exc

    def _run_sync(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        connection = sqlite3.connect(
            self._database_path,
            timeout=_BUSY_TIMEOUT_MS / 1_000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            connection.execute("PRAGMA journal_mode = WAL")
            return operation(connection)
        finally:
            connection.close()

    def _migrate_and_repair(self, connection: sqlite3.Connection) -> None:
        version_row = connection.execute("PRAGMA user_version").fetchone()
        if version_row is None:
            raise ChatUnavailableError("Could not read the Chat database schema.")
        version_value: object = version_row[0]
        if not isinstance(version_value, int):
            raise ChatUnavailableError("Chat database schema is invalid.")
        if version_value > _SCHEMA_VERSION:
            raise ChatUnavailableError(
                "Chat data was created by a newer FCC version. Update FCC to open it."
            )
        if version_value == 0:
            try:
                connection.executescript(_SCHEMA_V1)
                connection.execute(
                    """
                    INSERT INTO chat_settings (
                        id, system_prompt, last_model, last_reasoning, updated_at
                    ) VALUES (1, ?, NULL, 'medium', 0)
                    """,
                    (DEFAULT_CHAT_SYSTEM_PROMPT,),
                )
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        elif version_value != _SCHEMA_VERSION:
            raise ChatUnavailableError("Chat database schema is unsupported.")

        now = _now_ms()
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            DELETE FROM chat_generations
            WHERE visible = 0
            """
        )
        affected = connection.execute(
            """
            SELECT DISTINCT t.session_id
            FROM chat_generations AS g
            JOIN chat_turns AS t ON t.id = g.turn_id
            WHERE g.visible = 1 AND g.status = 'running'
            """
        ).fetchall()
        connection.execute(
            """
            UPDATE chat_generations
            SET status = 'interrupted', stop_reason = 'server_restart',
                finished_at = ?
            WHERE visible = 1 AND status = 'running'
            """,
            (now,),
        )
        connection.executemany(
            """
            UPDATE chat_sessions
            SET revision = revision + 1, updated_at = ? WHERE id = ?
            """,
            ((_now_ms(), _row_str(row, "session_id")) for row in affected),
        )
        connection.commit()

    @staticmethod
    def _load_preferences(connection: sqlite3.Connection) -> ChatPreferences:
        row = connection.execute("SELECT * FROM chat_settings WHERE id = 1").fetchone()
        if row is None:
            raise ChatUnavailableError("Chat settings are unavailable.")
        return ChatPreferences(
            system_prompt=_row_str(row, "system_prompt"),
            last_model=_row_optional_str(row, "last_model"),
            last_reasoning=ChatReasoning(_row_str(row, "last_reasoning")),
            updated_at=_row_int(row, "updated_at"),
        )

    @staticmethod
    def _get_session(connection: sqlite3.Connection, session_id: str) -> ChatSession:
        row = connection.execute(
            "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise ChatNotFoundError("Chat session not found.")
        return _session_from_row(row)

    @staticmethod
    def _latest_visible_generation_row(
        connection: sqlite3.Connection, session_id: str
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT g.* FROM chat_generations AS g
            JOIN chat_turns AS t ON t.id = g.turn_id
            WHERE t.session_id = ? AND g.visible = 1
            ORDER BY t.sequence DESC LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            raise ChatConflictError("This chat has no answer to operate on.")
        return row

    def _turn_by_id(self, connection: sqlite3.Connection, turn_id: str) -> ChatTurn:
        row = connection.execute(
            "SELECT * FROM chat_turns WHERE id = ?", (turn_id,)
        ).fetchone()
        if row is None:
            raise ChatNotFoundError("Chat turn not found.")
        return self._turn_from_row(connection, row)

    def _turn_from_row(
        self, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> ChatTurn:
        generation_row = connection.execute(
            """
            SELECT * FROM chat_generations
            WHERE turn_id = ? AND visible = 1
            """,
            (_row_str(row, "id"),),
        ).fetchone()
        if generation_row is None:
            raise ChatUnavailableError("Chat turn has no visible generation.")
        return ChatTurn(
            id=_row_str(row, "id"),
            session_id=_row_str(row, "session_id"),
            operation_id=_row_str(row, "operation_id"),
            sequence=_row_int(row, "sequence"),
            user_text=_row_str(row, "user_text"),
            created_at=_row_int(row, "created_at"),
            generation=self._generation_from_row(connection, generation_row),
        )

    def _generation_by_id(
        self, connection: sqlite3.Connection, generation_id: str
    ) -> ChatGeneration:
        row = connection.execute(
            "SELECT * FROM chat_generations WHERE id = ?", (generation_id,)
        ).fetchone()
        if row is None:
            raise ChatNotFoundError("Chat generation not found.")
        return self._generation_from_row(connection, row)

    @staticmethod
    def _generation_from_row(
        connection: sqlite3.Connection, row: sqlite3.Row
    ) -> ChatGeneration:
        segment_rows = connection.execute(
            """
            SELECT * FROM chat_generation_segments
            WHERE generation_id = ? ORDER BY ordinal
            """,
            (_row_str(row, "id"),),
        ).fetchall()
        return ChatGeneration(
            id=_row_str(row, "id"),
            status=GenerationStatus(_row_str(row, "status")),
            requested_model=_row_str(row, "requested_model"),
            actual_model=_row_optional_str(row, "actual_model"),
            reasoning=ChatReasoning(_row_str(row, "reasoning")),
            effective_output_limit=_row_int(row, "effective_output_limit"),
            stop_reason=_row_optional_str(row, "stop_reason"),
            error_code=_row_optional_str(row, "error_code"),
            error_message=_row_optional_str(row, "error_message"),
            started_at=_row_int(row, "started_at"),
            finished_at=_row_optional_int(row, "finished_at"),
            segments=tuple(
                ChatSegment(
                    ordinal=_row_int(segment, "ordinal"),
                    kind=SegmentKind(_row_str(segment, "kind")),
                    text=_row_str(segment, "text"),
                )
                for segment in segment_rows
            ),
        )

    @staticmethod
    def _get_compaction(
        connection: sqlite3.Connection, session_id: str
    ) -> ChatCompaction | None:
        row = connection.execute(
            "SELECT * FROM chat_compactions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return ChatCompaction(
            session_id=session_id,
            covered_through_sequence=_row_int(row, "covered_through_sequence"),
            summary=_row_str(row, "summary"),
            estimated_tokens=_row_int(row, "estimated_tokens"),
            requested_model=_row_str(row, "requested_model"),
            actual_model=_row_str(row, "actual_model"),
            updated_at=_row_int(row, "updated_at"),
        )

    @staticmethod
    def _bump_session(
        connection: sqlite3.Connection,
        session_id: str,
        expected_revision: int,
        now: int,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE chat_sessions
            SET revision = revision + 1, updated_at = ?
            WHERE id = ? AND revision = ?
            """,
            (now, session_id, expected_revision),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            raise ChatConflictError("This chat changed in another tab. Refresh it.")


def _session_from_row(row: sqlite3.Row) -> ChatSession:
    return ChatSession(
        id=_row_str(row, "id"),
        title=_row_str(row, "title"),
        model=_row_str(row, "model"),
        reasoning=ChatReasoning(_row_str(row, "reasoning")),
        revision=_row_int(row, "revision"),
        created_at=_row_int(row, "created_at"),
        updated_at=_row_int(row, "updated_at"),
    )


def _session_summary_from_row(row: sqlite3.Row) -> ChatSessionSummary:
    return ChatSessionSummary(
        id=_row_str(row, "id"),
        title=_row_str(row, "title"),
        model=_row_str(row, "model"),
        reasoning=ChatReasoning(_row_str(row, "reasoning")),
        revision=_row_int(row, "revision"),
        preview=_row_str(row, "preview"),
        created_at=_row_int(row, "created_at"),
        updated_at=_row_int(row, "updated_at"),
    )


def _expect_revision(session: ChatSession, expected: int) -> None:
    if session.revision != expected:
        raise ChatConflictError("This chat changed in another tab. Refresh it.")


def _title_from_text(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= 64:
        return collapsed
    return f"{collapsed[:61].rstrip()}..."


def _row_str(row: sqlite3.Row, key: str) -> str:
    value: object = row[key]
    if not isinstance(value, str):
        raise ChatUnavailableError(f"Chat database field {key!r} is invalid.")
    return value


def _row_optional_str(row: sqlite3.Row, key: str) -> str | None:
    value: object = row[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ChatUnavailableError(f"Chat database field {key!r} is invalid.")
    return value


def _row_int(row: sqlite3.Row, key: str) -> int:
    value: object = row[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ChatUnavailableError(f"Chat database field {key!r} is invalid.")
    return value


def _row_optional_int(row: sqlite3.Row, key: str) -> int | None:
    value: object = row[key]
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ChatUnavailableError(f"Chat database field {key!r} is invalid.")
    return value


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _owner_only(path: Path, mode: int) -> None:
    if os.name != "nt" and path.exists():
        path.chmod(mode)


_SCHEMA_V1 = """
BEGIN EXCLUSIVE;
CREATE TABLE IF NOT EXISTS chat_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    system_prompt TEXT NOT NULL,
    last_model TEXT,
    last_reasoning TEXT NOT NULL CHECK (
        last_reasoning IN ('off', 'low', 'medium', 'high', 'xhigh', 'max')
    ),
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    title_search TEXT NOT NULL,
    auto_title_pending INTEGER NOT NULL CHECK (auto_title_pending IN (0, 1)),
    model TEXT NOT NULL,
    reasoning TEXT NOT NULL CHECK (
        reasoning IN ('off', 'low', 'medium', 'high', 'xhigh', 'max')
    ),
    revision INTEGER NOT NULL CHECK (revision > 0),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS chat_sessions_updated_idx
    ON chat_sessions(updated_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS chat_turns (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    operation_id TEXT NOT NULL,
    sequence INTEGER NOT NULL CHECK (sequence > 0),
    user_text TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE(session_id, sequence)
);

CREATE TABLE IF NOT EXISTS chat_generations (
    id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL REFERENCES chat_turns(id) ON DELETE CASCADE,
    visible INTEGER NOT NULL CHECK (visible IN (0, 1)),
    status TEXT NOT NULL CHECK (
        status IN ('running', 'completed', 'stopped', 'interrupted', 'failed')
    ),
    requested_model TEXT NOT NULL,
    actual_model TEXT,
    reasoning TEXT NOT NULL CHECK (
        reasoning IN ('off', 'low', 'medium', 'high', 'xhigh', 'max')
    ),
    effective_output_limit INTEGER NOT NULL CHECK (effective_output_limit > 0),
    stop_reason TEXT,
    error_code TEXT,
    error_message TEXT,
    started_at INTEGER NOT NULL,
    finished_at INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS chat_visible_generation_idx
    ON chat_generations(turn_id) WHERE visible = 1;

CREATE TABLE IF NOT EXISTS chat_generation_segments (
    generation_id TEXT NOT NULL REFERENCES chat_generations(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    kind TEXT NOT NULL CHECK (kind IN ('thinking', 'text')),
    text TEXT NOT NULL,
    PRIMARY KEY(generation_id, ordinal)
);

CREATE TABLE IF NOT EXISTS chat_compactions (
    session_id TEXT PRIMARY KEY REFERENCES chat_sessions(id) ON DELETE CASCADE,
    covered_through_sequence INTEGER NOT NULL CHECK (covered_through_sequence > 0),
    summary TEXT NOT NULL,
    estimated_tokens INTEGER NOT NULL CHECK (estimated_tokens > 0),
    requested_model TEXT NOT NULL,
    actual_model TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
"""
