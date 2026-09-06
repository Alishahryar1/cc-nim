"""Private SQLite state for FCC coding conversations."""

import asyncio
import os
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path

import anyio.to_thread

from free_claude_code.application.code_sessions.models import (
    ACTIVE_RUN_STATUSES,
    CodeConflictError,
    CodeItem,
    CodeNotFoundError,
    CodePage,
    CodePrompt,
    CodeRun,
    CodeSession,
    CodeUnavailableError,
    now_ms,
)
from free_claude_code.core.interprocess_lock import InterprocessFileLock


class SQLiteCodeStore:
    def __init__(self, database_path: Path, lock_path: Path) -> None:
        self._path = database_path
        self._lock = InterprocessFileLock(lock_path)
        self._started = False
        self._lifecycle = asyncio.Lock()

    async def start(self) -> None:
        async with self._lifecycle:
            if self._started:
                return
            initialization = asyncio.create_task(
                anyio.to_thread.run_sync(self._initialize)
            )
            try:
                await asyncio.shield(initialization)
            except BaseException:
                # Direct asyncio cancellation can outlive the worker thread.
                # Drain it before releasing a lock that it may still acquire.
                await asyncio.gather(initialization, return_exceptions=True)
                self._lock.release()
                raise
            self._started = True

    def _initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._lock.acquire():
            raise CodeUnavailableError(
                "Code sessions is already owned by another FCC server."
            )
        if os.name != "nt":
            self._path.parent.chmod(0o700)
        self._execute(self._migrate)
        if os.name != "nt":
            self._path.chmod(0o600)
            for suffix in ("-wal", "-shm"):
                sidecar = Path(f"{self._path}{suffix}")
                if sidecar.exists():
                    sidecar.chmod(0o600)

    async def close(self) -> None:
        async with self._lifecycle:
            self._started = False
            await anyio.to_thread.run_sync(self._lock.release)

    def _execute[T](self, operation: Callable[[sqlite3.Connection], T]) -> T:
        try:
            with closing(sqlite3.connect(self._path, timeout=10)) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
                with connection:
                    return operation(connection)
        except sqlite3.Error as exc:
            raise CodeUnavailableError("Code session storage is unavailable.") from exc

    async def _run[T](self, operation: Callable[[sqlite3.Connection], T]) -> T:
        if not self._started:
            raise CodeUnavailableError("Code session storage is closed.")
        return await anyio.to_thread.run_sync(self._execute, operation)

    def _migrate(self, connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(_SCHEMA)
        connection.execute("BEGIN IMMEDIATE")
        for row in connection.execute("SELECT data FROM code_runs").fetchall():
            run = CodeRun.model_validate_json(row["data"])
            if run.status in ACTIVE_RUN_STATUSES:
                repaired = run.model_copy(
                    update={
                        "status": "interrupted",
                        "finished_at": now_ms(),
                        "error": "FCC restarted before this turn finished. Its input was not resent.",
                    }
                )
                connection.execute(
                    "UPDATE code_runs SET data = ? WHERE id = ?",
                    (repaired.model_dump_json(), run.id),
                )
        for row in connection.execute("SELECT data FROM code_prompts").fetchall():
            prompt = CodePrompt.model_validate_json(row["data"])
            if prompt.status in {"pending", "answering"}:
                repaired = prompt.model_copy(update={"status": "expired"})
                connection.execute(
                    "UPDATE code_prompts SET data = ? WHERE id = ?",
                    (repaired.model_dump_json(), prompt.id),
                )

    async def create(self, session: CodeSession) -> CodeSession:
        def operation(connection: sqlite3.Connection) -> CodeSession:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM code_deleted WHERE id = ?", (session.id,)
            ).fetchone():
                raise CodeConflictError(
                    "This session was deleted. Create a new session."
                )
            row = connection.execute(
                "SELECT data FROM code_sessions WHERE id = ?", (session.id,)
            ).fetchone()
            if row:
                existing = CodeSession.model_validate_json(row["data"])
                if existing.cwd != session.cwd or existing.harness != session.harness:
                    raise CodeConflictError(
                        "This session ID was already used for another folder."
                    )
                return existing
            connection.execute(
                "INSERT INTO code_sessions(id, updated_at, data) VALUES (?, ?, ?)",
                (session.id, session.updated_at, session.model_dump_json()),
            )
            return session

        return await self._run(operation)

    async def get_session(self, session_id: str) -> CodeSession:
        def operation(connection: sqlite3.Connection) -> CodeSession:
            row = connection.execute(
                "SELECT data FROM code_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise CodeNotFoundError("Code session not found.")
            return CodeSession.model_validate_json(row["data"])

        return await self._run(operation)

    async def list_sessions(
        self, cursor: tuple[int, str] | None, limit: int
    ) -> CodePage:
        def operation(connection: sqlite3.Connection) -> CodePage:
            rows = connection.execute(
                "SELECT data FROM code_sessions "
                + ("WHERE (updated_at, id) < (?, ?) " if cursor else "")
                + "ORDER BY updated_at DESC, id DESC LIMIT ?",
                (*cursor, limit + 1) if cursor else (limit + 1,),
            ).fetchall()
            sessions = tuple(
                CodeSession.model_validate_json(row["data"]) for row in rows[:limit]
            )
            last = sessions[-1] if sessions and len(rows) > limit else None
            return CodePage(sessions, (last.updated_at, last.id) if last else None)

        return await self._run(operation)

    async def pending_deletions(self) -> tuple[CodeSession, ...]:
        def operation(connection: sqlite3.Connection) -> tuple[CodeSession, ...]:
            sessions = (
                CodeSession.model_validate_json(row["data"])
                for row in connection.execute("SELECT data FROM code_sessions")
            )
            return tuple(session for session in sessions if session.status != "ready")

        return await self._run(operation)

    async def is_deleted(self, session_id: str) -> bool:
        return await self._run(
            lambda connection: (
                connection.execute(
                    "SELECT 1 FROM code_deleted WHERE id = ?", (session_id,)
                ).fetchone()
                is not None
            )
        )

    async def get_run(self, run_id: str) -> CodeRun | None:
        def operation(connection: sqlite3.Connection) -> CodeRun | None:
            row = connection.execute(
                "SELECT data FROM code_runs WHERE id = ?", (run_id,)
            ).fetchone()
            return CodeRun.model_validate_json(row["data"]) if row else None

        return await self._run(operation)

    async def latest_run(self, session_id: str) -> CodeRun | None:
        def operation(connection: sqlite3.Connection) -> CodeRun | None:
            row = connection.execute(
                "SELECT data FROM code_runs WHERE session_id = ? ORDER BY rowid DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            return CodeRun.model_validate_json(row["data"]) if row else None

        return await self._run(operation)

    async def items(
        self, session_id: str, before: int | None, limit: int | None
    ) -> tuple[CodeItem, ...]:
        def operation(connection: sqlite3.Connection) -> tuple[CodeItem, ...]:
            parameters: list[object] = [session_id]
            query = "SELECT data FROM code_items WHERE session_id = ?"
            if before is not None:
                query += " AND sequence < ?"
                parameters.append(before)
            query += " ORDER BY sequence DESC"
            if limit is not None:
                query += " LIMIT ?"
                parameters.append(limit)
            rows = connection.execute(query, parameters).fetchall()
            return tuple(
                CodeItem.model_validate_json(row["data"]) for row in reversed(rows)
            )

        return await self._run(operation)

    async def prompts(self, session_id: str) -> tuple[CodePrompt, ...]:
        def operation(connection: sqlite3.Connection) -> tuple[CodePrompt, ...]:
            return tuple(
                CodePrompt.model_validate_json(row["data"])
                for row in connection.execute(
                    "SELECT data FROM code_prompts WHERE session_id = ? ORDER BY rowid",
                    (session_id,),
                )
            )

        return await self._run(operation)

    async def save(
        self,
        session: CodeSession,
        *,
        run: CodeRun | None = None,
        items: Sequence[CodeItem] = (),
        prompts: Sequence[CodePrompt] = (),
    ) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE code_sessions SET updated_at = ?, data = ? WHERE id = ?",
                (session.updated_at, session.model_dump_json(), session.id),
            )
            if updated.rowcount != 1:
                raise CodeNotFoundError("Code session no longer exists.")
            if run is not None:
                saved = connection.execute(
                    "INSERT INTO code_runs(id, session_id, data) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET data = excluded.data WHERE code_runs.session_id = excluded.session_id",
                    (run.id, session.id, run.model_dump_json()),
                )
                if saved.rowcount != 1:
                    raise CodeConflictError(
                        "This Send ID was already used in another session."
                    )
            connection.executemany(
                "INSERT INTO code_items(id, session_id, sequence, data) VALUES (?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                (
                    (item.id, session.id, item.sequence, item.model_dump_json())
                    for item in items
                ),
            )
            connection.executemany(
                "INSERT INTO code_prompts(id, session_id, data) VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET data = excluded.data",
                (
                    (prompt.id, session.id, prompt.model_dump_json())
                    for prompt in prompts
                ),
            )

        await self._run(operation)

    async def delete(self, session_id: str) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT OR IGNORE INTO code_deleted(id) VALUES (?)", (session_id,)
            )
            connection.execute("DELETE FROM code_sessions WHERE id = ?", (session_id,))

        await self._run(operation)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS code_sessions(id TEXT PRIMARY KEY, updated_at INTEGER NOT NULL, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS code_sessions_recent ON code_sessions(updated_at DESC, id DESC);
CREATE TABLE IF NOT EXISTS code_runs(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES code_sessions(id) ON DELETE CASCADE, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS code_runs_session ON code_runs(session_id);
CREATE TABLE IF NOT EXISTS code_items(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES code_sessions(id) ON DELETE CASCADE, sequence INTEGER NOT NULL, data TEXT NOT NULL, UNIQUE(session_id, sequence));
CREATE TABLE IF NOT EXISTS code_prompts(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES code_sessions(id) ON DELETE CASCADE, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS code_prompts_session ON code_prompts(session_id);
CREATE TABLE IF NOT EXISTS code_deleted(id TEXT PRIMARY KEY);
"""
