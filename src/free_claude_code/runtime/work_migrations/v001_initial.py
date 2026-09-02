"""Create the first Work registry and operation journal schema."""

import sqlite3


def apply(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE work_sessions (
            thread_id TEXT PRIMARY KEY,
            cwd TEXT NOT NULL,
            cwd_key TEXT NOT NULL,
            model TEXT,
            reasoning_effort TEXT,
            collaboration_mode TEXT,
            permission_profile TEXT,
            revision INTEGER NOT NULL CHECK (revision > 0),
            registered_at_ms INTEGER NOT NULL CHECK (registered_at_ms >= 0)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX work_sessions_project_recency
        ON work_sessions (cwd_key, registered_at_ms DESC)
        """
    )
    connection.execute(
        """
        CREATE TABLE work_operations (
            operation_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('create', 'send', 'stop', 'delete')),
            session_id TEXT,
            intent_digest TEXT NOT NULL,
            state TEXT NOT NULL CHECK (
                state IN ('reserved', 'submitted', 'completed', 'interrupted', 'failed')
            ),
            result_thread_id TEXT,
            result_turn_id TEXT,
            error_code TEXT,
            error_message TEXT,
            created_at_ms INTEGER NOT NULL CHECK (created_at_ms >= 0),
            updated_at_ms INTEGER NOT NULL CHECK (updated_at_ms >= 0)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX work_operations_session_recency
        ON work_operations (session_id, created_at_ms DESC)
        """
    )
