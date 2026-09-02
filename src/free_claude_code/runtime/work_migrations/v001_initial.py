"""Create the Work registry and durable operation queue."""

import sqlite3


def apply(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE work_sessions (
            thread_id TEXT PRIMARY KEY CHECK (length(thread_id) > 0),
            cwd TEXT NOT NULL CHECK (length(cwd) > 0),
            cwd_key TEXT NOT NULL CHECK (length(cwd_key) > 0),
            model TEXT NOT NULL CHECK (length(model) > 0),
            reasoning_effort TEXT,
            revision INTEGER NOT NULL CHECK (revision > 0),
            registered_at_ms INTEGER NOT NULL CHECK (registered_at_ms >= 0)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX work_sessions_registration
        ON work_sessions (registered_at_ms DESC, thread_id)
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
            operation_id TEXT PRIMARY KEY CHECK (length(operation_id) = 36),
            kind TEXT NOT NULL CHECK (
                kind IN ('create', 'send', 'stop', 'delete', 'respond')
            ),
            session_id TEXT,
            interaction_id TEXT,
            intent_digest TEXT NOT NULL CHECK (length(intent_digest) = 64),
            payload_json TEXT,
            state TEXT NOT NULL CHECK (
                state IN (
                    'accepted', 'executing', 'unknown',
                    'succeeded', 'failed', 'abandoned'
                )
            ),
            expected_revision INTEGER CHECK (expected_revision > 0),
            captured_model TEXT,
            captured_reasoning_effort TEXT,
            native_thread_id TEXT,
            native_turn_id TEXT,
            native_connection_id TEXT,
            error_code TEXT,
            error_message TEXT,
            created_at_ms INTEGER NOT NULL CHECK (created_at_ms >= 0),
            updated_at_ms INTEGER NOT NULL CHECK (updated_at_ms >= created_at_ms),
            CHECK (
                (state IN ('accepted', 'executing') AND payload_json IS NOT NULL)
                OR
                (state IN ('unknown', 'succeeded', 'failed', 'abandoned')
                    AND payload_json IS NULL)
            ),
            CHECK (kind = 'create' OR session_id IS NOT NULL),
            CHECK (kind = 'respond' OR interaction_id IS NULL),
            CHECK (kind != 'respond' OR interaction_id IS NOT NULL)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX work_operations_dispatch
        ON work_operations (state, created_at_ms, operation_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX work_operations_session_state
        ON work_operations (session_id, state, kind, created_at_ms)
        """
    )
    connection.execute(
        """
        CREATE INDEX work_operations_interaction
        ON work_operations (interaction_id, kind)
        """
    )
