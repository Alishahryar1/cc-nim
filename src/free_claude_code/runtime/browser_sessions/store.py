"""Small atomic JSON store for browser-session metadata."""

import json
import os
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from free_claude_code.application.browser_sessions import BrowserSessionHarness

SCHEMA_VERSION = 1


@dataclass(slots=True)
class SessionRecord:
    session_id: str
    path: str
    name: str
    harness: BrowserSessionHarness
    native_id: str | None
    started_once: bool = False


@dataclass(slots=True)
class SessionCatalog:
    sessions: list[SessionRecord] = field(default_factory=list)


class SessionStoreError(Exception):
    """Persisted session metadata is unavailable or invalid."""


class BrowserSessionStore:
    """Load and atomically replace one versioned browser-session catalog."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> SessionCatalog:
        if not self.path.exists():
            return SessionCatalog()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return _decode_catalog(payload)
        except (OSError, UnicodeError, json.JSONDecodeError, SessionStoreError) as exc:
            raise SessionStoreError(
                "Browser Sessions metadata could not be read. "
                "The original file was preserved."
            ) from exc

    def save(self, catalog: SessionCatalog) -> None:
        parent = self.path.parent
        temporary = parent / f".{self.path.name}.{uuid4().hex}.tmp"
        try:
            parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    _encode_catalog(catalog), handle, indent=2, ensure_ascii=False
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise SessionStoreError(
                "Browser Sessions metadata could not be saved."
            ) from exc


def _encode_catalog(catalog: SessionCatalog) -> dict[str, object]:
    for session in catalog.sessions:
        _validate_identity_state(
            session.harness,
            session.native_id,
            session.started_once,
        )
    return {
        "version": SCHEMA_VERSION,
        "sessions": [
            {
                "id": session.session_id,
                "path": session.path,
                "name": session.name,
                "harness": session.harness.value,
                "native_id": session.native_id,
                "started_once": session.started_once,
            }
            for session in catalog.sessions
        ],
    }


def _decode_catalog(payload: object) -> SessionCatalog:
    root = _object(payload, "root")
    if root.get("version") != SCHEMA_VERSION:
        raise SessionStoreError("Unsupported browser Sessions schema version")
    sessions_value = root.get("sessions")
    if not isinstance(sessions_value, list):
        raise SessionStoreError("sessions must be a list")

    sessions: list[SessionRecord] = []
    session_ids: set[str] = set()
    for session_value in sessions_value:
        session = _object(session_value, "session")
        session_id = _string(session.get("id"), "session.id")
        if session_id in session_ids:
            raise SessionStoreError("Duplicate session id")
        session_ids.add(session_id)
        try:
            harness = BrowserSessionHarness(
                _string(session.get("harness"), "session.harness")
            )
        except ValueError as exc:
            raise SessionStoreError("Unknown session harness") from exc
        started_once = session.get("started_once")
        if not isinstance(started_once, bool):
            raise SessionStoreError("session.started_once must be a boolean")
        native_value = session.get("native_id")
        native_id = (
            None if native_value is None else _string(native_value, "session.native_id")
        )
        _validate_identity_state(harness, native_id, started_once)
        sessions.append(
            SessionRecord(
                session_id=session_id,
                path=_string(session.get("path"), "session.path"),
                name=_string(session.get("name"), "session.name"),
                harness=harness,
                native_id=native_id,
                started_once=started_once,
            )
        )
    return SessionCatalog(sessions=sessions)


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SessionStoreError(f"{name} must be an object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SessionStoreError(f"{name} must be a non-empty string")
    return value


def _validate_identity_state(
    harness: BrowserSessionHarness,
    native_id: str | None,
    started_once: bool,
) -> None:
    if harness is BrowserSessionHarness.CODEX:
        if (native_id is None) != (not started_once):
            raise SessionStoreError("Invalid Codex session identity state")
        return
    if native_id is None:
        raise SessionStoreError("Native session identity is required")
