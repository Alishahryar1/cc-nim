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
    name: str
    harness: BrowserSessionHarness
    native_id: str
    started_once: bool = False


@dataclass(slots=True)
class ProjectRecord:
    project_id: str
    path: str
    sessions: list[SessionRecord] = field(default_factory=list)


@dataclass(slots=True)
class SessionCatalog:
    projects: list[ProjectRecord] = field(default_factory=list)


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
    return {
        "version": SCHEMA_VERSION,
        "projects": [
            {
                "id": project.project_id,
                "path": project.path,
                "sessions": [
                    {
                        "id": session.session_id,
                        "name": session.name,
                        "harness": session.harness.value,
                        "native_id": session.native_id,
                        "started_once": session.started_once,
                    }
                    for session in project.sessions
                ],
            }
            for project in catalog.projects
        ],
    }


def _decode_catalog(payload: object) -> SessionCatalog:
    root = _object(payload, "root")
    if root.get("version") != SCHEMA_VERSION:
        raise SessionStoreError("Unsupported browser Sessions schema version")
    projects_value = root.get("projects")
    if not isinstance(projects_value, list):
        raise SessionStoreError("projects must be a list")

    projects: list[ProjectRecord] = []
    project_ids: set[str] = set()
    session_ids: set[str] = set()
    for project_value in projects_value:
        project = _object(project_value, "project")
        project_id = _string(project.get("id"), "project.id")
        path = _string(project.get("path"), "project.path")
        if project_id in project_ids:
            raise SessionStoreError("Duplicate project id")
        project_ids.add(project_id)
        sessions_value = project.get("sessions")
        if not isinstance(sessions_value, list):
            raise SessionStoreError("project.sessions must be a list")
        sessions: list[SessionRecord] = []
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
            sessions.append(
                SessionRecord(
                    session_id=session_id,
                    name=_string(session.get("name"), "session.name"),
                    harness=harness,
                    native_id=_string(session.get("native_id"), "session.native_id"),
                    started_once=started_once,
                )
            )
        projects.append(
            ProjectRecord(project_id=project_id, path=path, sessions=sessions)
        )
    return SessionCatalog(projects=projects)


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SessionStoreError(f"{name} must be an object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SessionStoreError(f"{name} must be a non-empty string")
    return value
