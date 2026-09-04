import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from free_claude_code.application.terminal import (
    TerminalAttachmentEvent,
    TerminalAttachmentSnapshot,
    TerminalClientRole,
    TerminalDeletedEvent,
    TerminalOutputEvent,
    TerminalResetEvent,
    TerminalSession,
    TerminalStateEvent,
    TerminalStatus,
)
from tests.api.support import create_test_app

SESSION_ID = "a18a0609-2993-4694-a478-21050e6a5a44"


class StubAttachment:
    def __init__(self, session: TerminalSession) -> None:
        self._initial = TerminalAttachmentSnapshot(
            session, b"initial output\r\n", TerminalClientRole.CONTROLLER
        )
        self._events_queue: asyncio.Queue[TerminalAttachmentEvent | None] = (
            asyncio.Queue()
        )
        self.claims = 0
        self.writes: list[str] = []
        self.resizes: list[tuple[int, int]] = []
        self.closed = False

    @property
    def initial(self) -> TerminalAttachmentSnapshot:
        return self._initial

    def __aiter__(self) -> AsyncIterator[TerminalAttachmentEvent]:
        return self._events()

    async def _events(self) -> AsyncIterator[TerminalAttachmentEvent]:
        while True:
            event = await self._events_queue.get()
            if event is None:
                return
            yield event

    async def claim(self) -> None:
        self.claims += 1

    async def write(self, data: str) -> None:
        self.writes.append(data)

    async def resize(self, *, rows: int, columns: int) -> None:
        self.resizes.append((rows, columns))

    async def aclose(self) -> None:
        self.closed = True
        self._events_queue.put_nowait(None)

    def emit(self, event: TerminalAttachmentEvent) -> None:
        self._events_queue.put_nowait(event)


class StubTerminal:
    def __init__(self) -> None:
        self.availability_error: str | None = None
        self.session = TerminalSession(
            id=SESSION_ID,
            name="Terminal 1",
            status=TerminalStatus.RUNNING,
            created_at=1234,
            rows=24,
            columns=80,
            exit_code=None,
            error=None,
        )
        self.deleted = False
        self.attachment: StubAttachment | None = None
        self.attach_dimensions: tuple[int, int] | None = None

    async def create_session(self) -> TerminalSession:
        return self.session

    async def list_sessions(self) -> tuple[TerminalSession, ...]:
        return (self.session,)

    async def get_session(self, session_id: str) -> TerminalSession:
        assert session_id == SESSION_ID
        return self.session

    async def rename_session(self, session_id: str, name: str) -> TerminalSession:
        assert session_id == SESSION_ID
        self.session = replace(self.session, name=name)
        return self.session

    async def attach(
        self, session_id: str, *, rows: int, columns: int
    ) -> StubAttachment:
        assert session_id == SESSION_ID
        self.attach_dimensions = (rows, columns)
        self.attachment = StubAttachment(self.session)
        return self.attachment

    async def stop_session(self, session_id: str) -> TerminalSession:
        assert session_id == SESSION_ID
        self.session = replace(
            self.session, status=TerminalStatus.EXITED, exit_code=143
        )
        return self.session

    async def delete_session(self, session_id: str) -> None:
        assert session_id == SESSION_ID
        self.deleted = True


def _client(terminal: StubTerminal | None = None) -> TestClient:
    return TestClient(
        create_test_app(terminal=terminal),
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    )


def test_terminal_deep_links_serve_admin_shell() -> None:
    response = _client().get(f"/admin/terminal/{SESSION_ID}")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


def test_terminal_http_lifecycle_routes_and_availability() -> None:
    terminal = StubTerminal()
    client = _client(terminal)

    listed = client.get("/admin/api/terminal/sessions").json()
    assert listed["available"] is True
    assert listed["error"] is None
    assert listed["sessions"][0]["id"] == SESSION_ID
    assert client.post("/admin/api/terminal/sessions").status_code == 201
    assert (
        client.get(f"/admin/api/terminal/sessions/{SESSION_ID}").json()["name"]
        == "Terminal 1"
    )
    renamed = client.patch(
        f"/admin/api/terminal/sessions/{SESSION_ID}", json={"name": "Build"}
    )
    assert renamed.json()["name"] == "Build"
    stopped = client.post(f"/admin/api/terminal/sessions/{SESSION_ID}/stop")
    assert (stopped.json()["status"], stopped.json()["exit_code"]) == (
        "exited",
        143,
    )
    deleted = client.delete(f"/admin/api/terminal/sessions/{SESSION_ID}")
    assert deleted.json() == {"deleted": True}
    assert terminal.deleted


def test_terminal_list_exposes_scoped_engine_failure() -> None:
    terminal = StubTerminal()
    terminal.availability_error = "Rerun the FCC installer and restart FCC."

    payload = _client(terminal).get("/admin/api/terminal/sessions").json()

    assert payload == {
        "available": False,
        "error": "Rerun the FCC installer and restart FCC.",
        "sessions": [
            {
                "id": SESSION_ID,
                "name": "Terminal 1",
                "status": "running",
                "created_at": 1234,
                "rows": 24,
                "columns": 80,
                "exit_code": None,
                "error": None,
            }
        ],
    }


def test_terminal_websocket_routes_input_claim_and_resize_to_attachment() -> None:
    terminal = StubTerminal()
    client = _client(terminal)

    with client.websocket_connect(
        f"/admin/api/terminal/sessions/{SESSION_ID}/attach?rows=40&columns=120",
        headers={"host": "127.0.0.1", "origin": "http://127.0.0.1"},
    ) as websocket:
        attached = websocket.receive_json()
        assert attached == {
            "type": "attached",
            "session": {
                "id": SESSION_ID,
                "name": "Terminal 1",
                "status": "running",
                "created_at": 1234,
                "rows": 24,
                "columns": 80,
                "exit_code": None,
                "error": None,
            },
            "role": "controller",
        }
        assert websocket.receive_json() == {"type": "reset", "role": "controller"}
        assert websocket.receive_bytes() == b"initial output\r\n"

        websocket.send_json({"type": "claim"})
        websocket.send_json({"type": "input", "data": "echo hello\r"})
        websocket.send_json({"type": "resize", "rows": 41, "columns": 121})

    assert terminal.attach_dimensions == (40, 120)
    assert terminal.attachment is not None
    assert terminal.attachment.claims == 1
    assert terminal.attachment.writes == ["echo hello\r"]
    assert terminal.attachment.resizes == [(41, 121)]
    assert terminal.attachment.closed


def test_terminal_websocket_orders_reset_output_state_and_delete_events() -> None:
    terminal = StubTerminal()
    client = _client(terminal)

    with client.websocket_connect(
        f"/admin/api/terminal/sessions/{SESSION_ID}/attach",
        headers={"host": "127.0.0.1", "origin": "http://127.0.0.1"},
    ) as websocket:
        websocket.receive_json()
        websocket.receive_json()
        websocket.receive_bytes()
        assert terminal.attachment is not None
        terminal.attachment.emit(
            TerminalResetEvent(b"snapshot", TerminalClientRole.OBSERVER)
        )
        terminal.attachment.emit(TerminalOutputEvent(b"live"))
        terminal.attachment.emit(TerminalStateEvent(terminal.session))
        terminal.attachment.emit(TerminalDeletedEvent())

        assert websocket.receive_json() == {"type": "reset", "role": "observer"}
        assert websocket.receive_bytes() == b"snapshot"
        assert websocket.receive_bytes() == b"live"
        assert websocket.receive_json()["type"] == "state"
        assert websocket.receive_json() == {"type": "deleted"}


@pytest.mark.parametrize(
    ("client_host", "host", "origin"),
    (
        ("127.0.0.1", "127.0.0.1", "https://example.com"),
        ("127.0.0.1", "example.com", "http://127.0.0.1"),
        ("203.0.113.10", "127.0.0.1", "http://127.0.0.1"),
    ),
)
def test_terminal_websocket_rejects_nonlocal_connection_before_accept(
    client_host: str,
    host: str,
    origin: str,
) -> None:
    terminal = StubTerminal()
    client = TestClient(
        create_test_app(terminal=terminal),
        base_url="http://127.0.0.1",
        client=(client_host, 50000),
    )

    with (
        pytest.raises(WebSocketDisconnect) as raised,
        client.websocket_connect(
            f"/admin/api/terminal/sessions/{SESSION_ID}/attach",
            headers={"host": host, "origin": origin},
        ),
    ):
        pass

    assert raised.value.code == 1008
    assert terminal.attachment is None


@pytest.mark.parametrize(
    "message",
    (
        b"binary is invalid",
        '{"type":"resize","rows":0,"columns":80}',
        '{"type":"input","data":1}',
        '{"type":"unknown"}',
    ),
)
def test_terminal_websocket_reports_invalid_control_message(
    message: bytes | str,
) -> None:
    terminal = StubTerminal()
    client = _client(terminal)

    with client.websocket_connect(
        f"/admin/api/terminal/sessions/{SESSION_ID}/attach",
        headers={"host": "127.0.0.1", "origin": "http://127.0.0.1"},
    ) as websocket:
        websocket.receive_json()
        websocket.receive_json()
        websocket.receive_bytes()
        if isinstance(message, bytes):
            websocket.send_bytes(message)
        else:
            websocket.send_text(message)
        assert websocket.receive_json() == {
            "type": "error",
            "message": "Invalid terminal control message.",
        }

    assert terminal.attachment is not None and terminal.attachment.closed
