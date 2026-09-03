import asyncio
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from free_claude_code.application.terminal import (
    TerminalAttachmentSnapshot,
    TerminalSession,
    TerminalStatus,
)
from tests.api.support import create_test_app

SESSION_ID = "a18a0609-2993-4694-a478-21050e6a5a44"


class StubAttachment:
    def __init__(self, session: TerminalSession) -> None:
        self._initial = TerminalAttachmentSnapshot(session, b"initial output\r\n")
        self.closed = False

    @property
    def initial(self) -> TerminalAttachmentSnapshot:
        return self._initial

    def __aiter__(self):
        return self._events()

    async def _events(self):
        await asyncio.Event().wait()
        if False:
            yield

    async def aclose(self) -> None:
        self.closed = True


class StubTerminal:
    def __init__(self) -> None:
        self.session = TerminalSession(
            id=SESSION_ID,
            name="Terminal 1",
            status=TerminalStatus.RUNNING,
            created_at=1234,
            rows=24,
            columns=80,
            exit_code=None,
            error=None,
            history_truncated=False,
        )
        self.writes: list[tuple[str, bytes]] = []
        self.resizes: list[tuple[str, int, int]] = []
        self.deleted = False
        self.attachment: StubAttachment | None = None

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

    async def attach(self, session_id: str) -> StubAttachment:
        assert session_id == SESSION_ID
        self.attachment = StubAttachment(self.session)
        return self.attachment

    async def write(self, session_id: str, data: bytes) -> None:
        self.writes.append((session_id, data))

    async def resize(
        self, session_id: str, *, rows: int, columns: int
    ) -> TerminalSession:
        self.resizes.append((session_id, rows, columns))
        self.session = replace(self.session, rows=rows, columns=columns)
        return self.session

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


def test_terminal_http_lifecycle_routes() -> None:
    terminal = StubTerminal()
    client = _client(terminal)

    assert (
        client.get("/admin/api/terminal/sessions").json()["sessions"][0]["id"]
        == SESSION_ID
    )
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


def test_terminal_websocket_attaches_sends_input_and_resizes() -> None:
    terminal = StubTerminal()
    client = _client(terminal)

    with client.websocket_connect(
        f"/admin/api/terminal/sessions/{SESSION_ID}/attach",
        headers={"host": "127.0.0.1", "origin": "http://127.0.0.1"},
    ) as websocket:
        attached = websocket.receive_json()
        assert attached["type"] == "attached"
        assert attached["session"]["id"] == SESSION_ID
        assert websocket.receive_bytes() == b"initial output\r\n"

        websocket.send_bytes(b"echo hello\r")
        websocket.send_json({"type": "resize", "rows": 40, "columns": 120})

    assert terminal.writes == [(SESSION_ID, b"echo hello\r")]
    assert terminal.resizes == [(SESSION_ID, 40, 120)]
    assert terminal.attachment is not None and terminal.attachment.closed


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


def test_terminal_websocket_reports_invalid_control_message() -> None:
    terminal = StubTerminal()
    client = _client(terminal)

    with client.websocket_connect(
        f"/admin/api/terminal/sessions/{SESSION_ID}/attach",
        headers={"host": "127.0.0.1", "origin": "http://127.0.0.1"},
    ) as websocket:
        assert websocket.receive_json()["type"] == "attached"
        assert websocket.receive_bytes() == b"initial output\r\n"
        websocket.send_text('{"type":"resize","rows":0,"columns":80}')
        assert websocket.receive_json() == {
            "type": "error",
            "message": "Invalid terminal control message.",
        }

    assert terminal.attachment is not None and terminal.attachment.closed
