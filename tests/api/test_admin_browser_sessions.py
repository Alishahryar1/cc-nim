import asyncio

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from free_claude_code.api.app import create_app
from free_claude_code.api.ports import ApiServices
from free_claude_code.application.browser_sessions import (
    BrowserSessionConflictError,
    BrowserSessionHarness,
    BrowserSessionNotFoundError,
    BrowserSessionSnapshot,
    BrowserSessionState,
    BrowserSessionUnavailableError,
    BrowserSessionValidationError,
    BrowserSessionView,
    HarnessAvailability,
    TerminalEvent,
)
from tests.api.support import create_test_app


class StubTerminal:
    def __init__(self) -> None:
        self.events: asyncio.Queue[TerminalEvent] = asyncio.Queue()
        self.events.put_nowait(
            TerminalEvent(kind="ready", state=BrowserSessionState.RUNNING)
        )
        self.events.put_nowait(TerminalEvent(kind="output", data=b"terminal output"))
        self.writes: list[str] = []
        self.resizes: list[tuple[int, int]] = []
        self.closed = False

    async def receive(self) -> TerminalEvent:
        return await self.events.get()

    async def write(self, data: str) -> None:
        self.writes.append(data)

    async def resize(self, columns: int, rows: int) -> None:
        self.resizes.append((columns, rows))
        self.events.put_nowait(TerminalEvent(kind="error", detail="test complete"))

    async def close(self) -> None:
        self.closed = True


class StubSessions:
    def __init__(self) -> None:
        self.terminal = StubTerminal()
        self.created: list[tuple[str, BrowserSessionHarness, str | None]] = []
        self.session = BrowserSessionView(
            session_id="ses_one",
            name="Review",
            harness=BrowserSessionHarness.CODEX,
            state=BrowserSessionState.RUNNING,
            project_name="project",
            path="C:\\project",
        )

    async def snapshot(self) -> BrowserSessionSnapshot:
        return BrowserSessionSnapshot(
            available=True,
            sessions=(self.session,),
            harnesses=(
                HarnessAvailability(BrowserSessionHarness.CLAUDE, True),
                HarnessAvailability(BrowserSessionHarness.CODEX, True),
                HarnessAvailability(BrowserSessionHarness.PI, True),
            ),
        )

    async def create_session(self, path, harness, name=None):
        if path == "bad":
            raise BrowserSessionValidationError("Project path is invalid.")
        self.created.append((path, harness, name))
        return self.session

    async def rename_session(self, session_id, name):
        return self.session

    async def start_session(self, session_id):
        return self.session

    async def stop_session(self, session_id):
        return self.session

    async def delete_session(self, session_id):
        return None

    async def attach_terminal(self, session_id):
        return self.terminal


def session_app(stub: StubSessions):
    base = create_test_app()
    services = base.state.services
    return create_app(
        ApiServices(
            requests=services.requests,
            admin=services.admin,
            tasks=services.tasks,
            sessions=stub,
        )
    )


def test_sessions_catalog_is_local_only_no_store_and_redacts_native_ids():
    app = session_app(StubSessions())
    local = TestClient(app, client=("127.0.0.1", 50000))
    remote = TestClient(app, client=("203.0.113.8", 50000))

    response = local.get("/admin/api/sessions")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["sessions"][0] == {
        "id": "ses_one",
        "name": "Review",
        "harness": "codex",
        "state": "running",
        "project_name": "project",
        "path": "C:\\project",
        "detail": None,
    }
    assert "native" not in response.text
    assert remote.get("/admin/api/sessions").status_code == 403


def test_sessions_domain_validation_maps_to_customer_400():
    client = TestClient(session_app(StubSessions()), client=("127.0.0.1", 50000))

    response = client.post(
        "/admin/api/sessions",
        json={"path": "bad", "harness": "codex", "name": None},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Project path is invalid."}


def test_session_creation_accepts_an_optional_name_without_project_registration():
    stub = StubSessions()
    client = TestClient(session_app(stub), client=("127.0.0.1", 50000))

    response = client.post(
        "/admin/api/sessions",
        json={"path": "C:\\project", "harness": "pi"},
    )

    assert response.status_code == 201
    assert stub.created == [
        ("C:\\project", BrowserSessionHarness.PI, None),
    ]


def test_unreleased_project_routes_are_removed():
    client = TestClient(session_app(StubSessions()), client=("127.0.0.1", 50000))

    assert (
        client.post("/admin/api/projects", json={"path": "C:\\project"}).status_code
        == 404
    )
    assert client.delete("/admin/api/projects/prj_one").status_code == 404


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (BrowserSessionNotFoundError("missing"), 404),
        (BrowserSessionConflictError("busy"), 409),
        (BrowserSessionUnavailableError("unavailable"), 503),
    ],
)
def test_sessions_domain_failures_keep_their_http_meaning(error, status_code):
    class FailingSessions(StubSessions):
        async def start_session(self, session_id):
            raise error

    client = TestClient(session_app(FailingSessions()), client=("127.0.0.1", 50000))

    response = client.post("/admin/api/sessions/ses_one/start")

    assert response.status_code == status_code
    assert response.json() == {"detail": str(error)}


def test_terminal_websocket_streams_binary_output_and_accepts_controls():
    stub = StubSessions()
    client = TestClient(session_app(stub), client=("127.0.0.1", 50000))

    with client.websocket_connect("/admin/api/sessions/ses_one/terminal") as socket:
        assert socket.receive_json() == {
            "type": "ready",
            "state": "running",
            "detail": None,
        }
        assert socket.receive_bytes() == b"terminal output"
        socket.send_json({"type": "input", "data": "hello\r"})
        socket.send_json({"type": "resize", "columns": 100, "rows": 30})
        assert socket.receive_json()["type"] == "error"

    assert stub.terminal.writes == ["hello\r"]
    assert stub.terminal.resizes == [(100, 30)]
    assert stub.terminal.closed


def test_terminal_websocket_rejects_nonlocal_browser_origin():
    stub = StubSessions()
    client = TestClient(session_app(stub), client=("127.0.0.1", 50000))

    with (
        pytest.raises(WebSocketDisconnect) as raised,
        client.websocket_connect(
            "/admin/api/sessions/ses_one/terminal",
            headers={"origin": "https://example.com"},
        ) as socket,
    ):
        socket.receive_text()

    assert raised.value.code == 1008
    assert stub.terminal.closed is False


def test_sessions_is_the_first_default_admin_view_and_assets_are_local():
    client = TestClient(session_app(StubSessions()), client=("127.0.0.1", 50000))

    page = client.get("/admin")

    assert page.status_code == 200
    assert page.text.index('data-view="sessions"') < page.text.index(
        'data-view="providers"'
    )
    assert 'id="view-sessions" class="admin-view active"' in page.text
    assert 'id="newSessionPath"' in page.text
    assert 'id="sessionTabs"' not in page.text
    assert 'id="projectDialog"' not in page.text
    for path in (
        "/admin/assets/admin-api.js",
        "/admin/assets/sessions.js",
        "/admin/assets/vendor/xterm.js",
        "/admin/assets/vendor/xterm.css",
        "/admin/assets/vendor/addon-fit.js",
    ):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
