from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from free_claude_code.application.event_feed import PublishedEvent
from free_claude_code.application.work import (
    WorkBootstrap,
    WorkCompatibilityError,
    WorkConflictError,
    WorkInteraction,
    WorkInteractionKind,
    WorkNotFoundError,
    WorkOperationAcknowledgement,
    WorkOperationKind,
    WorkOperationState,
    WorkSessionDetail,
    WorkSessionPage,
    WorkSessionRecord,
    WorkSessionSettings,
    WorkSessionSummary,
    WorkStatus,
    WorkTimelineItem,
    WorkTurnPage,
    WorkUnavailableError,
    WorkValidationError,
)
from free_claude_code.core.json_types import JsonObject, JsonValue
from tests.api.support import create_test_app

THREAD_ID = "thread-1"
OPERATION_ID = "7cd43d62-c1aa-42f8-9963-6c0811c0dfaf"


class StubSubscription:
    cursor = 4

    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self):
        return self._events()

    async def _events(self):
        yield PublishedEvent(
            event="session.status",
            id=5,
            data={"thread_id": THREAD_ID, "status": "working"},
        )

    async def aclose(self) -> None:
        self.closed = True


class StubWork:
    def __init__(self) -> None:
        self.settings = WorkSessionSettings(
            model="model-1",
            reasoning_effort="medium",
        )
        self.record = WorkSessionRecord(
            thread_id=THREAD_ID,
            cwd="C:\\example",
            cwd_key="c:\\example",
            settings=self.settings,
            revision=1,
            registered_at_ms=1,
        )
        self.summary = WorkSessionSummary(
            thread_id=THREAD_ID,
            cwd=self.record.cwd,
            title="Example",
            preview="Inspect the repository",
            status=WorkStatus.READY,
            revision=1,
            registered_at_ms=1,
            updated_at_ms=2,
            project_available=True,
            session_available=True,
        )
        self.timeline = WorkTimelineItem(
            thread_id=THREAD_ID,
            turn_id="turn-1",
            item_id="message-1",
            kind="agentMessage",
            status="completed",
            text="**safe** <script>unsafe()</script>",
            payload={"id": "message-1", "type": "agentMessage"},
        )
        self.subscription: StubSubscription | None = None
        self.last_call: tuple[str, object] | None = None
        self.error: Exception | None = None

    def _raise(self) -> None:
        if self.error is not None:
            raise self.error

    async def bootstrap(self) -> WorkBootstrap:
        self._raise()
        return WorkBootstrap(
            available=True,
            reason=None,
            codex_version="codex-cli 0.152.0",
            recent_projects=(self.record.cwd,),
            unresolved_creates=(),
            event_generation="generation-1",
            event_cursor=4,
        )

    async def subscribe(self) -> StubSubscription:
        self.subscription = StubSubscription()
        return self.subscription

    async def list_sessions(
        self,
        *,
        query: str,
        cursor: tuple[int, str] | None,
        limit: int,
    ) -> WorkSessionPage:
        self._raise()
        self.last_call = ("list", (query, cursor, limit))
        return WorkSessionPage((self.summary,), (1, THREAD_ID))

    async def create_session(
        self, *, cwd: str, operation_id: str
    ) -> WorkOperationAcknowledgement:
        self._raise()
        self.last_call = ("create", (cwd, operation_id))
        return self._operation(WorkOperationKind.CREATE)

    async def get_detail(self, thread_id: str) -> WorkSessionDetail:
        self._raise()
        self.last_call = ("detail", thread_id)
        return WorkSessionDetail(
            summary=self.summary,
            settings=self.settings,
            controls={"models": [{"id": "model-1"}]},
            turns=WorkTurnPage((self.timeline,), "older-turns"),
            live_items=(),
            interactions=(
                WorkInteraction(
                    interaction_id="interaction-1",
                    thread_id=THREAD_ID,
                    turn_id="turn-1",
                    kind=WorkInteractionKind.COMMAND_APPROVAL,
                    title="Command approval",
                    payload={"command": ["git", "status"]},
                ),
            ),
            event_cursor=4,
        )

    async def update_settings(
        self,
        thread_id: str,
        *,
        expected_revision: int,
        updates: JsonObject,
    ) -> WorkSessionRecord:
        self._raise()
        self.last_call = ("settings", (thread_id, expected_revision, updates))
        self.record = replace(self.record, revision=2)
        return self.record

    async def send(
        self,
        thread_id: str,
        *,
        expected_revision: int,
        operation_id: str,
        text: str,
    ) -> WorkOperationAcknowledgement:
        self._raise()
        self.last_call = (
            "send",
            (thread_id, expected_revision, operation_id, text),
        )
        return self._operation(WorkOperationKind.SEND)

    async def stop(
        self, thread_id: str, *, operation_id: str
    ) -> WorkOperationAcknowledgement:
        self._raise()
        self.last_call = ("stop", (thread_id, operation_id))
        return self._operation(WorkOperationKind.STOP)

    async def delete(
        self, thread_id: str, *, operation_id: str
    ) -> WorkOperationAcknowledgement:
        self._raise()
        self.last_call = ("delete", (thread_id, operation_id))
        return self._operation(WorkOperationKind.DELETE)

    async def remove_missing(self, thread_id: str) -> None:
        self._raise()
        self.last_call = ("remove", thread_id)

    async def respond(
        self,
        thread_id: str,
        interaction_id: str,
        *,
        operation_id: str,
        value: JsonValue,
    ) -> WorkOperationAcknowledgement:
        self._raise()
        self.last_call = (
            "respond",
            (thread_id, interaction_id, operation_id, value),
        )
        return self._operation(WorkOperationKind.RESPOND)

    async def get_operation(self, operation_id: str) -> WorkOperationAcknowledgement:
        self._raise()
        self.last_call = ("operation", operation_id)
        return self._operation(WorkOperationKind.SEND)

    async def abandon_operation(
        self, operation_id: str
    ) -> WorkOperationAcknowledgement:
        self._raise()
        self.last_call = ("abandon", operation_id)
        return replace(
            self._operation(WorkOperationKind.SEND),
            state=WorkOperationState.ABANDONED,
        )

    def _operation(self, kind: WorkOperationKind) -> WorkOperationAcknowledgement:
        return WorkOperationAcknowledgement(
            operation_id=OPERATION_ID,
            kind=kind,
            state=WorkOperationState.ACCEPTED,
            thread_id=THREAD_ID,
            turn_id=None,
        )


def _client(work: StubWork | None = None) -> TestClient:
    return TestClient(
        create_test_app(work=work),
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    )


def test_work_shell_deep_links_are_loopback_only_and_uncached() -> None:
    client = _client(StubWork())
    for path in ("/admin/work", f"/admin/work/{THREAD_ID}"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert 'id="view-work"' in response.text

    remote = TestClient(
        create_test_app(work=StubWork()),
        client=("203.0.113.10", 50000),
    )
    assert remote.get("/admin/work").status_code == 403


def test_work_reads_serialize_native_state_without_exposing_private_ids() -> None:
    work = StubWork()
    client = _client(work)

    bootstrap = client.get("/admin/api/work/bootstrap")
    listed = client.get("/admin/api/work/sessions", params={"query": "repo"})
    detail = client.get(f"/admin/api/work/sessions/{THREAD_ID}")
    assert bootstrap.status_code == listed.status_code == detail.status_code == 200
    assert bootstrap.json()["recent_projects"] == ["C:\\example"]
    assert bootstrap.json()["unresolved_creates"] == []
    assert listed.json()["sessions"][0]["thread_id"] == THREAD_ID
    assert listed.json()["next_cursor"]
    item = detail.json()["turns"]["items"][0]
    assert "<strong>safe</strong>" in item["html"]
    assert "<script>" not in item["html"]
    assert "connection_id" not in detail.text
    assert "request_id" not in detail.text
    for response in (bootstrap, listed, detail):
        assert response.headers["cache-control"] == "no-store"


def test_work_mutations_forward_explicit_revision_operation_and_answer_shapes() -> None:
    work = StubWork()
    client = _client(work)

    cases = (
        (
            "post",
            "/admin/api/work/sessions",
            {"cwd": "C:\\repo", "operation_id": OPERATION_ID},
            ("create", ("C:\\repo", OPERATION_ID)),
            202,
        ),
        (
            "patch",
            f"/admin/api/work/sessions/{THREAD_ID}/settings",
            {"expected_revision": 1, "updates": {"model": "model-1"}},
            ("settings", (THREAD_ID, 1, {"model": "model-1"})),
            200,
        ),
        (
            "post",
            f"/admin/api/work/sessions/{THREAD_ID}/turns",
            {
                "expected_revision": 1,
                "operation_id": OPERATION_ID,
                "text": "hello",
            },
            ("send", (THREAD_ID, 1, OPERATION_ID, "hello")),
            202,
        ),
        (
            "post",
            f"/admin/api/work/sessions/{THREAD_ID}/stop",
            {"operation_id": OPERATION_ID},
            ("stop", (THREAD_ID, OPERATION_ID)),
            202,
        ),
        (
            "post",
            f"/admin/api/work/sessions/{THREAD_ID}/delete",
            {"operation_id": OPERATION_ID},
            ("delete", (THREAD_ID, OPERATION_ID)),
            202,
        ),
        (
            "post",
            f"/admin/api/work/sessions/{THREAD_ID}/remove",
            None,
            ("remove", THREAD_ID),
            200,
        ),
        (
            "post",
            f"/admin/api/work/sessions/{THREAD_ID}/interactions/interaction-1/responses",
            {"operation_id": OPERATION_ID, "value": {"decision": "accept"}},
            (
                "respond",
                (
                    THREAD_ID,
                    "interaction-1",
                    OPERATION_ID,
                    {"decision": "accept"},
                ),
            ),
            202,
        ),
        (
            "get",
            f"/admin/api/work/operations/{OPERATION_ID}",
            None,
            ("operation", OPERATION_ID),
            200,
        ),
        (
            "post",
            f"/admin/api/work/operations/{OPERATION_ID}/abandon",
            {"confirm": True},
            ("abandon", OPERATION_ID),
            200,
        ),
    )
    for method, path, body, expected, status in cases:
        response = client.request(method, path, json=body)
        assert response.status_code == status
        assert response.headers["cache-control"] == "no-store"
        assert work.last_call == expected
        if status == 202:
            assert response.headers["location"] == (
                f"/admin/api/work/operations/{OPERATION_ID}"
            )


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (WorkValidationError("invalid"), 400),
        (WorkNotFoundError("missing"), 404),
        (WorkConflictError("stale"), 409),
        (WorkCompatibilityError("update Codex"), 426),
        (WorkUnavailableError("unavailable"), 503),
    ],
)
def test_work_error_types_have_narrow_statuses(error: Exception, status: int) -> None:
    work = StubWork()
    work.error = error
    response = _client(work).get("/admin/api/work/bootstrap")

    assert response.status_code == status
    assert response.json()["detail"] == str(error)
    assert response.headers["cache-control"] == "no-store"


def test_missing_work_service_and_invalid_cursor_fail_cleanly() -> None:
    unavailable = _client().get("/admin/api/work/bootstrap")
    invalid_cursor = _client(StubWork()).get(
        "/admin/api/work/sessions", params={"cursor": "not-base64"}
    )

    assert unavailable.status_code == 503
    assert invalid_cursor.status_code == 400


def test_work_event_feed_emits_barrier_before_updates_and_closes_subscription() -> None:
    work = StubWork()
    with (
        _client(work) as client,
        client.stream("GET", "/admin/api/work/events") as response,
    ):
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert body.index("event: feed.ready") < body.index("event: session.status")
    assert '"generation": "generation-1"' in body
    assert '"status": "working"' in body
    assert work.subscription is not None
    assert work.subscription.closed is True
