import base64
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from free_claude_code.application.chat import (
    ChatActiveOperation,
    ChatCompaction,
    ChatContextEstimate,
    ChatEventOverflowError,
    ChatModelOption,
    ChatOperationAcknowledgement,
    ChatOperationKind,
    ChatOperationPhase,
    ChatPreferences,
    ChatPublishedEvent,
    ChatReasoning,
    ChatSegment,
    ChatSession,
    ChatSessionDetail,
    ChatSessionPage,
    ChatSessionSummary,
    ChatTurn,
    ChatValidationError,
    GenerationStatus,
    SegmentKind,
)
from free_claude_code.application.chat.models import ChatGeneration
from free_claude_code.core.model_capabilities import ModelInputModality
from tests.api.support import create_test_app

SESSION_ID = "29e3b8fd-8744-4377-b8cf-4c9d48daf962"
OPERATION_ID = "7cd43d62-c1aa-42f8-9963-6c0811c0dfaf"


class StubSubscription:
    cursor = 4

    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self):
        return self._events()

    async def _events(self):
        yield ChatPublishedEvent(
            event="turn.completed",
            id=5,
            data={
                "operation_id": OPERATION_ID,
                "session_id": SESSION_ID,
                "kind": "send",
                "operation_sequence": 3,
            },
        )

    async def aclose(self) -> None:
        self.closed = True


class OverflowSubscription(StubSubscription):
    cursor = 7

    async def _events(self):
        if False:
            yield
        raise ChatEventOverflowError(9)


class StubChat:
    def __init__(self) -> None:
        self.session = ChatSession(
            id=SESSION_ID,
            title="Example",
            model="groq/model",
            reasoning=ChatReasoning.MEDIUM,
            revision=1,
            created_at=1,
            updated_at=2,
        )
        generation = ChatGeneration(
            id="generation",
            status=GenerationStatus.COMPLETED,
            requested_model="groq/model",
            actual_model="open_router/fallback",
            reasoning=ChatReasoning.MEDIUM,
            effective_output_limit=1024,
            stop_reason="end_turn",
            error_code=None,
            error_message=None,
            started_at=1,
            finished_at=2,
            segments=(ChatSegment(0, SegmentKind.TEXT, "**safe** <script>x</script>"),),
        )
        self.turn = ChatTurn(
            id="turn",
            session_id=SESSION_ID,
            operation_id="operation",
            sequence=1,
            user_text="hello",
            created_at=1,
            generation=generation,
        )
        self.preferences_value = ChatPreferences(
            system_prompt="prompt",
            last_model=self.session.model,
            last_reasoning=self.session.reasoning,
            updated_at=1,
        )
        self.last_subscription: StubSubscription | None = None
        self.deleted = False
        self.active_operation: ChatActiveOperation | None = None

    def availability(self) -> tuple[bool, str | None]:
        return True, None

    def models(self) -> tuple[ChatModelOption, ...]:
        return (
            ChatModelOption(
                model_ref="groq/model",
                provider_id="groq",
                model_id="model",
                supports_reasoning=True,
                input_modalities=frozenset({ModelInputModality.TEXT}),
                context_window_tokens=32_000,
                max_output_tokens=8_000,
            ),
        )

    async def subscribe(
        self,
    ) -> tuple[StubSubscription, tuple[ChatActiveOperation, ...]]:
        self.last_subscription = StubSubscription()
        active = (self.active_operation,) if self.active_operation is not None else ()
        return self.last_subscription, active

    async def preferences(self) -> ChatPreferences:
        return self.preferences_value

    async def save_system_prompt(self, value: str) -> ChatPreferences:
        self.preferences_value = replace(self.preferences_value, system_prompt=value)
        return self.preferences_value

    async def reset_system_prompt(self) -> ChatPreferences:
        return await self.save_system_prompt("default")

    async def create_session(self) -> ChatSession:
        return self.session

    async def get_detail(self, session_id: str) -> ChatSessionDetail:
        session = await self.get_session(session_id)
        turns, next_before, compaction = await self.get_turn_page(
            session_id,
            before_sequence=None,
            limit=50,
        )
        context: ChatContextEstimate | None
        context_error: str | None = None
        try:
            context = await self.estimate(session_id, draft="")
        except ChatValidationError as exc:
            context = None
            context_error = str(exc)
        return ChatSessionDetail(
            session=session,
            turns=turns,
            next_before=next_before,
            compaction=compaction,
            context=context,
            context_error=context_error,
        )

    async def list_sessions(
        self,
        *,
        query: str,
        cursor: tuple[int, str] | None,
        limit: int,
    ) -> ChatSessionPage:
        del query, cursor, limit
        return ChatSessionPage(
            sessions=(
                ChatSessionSummary(
                    id=self.session.id,
                    title=self.session.title,
                    model=self.session.model,
                    reasoning=self.session.reasoning,
                    revision=self.session.revision,
                    preview="hello",
                    created_at=self.session.created_at,
                    updated_at=self.session.updated_at,
                ),
            ),
            next_cursor=None,
        )

    async def get_session(self, session_id: str) -> ChatSession:
        assert session_id == SESSION_ID
        return self.session

    async def update_session(
        self,
        session_id: str,
        *,
        expected_revision: int,
        title: str | None,
        model: str | None,
        reasoning: ChatReasoning | None,
    ) -> ChatSession:
        assert session_id == SESSION_ID
        assert expected_revision == self.session.revision
        self.session = replace(
            self.session,
            title=title or self.session.title,
            model=model or self.session.model,
            reasoning=reasoning or self.session.reasoning,
            revision=self.session.revision + 1,
        )
        return self.session

    async def delete_session(self, session_id: str, *, expected_revision: int) -> None:
        assert session_id == SESSION_ID
        assert expected_revision == self.session.revision
        self.deleted = True

    async def get_turn_page(
        self,
        session_id: str,
        *,
        before_sequence: int | None,
        limit: int,
    ) -> tuple[tuple[ChatTurn, ...], int | None, ChatCompaction | None]:
        assert session_id == SESSION_ID
        del before_sequence, limit
        return (self.turn,), None, None

    async def estimate(self, session_id: str, *, draft: str) -> ChatContextEstimate:
        assert session_id == SESSION_ID
        del draft
        return ChatContextEstimate(100, 1_024, 32_000, 30_976, 0.01, False, False)

    async def send(
        self,
        session_id: str,
        *,
        expected_revision: int,
        operation_id: str,
        text: str,
    ) -> ChatOperationAcknowledgement:
        assert (session_id, expected_revision, operation_id, text) == (
            SESSION_ID,
            self.session.revision,
            OPERATION_ID,
            "hello",
        )
        return ChatOperationAcknowledgement(
            session_id=session_id,
            operation_id=operation_id,
            kind=ChatOperationKind.SEND,
        )

    async def retry(
        self,
        session_id: str,
        *,
        expected_revision: int,
        operation_id: str,
    ) -> ChatOperationAcknowledgement:
        del expected_revision
        return ChatOperationAcknowledgement(
            session_id=session_id,
            operation_id=operation_id,
            kind=ChatOperationKind.RETRY,
        )

    async def regenerate(
        self,
        session_id: str,
        *,
        expected_revision: int,
        operation_id: str,
    ) -> ChatOperationAcknowledgement:
        del expected_revision
        return ChatOperationAcknowledgement(
            session_id=session_id,
            operation_id=operation_id,
            kind=ChatOperationKind.REGENERATE,
        )

    async def compact(
        self,
        session_id: str,
        *,
        expected_revision: int,
        operation_id: str,
    ) -> ChatOperationAcknowledgement:
        del expected_revision
        return ChatOperationAcknowledgement(
            session_id=session_id,
            operation_id=operation_id,
            kind=ChatOperationKind.COMPACT,
        )

    async def stop(self, session_id: str, *, operation_id: str) -> bool:
        return (session_id, operation_id) == (SESSION_ID, OPERATION_ID)


class UnestimatableChat(StubChat):
    async def estimate(self, session_id: str, *, draft: str) -> ChatContextEstimate:
        assert session_id == SESSION_ID
        del draft
        raise ChatValidationError(
            "This model does not support reasoning. Set thinking to Off."
        )


class OverflowChat(StubChat):
    async def subscribe(
        self,
    ) -> tuple[OverflowSubscription, tuple[ChatActiveOperation, ...]]:
        subscription = OverflowSubscription()
        self.last_subscription = subscription
        return subscription, ()

    async def stop(self, session_id: str, *, operation_id: str) -> bool:
        raise AssertionError((session_id, operation_id))


def _client(chat: StubChat | None = None) -> TestClient:
    return TestClient(
        create_test_app(chat=chat),
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    )


def test_chat_deep_links_serve_the_versioned_admin_shell():
    response = _client().get(f"/admin/chat/{SESSION_ID}")

    assert response.status_code == 200
    assert "chat_sessions.js" in response.text
    assert "Chat Sessions" not in response.headers.get("cache-control", "")
    assert response.headers["cache-control"] == "no-store"


def test_chat_bootstrap_and_detail_project_rich_models_and_safe_markdown():
    chat = StubChat()
    chat.active_operation = ChatActiveOperation(
        session_id=SESSION_ID,
        operation_id=OPERATION_ID,
        kind=ChatOperationKind.SEND,
        phase=ChatOperationPhase.GENERATING,
        operation_sequence=2,
        submitted_text="next",
        turn_id="turn-next",
        generation_id="generation-next",
        regeneration=False,
        actual_model="groq/model",
        segments=(ChatSegment(0, SegmentKind.TEXT, "live"),),
    )
    client = _client(chat)

    bootstrap = client.get("/admin/api/chat/bootstrap").json()
    detail = client.get(f"/admin/api/chat/sessions/{SESSION_ID}").json()
    feed = client.get("/admin/api/chat/events")

    assert bootstrap["models"][0]["supports_reasoning"] is True
    assert bootstrap["models"][0]["input_modalities"] == ["text"]
    segment = detail["turns"][0]["generation"]["segments"][0]
    assert segment["text"] == "**safe** <script>x</script>"
    assert "<strong>safe</strong>" in segment["html"]
    assert "<script>" not in segment["html"]
    assert detail["turns"][0]["generation"]["actual_model"] == ("open_router/fallback")
    assert detail["turns"][0]["operation_id"] == "operation"
    assert "active_operation" not in detail
    assert '"submitted_text": "next"' in feed.text


def test_chat_detail_stays_readable_when_context_controls_need_repair():
    response = _client(UnestimatableChat()).get(
        f"/admin/api/chat/sessions/{SESSION_ID}"
    )

    assert response.status_code == 200
    assert response.json()["context"] is None
    assert response.json()["context_error"] == (
        "This model does not support reasoning. Set thinking to Off."
    )


def test_chat_crud_and_prompt_routes_use_application_port():
    chat = StubChat()
    client = _client(chat)

    created = client.post("/admin/api/chat/sessions", json={})
    renamed = client.patch(
        f"/admin/api/chat/sessions/{SESSION_ID}",
        json={"expected_revision": 1, "title": "Renamed"},
    )
    prompt = client.put(
        "/admin/api/chat/preferences/system-prompt",
        json={"value": "custom"},
    )
    deleted = client.request(
        "DELETE",
        f"/admin/api/chat/sessions/{SESSION_ID}",
        json={"expected_revision": 2},
    )

    assert created.status_code == 201
    assert renamed.json()["title"] == "Renamed"
    assert prompt.json()["system_prompt"] == "custom"
    assert deleted.json() == {"deleted": True}
    assert chat.deleted is True


def test_chat_long_operation_acknowledges_without_owning_its_event_stream():
    chat = StubChat()
    response = _client(chat).post(
        f"/admin/api/chat/sessions/{SESSION_ID}/send",
        json={
            "expected_revision": 1,
            "operation_id": OPERATION_ID,
            "text": "hello",
        },
    )

    assert response.status_code == 202
    assert response.json() == {
        "session_id": SESSION_ID,
        "operation_id": OPERATION_ID,
        "kind": "send",
    }
    assert chat.last_subscription is None


def test_chat_event_feed_starts_at_snapshot_barrier_and_closes_subscription():
    chat = StubChat()

    response = _client(chat).get("/admin/api/chat/events")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-accel-buffering"] == "no"
    assert "event: feed.ready" in response.text
    assert "id: 4" in response.text
    assert 'data: {"cursor": 4, "active_operations": []}' in response.text
    assert "event: turn.completed" in response.text
    assert "id: 5" in response.text
    assert chat.last_subscription is not None
    assert chat.last_subscription.closed is True


def test_chat_event_feed_overflow_requests_resync_without_stopping_work():
    chat = OverflowChat()

    response = _client(chat).get("/admin/api/chat/events")

    assert response.status_code == 200
    assert "event: feed.ready" in response.text
    assert "id: 7" in response.text
    assert "event: feed.resync_required" in response.text
    assert "id: 9" in response.text
    assert 'data: {"cursor": 9}' in response.text
    assert chat.last_subscription is not None
    assert chat.last_subscription.closed is True


def test_chat_routes_apply_loopback_and_origin_protection():
    chat = StubChat()
    remote = TestClient(
        create_test_app(chat=chat),
        client=("203.0.113.5", 50000),
    )
    local = _client(chat)

    assert remote.get("/admin/api/chat/bootstrap").status_code == 403
    assert (
        local.get(
            "/admin/api/chat/bootstrap",
            headers={"Origin": "https://example.com"},
        ).status_code
        == 403
    )
    assert (
        local.get(
            "/admin/api/chat/bootstrap",
            headers={"Host": "attacker.example:8000"},
        ).status_code
        == 403
    )


@pytest.mark.parametrize(
    "origin",
    (
        "http://127.0.0.1:not-a-port",
        "https://[::1",
    ),
)
def test_chat_routes_reject_malformed_local_origins(origin: str):
    response = _client(StubChat()).get(
        "/admin/api/chat/bootstrap",
        headers={"Origin": origin},
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Admin UI is local-only"}


@pytest.mark.parametrize(
    "origin",
    (
        "http://127.0.0.1:8000",
        "https://localhost",
        "http://[::1]:8000",
    ),
)
def test_chat_routes_accept_valid_local_origins(origin: str):
    response = _client(StubChat()).get(
        "/admin/api/chat/bootstrap",
        headers={"Origin": origin},
    )

    assert response.status_code == 200


def test_chat_without_composed_service_isolated_as_503():
    response = _client().get("/admin/api/chat/bootstrap")

    assert response.status_code == 503
    assert response.json()["code"] == "ChatUnavailableError"


def test_invalid_session_cursor_is_rejected():
    client = _client(StubChat())
    malformed = client.get("/admin/api/chat/sessions", params={"cursor": "not-valid"})
    non_uuid = base64.urlsafe_b64encode(b"1:not-a-session").decode().rstrip("=")
    invalid_id = client.get("/admin/api/chat/sessions", params={"cursor": non_uuid})

    assert malformed.status_code == 400
    assert malformed.json()["code"] == "ChatValidationError"
    assert invalid_id.status_code == 400
    assert invalid_id.json()["code"] == "ChatValidationError"
