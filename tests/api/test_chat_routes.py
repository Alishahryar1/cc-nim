import asyncio
import base64
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace
from io import BytesIO
from typing import BinaryIO, cast

import pytest
from fastapi.testclient import TestClient
from starlette.types import Message, Scope

from free_claude_code.api.chat_routes import _stream_response
from free_claude_code.application.chat import (
    MAX_CHAT_ATTACHMENT_BYTES,
    ChatAttachment,
    ChatAttachmentContent,
    ChatAttachmentKind,
    ChatCompaction,
    ChatContextEstimate,
    ChatModelOption,
    ChatPayloadTooLargeError,
    ChatPreferences,
    ChatReasoning,
    ChatSegment,
    ChatSession,
    ChatSessionDetail,
    ChatSessionPage,
    ChatSessionSummary,
    ChatStreamEvent,
    ChatTurn,
    ChatUnsupportedAttachmentError,
    ChatValidationError,
    GenerationStatus,
    SegmentKind,
)
from free_claude_code.application.chat.models import ChatGeneration
from free_claude_code.core.model_capabilities import ModelInputModality
from tests.api.support import create_test_app

SESSION_ID = "29e3b8fd-8744-4377-b8cf-4c9d48daf962"
OPERATION_ID = "7cd43d62-c1aa-42f8-9963-6c0811c0dfaf"


class StubStream:
    operation_id = OPERATION_ID

    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self) -> AsyncIterator[ChatStreamEvent]:
        return self._events()

    async def _events(self) -> AsyncIterator[ChatStreamEvent]:
        yield ChatStreamEvent(
            event="turn.completed",
            sequence=1,
            data={"operation_id": OPERATION_ID, "session_id": SESSION_ID},
        )

    async def aclose(self) -> None:
        self.closed = True


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
        self.last_stream: StubStream | None = None
        self.deleted = False
        self.active_operation = False

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
            active_operation=self.active_operation,
        )

    async def stage_attachment(
        self,
        session_id: str,
        *,
        filename: str,
        declared_media_type: str | None,
        source: BinaryIO,
    ) -> ChatAttachment:
        assert session_id == SESSION_ID
        assert declared_media_type == "text/plain"
        assert source.read() == b"hello"
        return ChatAttachment(
            id="e5f3e75e-a031-4d6c-88b6-1382abaca2f7",
            session_id=SESSION_ID,
            turn_id=None,
            position=0,
            filename=filename,
            kind=ChatAttachmentKind.TEXT,
            media_type="text/plain",
            byte_size=5,
            extracted_characters=5,
            created_at=3,
        )

    async def remove_attachment(self, session_id: str, attachment_id: str) -> None:
        assert session_id == SESSION_ID
        assert attachment_id == "e5f3e75e-a031-4d6c-88b6-1382abaca2f7"

    async def attachment_content(
        self, session_id: str, attachment_id: str
    ) -> ChatAttachmentContent:
        attachment = await self.stage_attachment(
            session_id,
            filename="note.txt",
            declared_media_type="text/plain",
            source=BytesIO(b"hello"),
        )
        assert attachment_id == attachment.id
        return ChatAttachmentContent(attachment=attachment, data=b"hello")

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

    async def estimate(
        self,
        session_id: str,
        *,
        draft: str,
        attachment_ids: tuple[str, ...] = (),
    ) -> ChatContextEstimate:
        assert session_id == SESSION_ID
        del draft, attachment_ids
        return ChatContextEstimate(100, 1_024, 32_000, 30_976, 0.01, False, False)

    async def send(
        self,
        session_id: str,
        *,
        expected_revision: int,
        operation_id: str,
        text: str,
        attachment_ids: tuple[str, ...] = (),
    ) -> StubStream:
        assert (session_id, expected_revision, operation_id, text) == (
            SESSION_ID,
            self.session.revision,
            OPERATION_ID,
            "hello",
        )
        assert attachment_ids == ()
        self.last_stream = StubStream()
        return self.last_stream

    async def retry(
        self,
        session_id: str,
        *,
        expected_revision: int,
        operation_id: str,
    ) -> StubStream:
        del session_id, expected_revision, operation_id
        return StubStream()

    async def regenerate(
        self,
        session_id: str,
        *,
        expected_revision: int,
        operation_id: str,
    ) -> StubStream:
        del session_id, expected_revision, operation_id
        return StubStream()

    async def compact(
        self,
        session_id: str,
        *,
        expected_revision: int,
        operation_id: str,
    ) -> StubStream:
        del session_id, expected_revision, operation_id
        return StubStream()

    async def stop(self, session_id: str, *, operation_id: str) -> bool:
        return (session_id, operation_id) == (SESSION_ID, OPERATION_ID)


class UnestimatableChat(StubChat):
    async def estimate(
        self,
        session_id: str,
        *,
        draft: str,
        attachment_ids: tuple[str, ...] = (),
    ) -> ChatContextEstimate:
        assert session_id == SESSION_ID
        del draft, attachment_ids
        raise ChatValidationError(
            "This model does not support reasoning. Set thinking to Off."
        )


class RejectedAttachmentChat(StubChat):
    def __init__(self, error: ChatValidationError) -> None:
        super().__init__()
        self.error = error

    async def stage_attachment(
        self,
        session_id: str,
        *,
        filename: str,
        declared_media_type: str | None,
        source: BinaryIO,
    ) -> ChatAttachment:
        del session_id, filename, declared_media_type, source
        raise self.error


class AttachmentLimitChat(StubChat):
    async def estimate(
        self,
        session_id: str,
        *,
        draft: str,
        attachment_ids: tuple[str, ...] = (),
    ) -> ChatContextEstimate:
        if len(attachment_ids) > 5:
            raise ChatPayloadTooLargeError(
                "A message may contain at most five attachments."
            )
        return await super().estimate(
            session_id,
            draft=draft,
            attachment_ids=attachment_ids,
        )

    async def send(
        self,
        session_id: str,
        *,
        expected_revision: int,
        operation_id: str,
        text: str,
        attachment_ids: tuple[str, ...] = (),
    ) -> StubStream:
        if len(attachment_ids) > 5:
            raise ChatPayloadTooLargeError(
                "A message may contain at most five attachments."
            )
        return await super().send(
            session_id,
            expected_revision=expected_revision,
            operation_id=operation_id,
            text=text,
            attachment_ids=attachment_ids,
        )


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
    chat.active_operation = True
    client = _client(chat)

    bootstrap = client.get("/admin/api/chat/bootstrap").json()
    detail = client.get(f"/admin/api/chat/sessions/{SESSION_ID}").json()

    assert bootstrap["models"][0]["supports_reasoning"] is True
    assert bootstrap["models"][0]["input_modalities"] == ["text"]
    segment = detail["turns"][0]["generation"]["segments"][0]
    assert segment["text"] == "**safe** <script>x</script>"
    assert "<strong>safe</strong>" in segment["html"]
    assert "<script>" not in segment["html"]
    assert detail["turns"][0]["generation"]["actual_model"] == ("open_router/fallback")
    assert detail["turns"][0]["operation_id"] == "operation"
    assert detail["active_operation"] is True


def test_chat_attachment_routes_upload_download_and_remove():
    client = _client(StubChat())
    attachment_id = "e5f3e75e-a031-4d6c-88b6-1382abaca2f7"

    uploaded = client.post(
        f"/admin/api/chat/sessions/{SESSION_ID}/attachments",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    downloaded = client.get(
        f"/admin/api/chat/sessions/{SESSION_ID}/attachments/{attachment_id}/content"
    )
    removed = client.delete(
        f"/admin/api/chat/sessions/{SESSION_ID}/attachments/{attachment_id}"
    )

    assert uploaded.status_code == 201
    assert uploaded.json()["filename"] == "note.txt"
    assert uploaded.json()["content_url"].endswith(f"{attachment_id}/content")
    assert "hello" not in uploaded.text
    assert "\\" not in uploaded.json()["content_url"]
    assert downloaded.content == b"hello"
    assert downloaded.headers["content-type"].startswith("text/plain")
    assert downloaded.headers["content-disposition"].startswith("attachment;")
    assert downloaded.headers["x-content-type-options"] == "nosniff"
    assert downloaded.headers["cache-control"] == "no-store"
    assert removed.json() == {"deleted": True}


def test_chat_attachment_upload_requires_exactly_one_file():
    response = _client(StubChat()).post(
        f"/admin/api/chat/sessions/{SESSION_ID}/attachments",
        files=[
            ("file", ("one.txt", b"one", "text/plain")),
            ("file", ("two.txt", b"two", "text/plain")),
        ],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Upload exactly one attachment file."


@pytest.mark.parametrize(
    ("content", "headers"),
    [
        (b"not multipart", {}),
        (
            b"not a multipart body",
            {"Content-Type": "multipart/form-data; boundary=fcc-invalid"},
        ),
    ],
)
def test_chat_attachment_upload_maps_malformed_multipart_to_validation_error(
    content: bytes,
    headers: dict[str, str],
):
    response = _client(StubChat()).post(
        f"/admin/api/chat/sessions/{SESSION_ID}/attachments",
        content=content,
        headers=headers,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Upload exactly one attachment file."


def test_chat_attachment_upload_rejects_file_under_an_extra_field_name():
    response = _client(StubChat()).post(
        f"/admin/api/chat/sessions/{SESSION_ID}/attachments",
        files=[
            ("file", ("one.txt", b"one", "text/plain")),
            ("ignored", ("two.txt", b"two", "text/plain")),
        ],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Upload exactly one attachment file."


def test_chat_attachment_upload_rejects_declared_oversize_before_parsing():
    response = _client(StubChat()).post(
        f"/admin/api/chat/sessions/{SESSION_ID}/attachments",
        files={"file": ("large.txt", b"small", "text/plain")},
        headers={"Content-Length": str(MAX_CHAT_ATTACHMENT_BYTES + 65_537)},
    )

    assert response.status_code == 413
    assert response.json()["code"] == "ChatPayloadTooLargeError"


def test_chat_attachment_upload_stops_chunked_oversize_while_streaming():
    boundary = "fcc-attachment-boundary"
    prefix = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="large.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()

    def body():
        yield prefix
        yield b"x" * (MAX_CHAT_ATTACHMENT_BYTES + 65_537)
        yield suffix

    response = _client(StubChat()).post(
        f"/admin/api/chat/sessions/{SESSION_ID}/attachments",
        content=body(),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Transfer-Encoding": "chunked",
        },
    )

    assert response.status_code == 413
    assert response.json()["code"] == "ChatPayloadTooLargeError"


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ChatPayloadTooLargeError("too large"), 413),
        (ChatUnsupportedAttachmentError("unsupported"), 422),
    ],
)
def test_chat_attachment_errors_keep_specific_http_status(
    error: ChatValidationError,
    status: int,
):
    response = _client(RejectedAttachmentChat(error)).post(
        f"/admin/api/chat/sessions/{SESSION_ID}/attachments",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )

    assert response.status_code == status
    assert response.json()["code"] == type(error).__name__


@pytest.mark.parametrize("action", ("estimate", "send"))
def test_chat_attachment_count_limit_is_reported_as_413(action: str):
    attachment_ids = [str(uuid.uuid4()) for _index in range(6)]
    payload = (
        {"draft": "hello", "attachment_ids": attachment_ids}
        if action == "estimate"
        else {
            "expected_revision": 1,
            "operation_id": OPERATION_ID,
            "text": "hello",
            "attachment_ids": attachment_ids,
        }
    )

    response = _client(AttachmentLimitChat()).post(
        f"/admin/api/chat/sessions/{SESSION_ID}/{action}",
        json=payload,
    )

    assert response.status_code == 413
    assert response.json()["code"] == "ChatPayloadTooLargeError"


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


def test_chat_stream_serializes_events_and_closes_operation():
    chat = StubChat()
    response = _client(chat).post(
        f"/admin/api/chat/sessions/{SESSION_ID}/send",
        json={
            "expected_revision": 1,
            "operation_id": OPERATION_ID,
            "text": "hello",
        },
    )

    assert response.status_code == 200
    assert "event: turn.completed" in response.text
    assert "id: 1" in response.text
    assert chat.last_stream is not None and chat.last_stream.closed is True


@pytest.mark.asyncio
async def test_chat_stream_closes_when_cancelled_before_body_iteration():
    stream = StubStream()
    response = _stream_response(stream)
    response_started = asyncio.Event()

    async def receive() -> Message:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(message: Message) -> None:
        assert message["type"] == "http.response.start"
        response_started.set()
        await asyncio.Event().wait()

    scope = cast(
        Scope,
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": f"/admin/api/chat/sessions/{SESSION_ID}/send",
            "raw_path": b"/admin/api/chat/sessions/send",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8000),
        },
    )
    response_task = asyncio.create_task(response(scope, receive, send))
    await asyncio.wait_for(response_started.wait(), timeout=1)
    response_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await response_task
    assert stream.closed is True


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
