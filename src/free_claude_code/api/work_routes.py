"""Loopback-only HTTP adapter for Codex-backed Work Sessions."""

import base64
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

from free_claude_code.application.event_feed import (
    EventFeedOverflowError,
    PublishedEvent,
)
from free_claude_code.application.work import (
    WorkApplicationPort,
    WorkBootstrap,
    WorkInteraction,
    WorkOperationAcknowledgement,
    WorkSessionDetail,
    WorkSessionRecord,
    WorkSessionSummary,
    WorkTimelineItem,
    WorkTurnPage,
    WorkUnavailableError,
    WorkValidationError,
)
from free_claude_code.core.json_types import JsonObject, JsonValue

from .admin_markdown import render_admin_markdown
from .admin_routes import admin_page_response
from .admin_security import require_loopback_admin
from .dependencies import get_services
from .ports import ApiServices

router = APIRouter()


class WorkCreatePayload(BaseModel):
    cwd: str = Field(max_length=32_768)
    operation_id: str


class WorkSettingsPayload(BaseModel):
    expected_revision: int = Field(gt=0)
    updates: JsonObject


class WorkSendPayload(BaseModel):
    expected_revision: int = Field(gt=0)
    operation_id: str
    text: str = Field(max_length=1_000_000)


class WorkOperationPayload(BaseModel):
    operation_id: str


class WorkInteractionPayload(BaseModel):
    operation_id: str
    value: JsonValue


class WorkAbandonPayload(BaseModel):
    confirm: bool


@router.get("/admin/work", include_in_schema=False)
@router.get("/admin/work/{thread_id}", include_in_schema=False)
def work_page(request: Request, thread_id: str | None = None):
    del thread_id
    require_loopback_admin(request)
    return admin_page_response()


@router.get("/admin/api/work/bootstrap")
async def work_bootstrap(
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    return _bootstrap_payload(await _work(services).bootstrap())


@router.get("/admin/api/work/events", response_class=EventSourceResponse)
async def work_events(
    request: Request,
    services: ApiServices = Depends(get_services),
) -> AsyncIterator[ServerSentEvent]:
    require_loopback_admin(request)
    work = _work(services)
    subscription = await work.subscribe()
    bootstrap = await work.bootstrap()
    try:
        yield ServerSentEvent(
            event="feed.ready",
            id=str(subscription.cursor),
            retry=1_000,
            data={
                "cursor": subscription.cursor,
                "generation": bootstrap.event_generation,
            },
        )
        try:
            async for event in subscription:
                yield _sse_event(event)
        except EventFeedOverflowError as exc:
            yield ServerSentEvent(
                event="feed.resync_required",
                id=str(exc.cursor),
                data={"cursor": exc.cursor},
            )
    finally:
        await subscription.aclose()


@router.get("/admin/api/work/sessions")
async def list_work_sessions(
    request: Request,
    query: str = Query(default="", max_length=200),
    cursor: str | None = None,
    limit: int = Query(default=25, ge=1, le=25),
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    page = await _work(services).list_sessions(
        query=query,
        cursor=_decode_cursor(cursor),
        limit=limit,
    )
    return {
        "sessions": [_summary_payload(summary) for summary in page.sessions],
        "next_cursor": _encode_cursor(page.next_cursor),
    }


@router.post("/admin/api/work/sessions", status_code=202)
async def create_work_session(
    payload: WorkCreatePayload,
    request: Request,
    response: Response,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    acknowledgement = await _work(services).create_session(
        cwd=payload.cwd,
        operation_id=payload.operation_id,
    )
    _set_operation_location(response, acknowledgement)
    return _operation_payload(acknowledgement)


@router.get("/admin/api/work/sessions/{thread_id}")
async def get_work_session(
    thread_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    return _detail_payload(await _work(services).get_detail(thread_id))


@router.patch("/admin/api/work/sessions/{thread_id}/settings")
async def update_work_settings(
    thread_id: str,
    payload: WorkSettingsPayload,
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    return _record_payload(
        await _work(services).update_settings(
            thread_id,
            expected_revision=payload.expected_revision,
            updates=payload.updates,
        )
    )


@router.post("/admin/api/work/sessions/{thread_id}/turns", status_code=202)
async def send_work_turn(
    thread_id: str,
    payload: WorkSendPayload,
    request: Request,
    response: Response,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    acknowledgement = await _work(services).send(
        thread_id,
        expected_revision=payload.expected_revision,
        operation_id=payload.operation_id,
        text=payload.text,
    )
    _set_operation_location(response, acknowledgement)
    return _operation_payload(acknowledgement)


@router.post("/admin/api/work/sessions/{thread_id}/stop", status_code=202)
async def stop_work_turn(
    thread_id: str,
    payload: WorkOperationPayload,
    request: Request,
    response: Response,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    acknowledgement = await _work(services).stop(
        thread_id, operation_id=payload.operation_id
    )
    _set_operation_location(response, acknowledgement)
    return _operation_payload(acknowledgement)


@router.post("/admin/api/work/sessions/{thread_id}/delete", status_code=202)
async def delete_work_session(
    thread_id: str,
    payload: WorkOperationPayload,
    request: Request,
    response: Response,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    acknowledgement = await _work(services).delete(
        thread_id, operation_id=payload.operation_id
    )
    _set_operation_location(response, acknowledgement)
    return _operation_payload(acknowledgement)


@router.post("/admin/api/work/sessions/{thread_id}/remove")
async def remove_missing_work_session(
    thread_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    await _work(services).remove_missing(thread_id)
    return {"removed": True}


@router.post(
    "/admin/api/work/sessions/{thread_id}/interactions/{interaction_id}/responses",
    status_code=202,
)
async def respond_to_work_interaction(
    thread_id: str,
    interaction_id: str,
    payload: WorkInteractionPayload,
    request: Request,
    response: Response,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    acknowledgement = await _work(services).respond(
        thread_id,
        interaction_id,
        operation_id=payload.operation_id,
        value=payload.value,
    )
    _set_operation_location(response, acknowledgement)
    return _operation_payload(acknowledgement)


@router.get("/admin/api/work/operations/{operation_id}")
async def get_work_operation(
    operation_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    return _operation_payload(await _work(services).get_operation(operation_id))


@router.post("/admin/api/work/operations/{operation_id}/abandon")
async def abandon_work_operation(
    operation_id: str,
    payload: WorkAbandonPayload,
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    if not payload.confirm:
        raise WorkValidationError("Confirm before continuing past uncertainty.")
    return _operation_payload(await _work(services).abandon_operation(operation_id))


def _work(services: ApiServices) -> WorkApplicationPort:
    if services.work is None:
        raise WorkUnavailableError("Work Sessions is unavailable.")
    return services.work


def _bootstrap_payload(bootstrap: WorkBootstrap) -> JsonObject:
    return {
        "available": bootstrap.available,
        "reason": bootstrap.reason,
        "codex_version": bootstrap.codex_version,
        "recent_projects": list(bootstrap.recent_projects),
        "unresolved_creates": [
            _operation_payload(operation) for operation in bootstrap.unresolved_creates
        ],
        "event_generation": bootstrap.event_generation,
        "event_cursor": bootstrap.event_cursor,
    }


def _summary_payload(summary: WorkSessionSummary) -> JsonObject:
    return {
        "thread_id": summary.thread_id,
        "cwd": summary.cwd,
        "title": summary.title,
        "preview": summary.preview[:240],
        "status": summary.status.value,
        "revision": summary.revision,
        "registered_at_ms": summary.registered_at_ms,
        "updated_at_ms": summary.updated_at_ms,
        "project_available": summary.project_available,
        "session_available": summary.session_available,
    }


def _record_payload(record: WorkSessionRecord) -> JsonObject:
    return {
        "thread_id": record.thread_id,
        "cwd": record.cwd,
        "revision": record.revision,
        "settings": {
            "model": record.settings.model,
            "reasoning_effort": record.settings.reasoning_effort,
        },
    }


def _detail_payload(detail: WorkSessionDetail) -> JsonObject:
    return {
        "summary": _summary_payload(detail.summary),
        "settings": {
            "model": detail.settings.model,
            "reasoning_effort": detail.settings.reasoning_effort,
        },
        "controls": detail.controls,
        "turns": _turn_page_payload(detail.turns),
        "live_items": [_timeline_payload(item) for item in detail.live_items],
        "interactions": [
            _interaction_payload(interaction) for interaction in detail.interactions
        ],
        "event_cursor": detail.event_cursor,
    }


def _turn_page_payload(page: WorkTurnPage) -> JsonObject:
    return {
        "items": [_timeline_payload(item) for item in page.items],
        "next_cursor": page.next_cursor,
    }


def _timeline_payload(item: WorkTimelineItem) -> JsonObject:
    payload: JsonObject = {
        "thread_id": item.thread_id,
        "turn_id": item.turn_id,
        "item_id": item.item_id,
        "kind": item.kind,
        "status": item.status,
        "text": item.text,
        "payload": item.payload,
    }
    if item.text and item.kind in {"agentMessage", "reasoning", "plan"}:
        payload["html"] = render_admin_markdown(item.text)
    return payload


def _interaction_payload(interaction: WorkInteraction) -> JsonObject:
    return {
        "interaction_id": interaction.interaction_id,
        "thread_id": interaction.thread_id,
        "turn_id": interaction.turn_id,
        "kind": interaction.kind.value,
        "title": interaction.title,
        "payload": interaction.payload,
    }


def _operation_payload(
    acknowledgement: WorkOperationAcknowledgement,
) -> JsonObject:
    return {
        "operation_id": acknowledgement.operation_id,
        "kind": acknowledgement.kind.value,
        "state": acknowledgement.state.value,
        "thread_id": acknowledgement.thread_id,
        "turn_id": acknowledgement.turn_id,
        "error_code": acknowledgement.error_code,
        "error_message": acknowledgement.error_message,
    }


def _set_operation_location(
    response: Response,
    acknowledgement: WorkOperationAcknowledgement,
) -> None:
    response.headers["Location"] = (
        f"/admin/api/work/operations/{acknowledgement.operation_id}"
    )


def _sse_event(event: PublishedEvent) -> ServerSentEvent:
    return ServerSentEvent(event=event.event, id=str(event.id), data=event.data)


def _encode_cursor(cursor: tuple[int, str] | None) -> str | None:
    if cursor is None:
        return None
    raw = f"{cursor[0]}:{cursor[1]}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str | None) -> tuple[int, str] | None:
    if cursor is None:
        return None
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.b64decode(padded, altchars=b"-_", validate=True).decode()
        timestamp_text, separator, thread_id = raw.partition(":")
        timestamp = int(timestamp_text)
    except (ValueError, UnicodeError) as exc:
        raise WorkValidationError("Invalid Work session cursor.") from exc
    if not separator or timestamp < 0 or not thread_id:
        raise WorkValidationError("Invalid Work session cursor.")
    return timestamp, thread_id
