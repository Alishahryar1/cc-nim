"""Loopback-only HTTP adapter for local Chat Sessions."""

import base64
import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from free_claude_code.application.chat import (
    ChatApplicationPort,
    ChatCompaction,
    ChatContextEstimate,
    ChatModelOption,
    ChatOperationStream,
    ChatPreferences,
    ChatReasoning,
    ChatSession,
    ChatSessionSummary,
    ChatStreamEvent,
    ChatTurn,
    ChatUnavailableError,
    ChatValidationError,
)
from free_claude_code.core.json_types import JsonObject

from .admin_routes import admin_page_response
from .admin_security import require_loopback_admin
from .chat_markdown import render_chat_markdown
from .dependencies import get_services
from .ports import ApiServices
from .response_streams import ManagedStreamingResponse

router = APIRouter()


class ChatSessionUpdatePayload(BaseModel):
    expected_revision: int = Field(gt=0)
    title: str | None = None
    model: str | None = None
    reasoning: ChatReasoning | None = None


class ChatRevisionPayload(BaseModel):
    expected_revision: int = Field(gt=0)


class ChatOperationPayload(ChatRevisionPayload):
    operation_id: str


class ChatSendPayload(ChatOperationPayload):
    text: str = Field(max_length=1_000_000)


class ChatEstimatePayload(BaseModel):
    draft: str = Field(default="", max_length=1_000_000)


class ChatStopPayload(BaseModel):
    operation_id: str


class ChatPromptPayload(BaseModel):
    value: str = Field(max_length=100_000)


@router.get("/admin/chat", include_in_schema=False)
@router.get("/admin/chat/{session_id}", include_in_schema=False)
def chat_page(request: Request, session_id: str | None = None):
    del session_id
    require_loopback_admin(request)
    return admin_page_response()


@router.get("/admin/api/chat/bootstrap")
async def chat_bootstrap(
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    chat = _chat(services)
    available, message = chat.availability()
    payload: JsonObject = {
        "available": available,
        "message": message,
        "models": [_model_payload(option) for option in chat.models()],
        "preferences": None,
    }
    if available:
        payload["preferences"] = _preferences_payload(await chat.preferences())
    return payload


@router.get("/admin/api/chat/sessions")
async def list_chat_sessions(
    request: Request,
    query: str = Query(default="", max_length=200),
    cursor: str | None = None,
    limit: int = Query(default=25, ge=1, le=25),
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    page = await _chat(services).list_sessions(
        query=query,
        cursor=_decode_cursor(cursor),
        limit=limit,
    )
    return {
        "sessions": [_session_summary_payload(session) for session in page.sessions],
        "next_cursor": _encode_cursor(page.next_cursor),
    }


@router.post("/admin/api/chat/sessions", status_code=201)
async def create_chat_session(
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    return _session_payload(await _chat(services).create_session())


@router.get("/admin/api/chat/sessions/{session_id}")
async def get_chat_session(
    session_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    detail = await _chat(services).get_detail(session_id)
    return {
        "session": _session_payload(detail.session),
        "turns": [_turn_payload(turn) for turn in detail.turns],
        "next_before": detail.next_before,
        "compaction": _compaction_payload(detail.compaction),
        "context": (
            _estimate_payload(detail.context) if detail.context is not None else None
        ),
        "context_error": detail.context_error,
        "active_operation": detail.active_operation,
    }


@router.patch("/admin/api/chat/sessions/{session_id}")
async def update_chat_session(
    session_id: str,
    payload: ChatSessionUpdatePayload,
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    if payload.title is None and payload.model is None and payload.reasoning is None:
        raise ChatValidationError("Choose a chat field to update.")
    session = await _chat(services).update_session(
        session_id,
        expected_revision=payload.expected_revision,
        title=payload.title,
        model=payload.model,
        reasoning=payload.reasoning,
    )
    return _session_payload(session)


@router.delete("/admin/api/chat/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    payload: ChatRevisionPayload,
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    await _chat(services).delete_session(
        session_id,
        expected_revision=payload.expected_revision,
    )
    return {"deleted": True}


@router.get("/admin/api/chat/sessions/{session_id}/turns")
async def get_chat_turns(
    session_id: str,
    request: Request,
    before: int | None = Query(default=None, ge=1),
    limit: int = Query(default=50, ge=1, le=50),
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    turns, next_before, compaction = await _chat(services).get_turn_page(
        session_id,
        before_sequence=before,
        limit=limit,
    )
    return {
        "turns": [_turn_payload(turn) for turn in turns],
        "next_before": next_before,
        "compaction": _compaction_payload(compaction),
    }


@router.post("/admin/api/chat/sessions/{session_id}/estimate")
async def estimate_chat_context(
    session_id: str,
    payload: ChatEstimatePayload,
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    estimate = await _chat(services).estimate(session_id, draft=payload.draft)
    return _estimate_payload(estimate)


@router.post("/admin/api/chat/sessions/{session_id}/send")
async def send_chat_message(
    session_id: str,
    payload: ChatSendPayload,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    stream = await _chat(services).send(
        session_id,
        expected_revision=payload.expected_revision,
        operation_id=payload.operation_id,
        text=payload.text,
    )
    return _stream_response(stream)


@router.post("/admin/api/chat/sessions/{session_id}/retry")
async def retry_chat_message(
    session_id: str,
    payload: ChatOperationPayload,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    stream = await _chat(services).retry(
        session_id,
        expected_revision=payload.expected_revision,
        operation_id=payload.operation_id,
    )
    return _stream_response(stream)


@router.post("/admin/api/chat/sessions/{session_id}/regenerate")
async def regenerate_chat_message(
    session_id: str,
    payload: ChatOperationPayload,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    stream = await _chat(services).regenerate(
        session_id,
        expected_revision=payload.expected_revision,
        operation_id=payload.operation_id,
    )
    return _stream_response(stream)


@router.post("/admin/api/chat/sessions/{session_id}/compact")
async def compact_chat_session(
    session_id: str,
    payload: ChatOperationPayload,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    stream = await _chat(services).compact(
        session_id,
        expected_revision=payload.expected_revision,
        operation_id=payload.operation_id,
    )
    return _stream_response(stream)


@router.post("/admin/api/chat/sessions/{session_id}/stop")
async def stop_chat_operation(
    session_id: str,
    payload: ChatStopPayload,
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    stopped = await _chat(services).stop(
        session_id,
        operation_id=payload.operation_id,
    )
    return {"stopped": stopped}


@router.get("/admin/api/chat/preferences")
async def get_chat_preferences(
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    return _preferences_payload(await _chat(services).preferences())


@router.put("/admin/api/chat/preferences/system-prompt")
async def save_chat_system_prompt(
    payload: ChatPromptPayload,
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    return _preferences_payload(await _chat(services).save_system_prompt(payload.value))


@router.delete("/admin/api/chat/preferences/system-prompt")
async def reset_chat_system_prompt(
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    return _preferences_payload(await _chat(services).reset_system_prompt())


def _chat(services: ApiServices) -> ChatApplicationPort:
    if services.chat is None:
        raise ChatUnavailableError("Chat Sessions is unavailable.")
    return services.chat


def _stream_response(stream: ChatOperationStream) -> ManagedStreamingResponse:
    response = ManagedStreamingResponse(
        _stream_events(stream),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )
    response.bind_release(stream.aclose)
    return response


async def _stream_events(stream: ChatOperationStream) -> AsyncIterator[str]:
    try:
        async for event in stream:
            yield _sse_event(event)
    finally:
        await stream.aclose()


def _sse_event(event: ChatStreamEvent) -> str:
    data = json.dumps(event.data, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event.sequence}\nevent: {event.event}\ndata: {data}\n\n"


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
        updated_at_text, separator, session_id = raw.partition(":")
        updated_at = int(updated_at_text)
    except (ValueError, UnicodeError) as exc:
        raise ChatValidationError("Invalid session cursor.") from exc
    if not separator or updated_at < 0 or not _is_canonical_uuid(session_id):
        raise ChatValidationError("Invalid session cursor.")
    return updated_at, session_id


def _is_canonical_uuid(value: str) -> bool:
    try:
        parsed = uuid.UUID(value)
    except ValueError, AttributeError:
        return False
    return parsed.version == 4 and str(parsed) == value.lower()


def _preferences_payload(preferences: ChatPreferences) -> JsonObject:
    return {
        "system_prompt": preferences.system_prompt,
        "last_model": preferences.last_model,
        "last_reasoning": preferences.last_reasoning.value,
        "updated_at": preferences.updated_at,
    }


def _model_payload(option: ChatModelOption) -> JsonObject:
    return {
        "model_ref": option.model_ref,
        "provider_id": option.provider_id,
        "model_id": option.model_id,
        "supports_reasoning": option.supports_reasoning,
        "input_modalities": (
            sorted(modality.value for modality in option.input_modalities)
            if option.input_modalities is not None
            else None
        ),
        "context_window_tokens": option.context_window_tokens,
        "max_output_tokens": option.max_output_tokens,
    }


def _session_payload(session: ChatSession) -> JsonObject:
    return {
        "id": session.id,
        "title": session.title,
        "model": session.model,
        "reasoning": session.reasoning.value,
        "revision": session.revision,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


def _session_summary_payload(session: ChatSessionSummary) -> JsonObject:
    return {
        **_session_payload(
            ChatSession(
                id=session.id,
                title=session.title,
                model=session.model,
                reasoning=session.reasoning,
                revision=session.revision,
                created_at=session.created_at,
                updated_at=session.updated_at,
            )
        ),
        "preview": session.preview[:240],
    }


def _turn_payload(turn: ChatTurn) -> JsonObject:
    generation = turn.generation
    return {
        "id": turn.id,
        "session_id": turn.session_id,
        "operation_id": turn.operation_id,
        "sequence": turn.sequence,
        "user_text": turn.user_text,
        "created_at": turn.created_at,
        "generation": {
            "id": generation.id,
            "status": generation.status.value,
            "requested_model": generation.requested_model,
            "actual_model": generation.actual_model,
            "reasoning": generation.reasoning.value,
            "effective_output_limit": generation.effective_output_limit,
            "stop_reason": generation.stop_reason,
            "error_code": generation.error_code,
            "error_message": generation.error_message,
            "started_at": generation.started_at,
            "finished_at": generation.finished_at,
            "segments": [
                {
                    "ordinal": segment.ordinal,
                    "kind": segment.kind.value,
                    "text": segment.text,
                    "html": render_chat_markdown(segment.text),
                }
                for segment in generation.segments
            ],
        },
    }


def _compaction_payload(compaction: ChatCompaction | None) -> JsonObject | None:
    if compaction is None:
        return None
    return {
        "covered_through_sequence": compaction.covered_through_sequence,
        "summary": compaction.summary,
        "summary_html": render_chat_markdown(compaction.summary),
        "estimated_tokens": compaction.estimated_tokens,
        "requested_model": compaction.requested_model,
        "actual_model": compaction.actual_model,
        "updated_at": compaction.updated_at,
    }


def _estimate_payload(estimate: ChatContextEstimate) -> JsonObject:
    return {
        "estimated_input_tokens": estimate.estimated_input_tokens,
        "completion_tokens": estimate.completion_tokens,
        "context_window_tokens": estimate.context_window_tokens,
        "usable_input_tokens": estimate.usable_input_tokens,
        "usage_ratio": estimate.usage_ratio,
        "should_auto_compact": estimate.should_auto_compact,
        "can_compact": estimate.can_compact,
    }
