"""Loopback-only HTTP and WebSocket adapters for Terminal Sessions."""

import asyncio
from contextlib import suppress
from typing import Literal

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError

from free_claude_code.application.terminal import (
    TerminalApplicationPort,
    TerminalAttachmentOverflowError,
    TerminalAttachmentPort,
    TerminalDeletedEvent,
    TerminalError,
    TerminalNotFoundError,
    TerminalOutputEvent,
    TerminalSession,
    TerminalStateEvent,
    TerminalUnavailableError,
    TerminalValidationError,
)
from free_claude_code.core.json_types import JsonObject

from .admin_routes import admin_page_response
from .admin_security import require_loopback_admin, require_loopback_admin_websocket
from .dependencies import get_services
from .ports import ApiServices

router = APIRouter()


class TerminalRenamePayload(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class TerminalResizeMessage(BaseModel):
    type: Literal["resize"]
    rows: int = Field(ge=1, le=500)
    columns: int = Field(ge=1, le=1_000)


@router.get("/admin/terminal", include_in_schema=False)
@router.get("/admin/terminal/{session_id}", include_in_schema=False)
def terminal_page(request: Request, session_id: str | None = None):
    del session_id
    require_loopback_admin(request)
    return admin_page_response()


@router.get("/admin/api/terminal/sessions")
async def list_terminal_sessions(
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    sessions = await _terminal(services).list_sessions()
    return {"sessions": [_session_payload(session) for session in sessions]}


@router.post("/admin/api/terminal/sessions", status_code=201)
async def create_terminal_session(
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    return _session_payload(await _terminal(services).create_session())


@router.get("/admin/api/terminal/sessions/{session_id}")
async def get_terminal_session(
    session_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    return _session_payload(await _terminal(services).get_session(session_id))


@router.patch("/admin/api/terminal/sessions/{session_id}")
async def rename_terminal_session(
    session_id: str,
    payload: TerminalRenamePayload,
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    return _session_payload(
        await _terminal(services).rename_session(session_id, payload.name)
    )


@router.post("/admin/api/terminal/sessions/{session_id}/stop")
async def stop_terminal_session(
    session_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    return _session_payload(await _terminal(services).stop_session(session_id))


@router.delete("/admin/api/terminal/sessions/{session_id}")
async def delete_terminal_session(
    session_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
) -> JsonObject:
    require_loopback_admin(request)
    await _terminal(services).delete_session(session_id)
    return {"deleted": True}


@router.websocket("/admin/api/terminal/sessions/{session_id}/attach")
async def attach_terminal(websocket: WebSocket, session_id: str) -> None:
    require_loopback_admin_websocket(websocket)
    services: ApiServices = websocket.app.state.services
    terminal = _terminal(services)
    try:
        attachment = await terminal.attach(session_id)
    except TerminalNotFoundError as exc:
        await websocket.close(code=4404, reason=str(exc))
        return
    except TerminalError as exc:
        await websocket.close(code=1011, reason=str(exc))
        return

    await websocket.accept()
    sender = asyncio.create_task(
        _send_terminal(websocket, attachment),
        name=f"terminal-websocket-send-{session_id}",
    )
    receiver = asyncio.create_task(
        _receive_terminal(websocket, terminal, session_id),
        name=f"terminal-websocket-receive-{session_id}",
    )
    try:
        done, pending = await asyncio.wait(
            {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in pending:
            with suppress(asyncio.CancelledError):
                await task
        failure: TerminalError | None = None
        for task in done:
            try:
                task.result()
            except WebSocketDisconnect:
                pass
            except TerminalError as exc:
                failure = exc
        if failure is not None:
            with suppress(RuntimeError, WebSocketDisconnect):
                await websocket.send_json({"type": "error", "message": str(failure)})
    except WebSocketDisconnect:
        pass
    finally:
        await attachment.aclose()


async def _send_terminal(
    websocket: WebSocket, attachment: TerminalAttachmentPort
) -> None:
    initial = attachment.initial
    await websocket.send_json(
        {"type": "attached", "session": _session_payload(initial.session)}
    )
    if initial.session.history_truncated:
        await websocket.send_json({"type": "history_truncated"})
    if initial.output:
        await websocket.send_bytes(initial.output)

    try:
        async for event in attachment:
            if isinstance(event, TerminalOutputEvent):
                await websocket.send_bytes(event.data)
            elif isinstance(event, TerminalStateEvent):
                await websocket.send_json(
                    {"type": "state", "session": _session_payload(event.session)}
                )
            elif isinstance(event, TerminalDeletedEvent):
                await websocket.send_json({"type": "deleted"})
                return
    except TerminalAttachmentOverflowError:
        await websocket.send_json({"type": "resync_required"})


async def _receive_terminal(
    websocket: WebSocket,
    terminal: TerminalApplicationPort,
    session_id: str,
) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        data = message.get("bytes")
        if data is not None:
            await terminal.write(session_id, data)
            continue
        text = message.get("text")
        if text is None:
            continue
        try:
            resize = TerminalResizeMessage.model_validate_json(text)
        except ValidationError as exc:
            raise TerminalValidationError("Invalid terminal control message.") from exc
        await terminal.resize(
            session_id,
            rows=resize.rows,
            columns=resize.columns,
        )


def _terminal(services: ApiServices) -> TerminalApplicationPort:
    if services.terminal is None:
        raise TerminalUnavailableError("Terminal Sessions is unavailable.")
    return services.terminal


def _session_payload(session: TerminalSession) -> JsonObject:
    return {
        "id": session.id,
        "name": session.name,
        "status": session.status.value,
        "created_at": session.created_at,
        "rows": session.rows,
        "columns": session.columns,
        "exit_code": session.exit_code,
        "error": session.error,
        "history_truncated": session.history_truncated,
    }
