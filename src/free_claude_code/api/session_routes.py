"""Local-only Admin APIs for native browser terminal sessions."""

import asyncio
import json
from collections.abc import Awaitable

from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket
from pydantic import BaseModel, ConfigDict, Field
from starlette.websockets import WebSocketDisconnect

from free_claude_code.application.browser_sessions import (
    BrowserSessionConflictError,
    BrowserSessionHarness,
    BrowserSessionNotFoundError,
    BrowserSessionUnavailableError,
    BrowserSessionValidationError,
    BrowserTerminalPort,
)

from .admin_routes import require_loopback_admin
from .dependencies import get_services
from .ports import ApiServices

router = APIRouter()
MAX_INPUT_BYTES = 64 * 1024
MIN_COLUMNS = 20
MAX_COLUMNS = 400
MIN_ROWS = 5
MAX_ROWS = 200


class _StrictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionPayload(_StrictPayload):
    path: str = Field(min_length=1)
    harness: BrowserSessionHarness
    name: str | None = Field(default=None, max_length=80)


class RenamePayload(_StrictPayload):
    name: str = Field(min_length=1, max_length=80)


@router.get("/admin/api/sessions")
async def sessions_snapshot(
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    return (await services.sessions.snapshot()).as_dict()


@router.post("/admin/api/sessions", status_code=201)
async def create_session(
    payload: SessionPayload,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    session = await _domain_call(
        services.sessions.create_session(payload.path, payload.harness, payload.name)
    )
    return session.as_dict()


@router.patch("/admin/api/sessions/{session_id}")
async def rename_session(
    session_id: str,
    payload: RenamePayload,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    session = await _domain_call(
        services.sessions.rename_session(session_id, payload.name)
    )
    return session.as_dict()


@router.post("/admin/api/sessions/{session_id}/start")
async def start_session(
    session_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    session = await _domain_call(services.sessions.start_session(session_id))
    return session.as_dict()


@router.post("/admin/api/sessions/{session_id}/stop")
async def stop_session(
    session_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    session = await _domain_call(services.sessions.stop_session(session_id))
    return session.as_dict()


@router.delete("/admin/api/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    await _domain_call(services.sessions.delete_session(session_id))
    return Response(status_code=204)


@router.websocket("/admin/api/sessions/{session_id}/terminal")
async def terminal(
    websocket: WebSocket,
    session_id: str,
):
    try:
        require_loopback_admin(websocket)
    except HTTPException:
        await websocket.close(code=1008)
        return
    services: ApiServices = websocket.app.state.services
    try:
        attachment = await services.sessions.attach_terminal(session_id)
    except BrowserSessionNotFoundError:
        await websocket.close(code=4404)
        return
    except BrowserSessionConflictError, BrowserSessionUnavailableError:
        await websocket.close(code=1013)
        return

    try:
        await websocket.accept()
    except Exception:
        await attachment.close()
        raise
    sender = asyncio.create_task(_send_terminal_events(websocket, attachment))
    receiver = asyncio.create_task(_receive_terminal_input(websocket, attachment))
    try:
        await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        # A cancelled WebSocket handler is a browser detach, not a process stop.
        pass
    finally:
        for task in (sender, receiver):
            if not task.done():
                task.cancel()
        await asyncio.gather(sender, receiver, return_exceptions=True)
        await attachment.close()


async def _send_terminal_events(
    websocket: WebSocket, attachment: BrowserTerminalPort
) -> None:
    try:
        while True:
            event = await attachment.receive()
            if event.kind == "output":
                await websocket.send_bytes(event.data or b"")
                continue
            await websocket.send_json(
                {
                    "type": event.kind,
                    "state": event.state.value if event.state is not None else None,
                    "detail": event.detail,
                }
            )
            if event.kind in {"replaced", "deleted", "error"}:
                return
    except WebSocketDisconnect:
        return


async def _receive_terminal_input(
    websocket: WebSocket, attachment: BrowserTerminalPort
) -> None:
    try:
        while True:
            raw = await websocket.receive_text()
            if len(raw.encode("utf-8")) > MAX_INPUT_BYTES:
                await websocket.close(code=1008)
                return
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.close(code=1008)
                return
            if not isinstance(payload, dict):
                await websocket.close(code=1008)
                return
            if payload.get("type") == "input" and isinstance(payload.get("data"), str):
                await attachment.write(payload["data"])
                continue
            if payload.get("type") == "resize":
                columns = payload.get("columns")
                rows = payload.get("rows")
                if (
                    isinstance(columns, int)
                    and not isinstance(columns, bool)
                    and MIN_COLUMNS <= columns <= MAX_COLUMNS
                    and isinstance(rows, int)
                    and not isinstance(rows, bool)
                    and MIN_ROWS <= rows <= MAX_ROWS
                ):
                    await attachment.resize(columns, rows)
                    continue
            await websocket.close(code=1008)
            return
    except WebSocketDisconnect, BrowserSessionConflictError:
        return


async def _domain_call[T](operation: Awaitable[T]) -> T:
    try:
        return await operation
    except BrowserSessionValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except BrowserSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BrowserSessionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except BrowserSessionUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
