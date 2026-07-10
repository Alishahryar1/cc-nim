"""Ingress-owned HTTP request correlation."""

import uuid

from fastapi import Request, Response

REQUEST_ID_HEADER = "request-id"
OPENAI_REQUEST_ID_HEADER = "x-request-id"
_REQUEST_ID_STATE_ATTRIBUTE = "fcc_request_id"
_OPENAI_REQUEST_ID_PATHS = frozenset({"/v1/responses", "/v1/models"})


def new_request_id() -> str:
    """Return a new opaque FCC request identifier."""
    return f"req_{uuid.uuid4().hex}"


def set_request_id(request: Request, request_id: str) -> None:
    """Attach the ingress correlation identifier to request state."""
    setattr(request.state, _REQUEST_ID_STATE_ATTRIBUTE, request_id)


def get_request_id(request: Request) -> str:
    """Return the ingress correlation identifier, creating a fallback if needed."""
    request_id = getattr(request.state, _REQUEST_ID_STATE_ATTRIBUTE, None)
    if isinstance(request_id, str) and request_id:
        return request_id
    request_id = new_request_id()
    set_request_id(request, request_id)
    return request_id


def attach_request_id_headers(
    response: Response, *, request_id: str, path: str
) -> None:
    """Expose FCC correlation using the public protocol header names."""
    response.headers[REQUEST_ID_HEADER] = request_id
    if path in _OPENAI_REQUEST_ID_PATHS:
        response.headers[OPENAI_REQUEST_ID_HEADER] = request_id
