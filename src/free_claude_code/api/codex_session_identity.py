"""Private Codex request correlation for browser-launched native sessions."""

import json
from collections.abc import Mapping
from uuid import UUID

from fastapi import Request

from free_claude_code.application.browser_sessions import (
    BROWSER_SESSION_HEADER,
    BrowserSessionIdentityObservation,
    BrowserSessionIdentityPort,
)
from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.openai_responses import OpenAIResponsesRequest

_TURN_METADATA_KEY = "x-codex-turn-metadata"
_IDENTITY_METADATA_KEYS = frozenset(
    {"request_kind", "session_id", "thread_id", "turn_id"}
)
_ROOTED_REQUEST_KINDS = frozenset({"prewarm", "turn", "compaction"})
_REQUEST_KINDS = _ROOTED_REQUEST_KINDS | {"memory"}


async def observe_correlated_codex_identity(
    request: Request,
    request_data: OpenAIResponsesRequest,
    identity_port: BrowserSessionIdentityPort,
) -> bool:
    """Consume FCC's private correlation contract before provider acquisition."""

    token = request.headers.get(BROWSER_SESSION_HEADER)
    if token is None:
        return False
    if not token:
        raise InvalidRequestError("Codex browser-session correlation is empty.")

    metadata = _string_mapping(request_data.client_metadata, "client_metadata")
    serialized = metadata.get(_TURN_METADATA_KEY)
    if serialized is None:
        raise InvalidRequestError("Codex turn metadata is missing.")
    header_metadata = request.headers.get(_TURN_METADATA_KEY)
    if header_metadata is None:
        raise InvalidRequestError("Codex turn metadata header is missing.")
    if header_metadata != serialized:
        raise InvalidRequestError("Codex turn metadata projections do not match.")
    try:
        canonical = json.loads(serialized)
    except json.JSONDecodeError as exc:
        raise InvalidRequestError("Codex turn metadata is not valid JSON.") from exc
    canonical = _string_mapping(canonical, "Codex turn metadata", allow_extra=True)
    request_kind = canonical.get("request_kind")
    if request_kind not in _REQUEST_KINDS:
        raise InvalidRequestError("Codex request kind is missing or unsupported.")

    if request_kind == "memory":
        _validate_memory_projection(request, metadata, canonical)
        observation = BrowserSessionIdentityObservation(binding_token=token)
    else:
        session_id = _matching_uuid(
            "session_id",
            canonical.get("session_id"),
            metadata.get("session_id"),
            request.headers.get("session-id"),
        )
        _matching_uuid(
            "thread_id",
            canonical.get("thread_id"),
            metadata.get("thread_id"),
            request.headers.get("thread-id"),
        )
        if request_kind == "turn":
            turn_id = _matching_uuid(
                "turn_id",
                canonical.get("turn_id"),
                metadata.get("turn_id"),
            )
            if turn_id.version != 7:
                raise InvalidRequestError("Codex turn_id must be UUIDv7.")
            observation = BrowserSessionIdentityObservation(
                binding_token=token,
                native_id=session_id,
                turn_id=turn_id,
            )
        else:
            observation = BrowserSessionIdentityObservation(binding_token=token)

    await identity_port.observe_identity(observation)
    return True


def _validate_memory_projection(
    request: Request,
    metadata: Mapping[str, str],
    canonical: Mapping[str, str],
) -> None:
    if any(
        canonical.get(key) is not None for key in ("session_id", "thread_id", "turn_id")
    ):
        raise InvalidRequestError("Codex memory metadata must be detached from a root.")
    if metadata.get("turn_id") is not None:
        raise InvalidRequestError("Codex memory metadata must be detached from a turn.")
    _matching_optional_uuid(
        "session_id", metadata.get("session_id"), request.headers.get("session-id")
    )
    _matching_optional_uuid(
        "thread_id", metadata.get("thread_id"), request.headers.get("thread-id")
    )


def _string_mapping(
    value: object,
    name: str,
    *,
    allow_extra: bool = False,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise InvalidRequestError(f"{name} must be an object.")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise InvalidRequestError(f"{name} keys must be strings.")
        if isinstance(item, str):
            result[key] = item
        elif not allow_extra or key in _IDENTITY_METADATA_KEYS:
            raise InvalidRequestError(f"{name} values must be strings.")
    return result


def _matching_uuid(name: str, *values: str | None) -> UUID:
    if any(value is None for value in values):
        raise InvalidRequestError(f"Codex {name} is missing.")
    parsed = tuple(_uuid(value, name) for value in values if value is not None)
    if len(set(parsed)) != 1:
        raise InvalidRequestError(f"Codex {name} values do not match.")
    return parsed[0]


def _matching_optional_uuid(name: str, *values: str | None) -> None:
    present = tuple(value for value in values if value is not None)
    if not present:
        return
    if len(present) != len(values):
        raise InvalidRequestError(f"Codex {name} projections are incomplete.")
    _matching_uuid(name, *present)


def _uuid(value: str, name: str) -> UUID:
    try:
        return UUID(value)
    except ValueError as exc:
        raise InvalidRequestError(f"Codex {name} is not a UUID.") from exc
