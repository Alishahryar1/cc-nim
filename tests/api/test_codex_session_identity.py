import json
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from free_claude_code.api.codex_session_identity import (
    observe_correlated_codex_identity,
)
from free_claude_code.api.ports import ApiServices
from free_claude_code.application.browser_sessions import (
    BROWSER_SESSION_HEADER,
    BrowserSessionIdentityObservation,
)
from free_claude_code.application.errors import (
    ApplicationConflictError,
    ApplicationUnavailableError,
    InvalidRequestError,
)
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from tests.api.support import create_test_app

ROOT_ID = "019f0000-0000-7000-8000-000000000001"
THREAD_ID = "019f0000-0000-7000-8000-000000000002"
TURN_ID = "019f0000-0000-7000-8000-000000000010"
TOKEN = "opaque-launch-token"


class StubIdentityPort:
    def __init__(self, error: Exception | None = None) -> None:
        self.observations: list[BrowserSessionIdentityObservation] = []
        self.error = error

    async def observe_identity(
        self, observation: BrowserSessionIdentityObservation
    ) -> None:
        self.observations.append(observation)
        if self.error is not None:
            raise self.error


class StubRequestRuntime:
    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self.acquire_mock = AsyncMock(
            side_effect=AssertionError("provider lease must not be acquired")
        )

    async def acquire(self):
        return await self.acquire_mock()

    def current_settings(self):
        return self._delegate.current_settings()

    def cached_model_supports_thinking(self, provider_id: str, model_id: str):
        return self._delegate.cached_model_supports_thinking(provider_id, model_id)

    def cached_prefixed_model_infos(self):
        return self._delegate.cached_prefixed_model_infos()


def _canonical_metadata(
    request_kind: str,
    *,
    root_id: str = ROOT_ID,
    thread_id: str = THREAD_ID,
    turn_id: str = TURN_ID,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "request_kind": request_kind,
        "workspaces": {"project": {"has_changes": True}},
    }
    if request_kind != "memory":
        metadata.update(session_id=root_id, thread_id=thread_id)
    if request_kind == "turn":
        metadata["turn_id"] = turn_id
    return metadata


def _request_and_data(
    request_kind: str,
    *,
    canonical: dict[str, object] | None = None,
    metadata_overrides: dict[str, object] | None = None,
    header_overrides: dict[str, str] | None = None,
) -> tuple[Request, OpenAIResponsesRequest]:
    canonical = canonical or _canonical_metadata(request_kind)
    serialized = json.dumps(canonical, separators=(",", ":"))
    metadata: dict[str, object] = {
        "x-codex-turn-metadata": serialized,
        "session_id": ROOT_ID,
        "thread_id": THREAD_ID,
    }
    if request_kind == "turn":
        metadata["turn_id"] = TURN_ID
    if metadata_overrides:
        metadata.update(metadata_overrides)
    headers = {
        BROWSER_SESSION_HEADER: TOKEN,
        "x-codex-turn-metadata": serialized,
        "session-id": ROOT_ID,
        "thread-id": THREAD_ID,
    }
    if header_overrides:
        headers.update(header_overrides)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/responses",
            "headers": [
                (key.encode("ascii"), value.encode("ascii"))
                for key, value in headers.items()
            ],
        }
    )
    data = OpenAIResponsesRequest(
        model="nvidia_nim/test-model",
        input="Hello",
        client_metadata=metadata,
    )
    return request, data


@pytest.mark.asyncio
async def test_turn_correlation_cross_checks_metadata_and_observes_root() -> None:
    request, data = _request_and_data("turn")
    port = StubIdentityPort()

    correlated = await observe_correlated_codex_identity(request, data, port)

    assert correlated is True
    assert port.observations == [
        BrowserSessionIdentityObservation(
            binding_token=TOKEN,
            native_id=UUID(ROOT_ID),
            turn_id=UUID(TURN_ID),
        )
    ]


@pytest.mark.parametrize("request_kind", ["prewarm", "compaction", "memory"])
@pytest.mark.asyncio
async def test_non_turn_correlation_is_validate_only(request_kind: str) -> None:
    request, data = _request_and_data(request_kind)
    port = StubIdentityPort()

    await observe_correlated_codex_identity(request, data, port)

    assert port.observations == [BrowserSessionIdentityObservation(binding_token=TOKEN)]


@pytest.mark.asyncio
async def test_absent_private_header_preserves_existing_responses_path() -> None:
    request = Request(
        {"type": "http", "method": "POST", "path": "/v1/responses", "headers": []}
    )
    data = OpenAIResponsesRequest(model="nvidia_nim/test-model", input="Hello")
    port = StubIdentityPort()

    correlated = await observe_correlated_codex_identity(request, data, port)

    assert correlated is False
    assert port.observations == []


@pytest.mark.parametrize(
    ("request_kind", "canonical", "metadata_overrides", "header_overrides", "error"),
    [
        (
            "turn",
            _canonical_metadata("turn"),
            {"session_id": THREAD_ID},
            None,
            "session_id values do not match",
        ),
        (
            "turn",
            _canonical_metadata("turn", turn_id="019f0000-0000-4000-8000-000000000010"),
            {"turn_id": "019f0000-0000-4000-8000-000000000010"},
            None,
            "turn_id must be UUIDv7",
        ),
        (
            "unknown",
            {"request_kind": "unknown"},
            None,
            None,
            "request kind is missing or unsupported",
        ),
        (
            "memory",
            {"request_kind": "memory", "session_id": ROOT_ID},
            None,
            None,
            "memory metadata must be detached",
        ),
        (
            "memory",
            {"request_kind": "memory", "session_id": 42},
            None,
            None,
            "metadata values must be strings",
        ),
        (
            "memory",
            {"request_kind": "memory"},
            {"turn_id": TURN_ID},
            None,
            "memory metadata must be detached from a turn",
        ),
        (
            "turn",
            _canonical_metadata("turn"),
            None,
            {"x-codex-turn-metadata": "{}"},
            "metadata projections do not match",
        ),
    ],
)
@pytest.mark.asyncio
async def test_invalid_correlated_metadata_fails_before_observation(
    request_kind,
    canonical,
    metadata_overrides,
    header_overrides,
    error,
) -> None:
    request, data = _request_and_data(
        request_kind,
        canonical=canonical,
        metadata_overrides=metadata_overrides,
        header_overrides=header_overrides,
    )
    port = StubIdentityPort()

    with pytest.raises(InvalidRequestError, match=error):
        await observe_correlated_codex_identity(request, data, port)

    assert port.observations == []


@pytest.mark.parametrize(
    ("error", "status_code", "should_retry"),
    [
        (ApplicationConflictError("stale launch"), 409, "false"),
        (ApplicationUnavailableError("metadata unavailable"), 503, None),
    ],
)
def test_route_serializes_correlation_failures_before_provider_acquisition(
    error, status_code, should_retry
) -> None:
    base = create_test_app()
    services = base.state.services
    requests = StubRequestRuntime(services.requests)
    app = create_test_app()
    app.state.services = ApiServices(
        requests=requests,
        admin=app.state.services.admin,
        tasks=app.state.services.tasks,
        sessions=app.state.services.sessions,
        session_identities=StubIdentityPort(error),
    )
    request, data = _request_and_data("turn")
    payload = data.model_dump(mode="json", exclude_none=True)
    headers = {key.decode(): value.decode() for key, value in request.scope["headers"]}

    with TestClient(app) as client:
        response = client.post("/v1/responses", json=payload, headers=headers)

    assert response.status_code == status_code
    assert response.headers.get("x-should-retry") == should_retry
    assert response.json()["error"]["message"] == str(error)
    requests.acquire_mock.assert_not_awaited()


def test_route_rejects_malformed_correlation_before_provider_acquisition() -> None:
    base = create_test_app()
    services = base.state.services
    requests = StubRequestRuntime(services.requests)
    app = create_test_app()
    app.state.services = ApiServices(
        requests=requests,
        admin=app.state.services.admin,
        tasks=app.state.services.tasks,
        sessions=app.state.services.sessions,
        session_identities=StubIdentityPort(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            json={"model": "nvidia_nim/test-model", "input": "Hello"},
            headers={BROWSER_SESSION_HEADER: TOKEN},
        )

    assert response.status_code == 400
    assert response.headers["x-should-retry"] == "false"
    assert response.json()["error"]["type"] == "invalid_request_error"
    requests.acquire_mock.assert_not_awaited()
