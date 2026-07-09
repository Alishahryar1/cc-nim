from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from providers.exceptions import RateLimitError
from providers.nvidia_nim import NvidiaNimProvider

app = create_app()

# Mock provider
mock_provider = MagicMock(spec=NvidiaNimProvider)

# Track stream_response calls for test_model_mapping
_stream_response_calls: list = []


async def _mock_stream_response(*args, **kwargs):
    """Minimal async generator for streaming tests."""
    _stream_response_calls.append((args, kwargs))
    yield "event: message_start\ndata: {}\n\n"
    yield "[DONE]\n\n"


async def _mock_pre_start_rate_limit(*args, **kwargs):
    """Provider stream that fails before any downstream-visible SSE chunk."""
    _stream_response_calls.append((args, kwargs))
    raise RateLimitError("upstream is busy")
    yield "unreachable"


mock_provider.stream_response = _mock_stream_response


@pytest.fixture(scope="module")
def client():
    """HTTP client with provider resolution stubbed; patch only for this file."""
    with (
        patch("api.dependencies.resolve_provider", return_value=mock_provider),
        patch(
            "providers.runtime.ProviderRuntime.validate_configured_models",
            new_callable=AsyncMock,
        ),
        patch("providers.runtime.ProviderRuntime.start_model_list_refresh"),
        TestClient(app) as test_client,
    ):
        yield test_client


def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_models_list(client: TestClient):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["has_more"] is False
    ids = [item["id"] for item in data["data"]]
    assert "claude-sonnet-4-20250514" in ids
    assert data["first_id"] == ids[0]
    assert data["last_id"] == ids[-1]


def test_probe_endpoints_return_204_with_allow_headers(client: TestClient):
    responses = [
        client.head("/"),
        client.options("/"),
        client.head("/health"),
        client.options("/health"),
        client.head("/v1/messages"),
        client.options("/v1/messages"),
        client.head("/v1/messages/count_tokens"),
        client.options("/v1/messages/count_tokens"),
    ]

    for response in responses:
        assert response.status_code == 204
        assert "Allow" in response.headers


def test_create_message_stream(client: TestClient):
    """Create message returns streaming response."""
    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        "stream": True,
    }
    response = client.post("/v1/messages", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    content = b"".join(response.iter_bytes())
    assert b"message_start" in content or b"event:" in content


def test_create_message_pre_start_provider_error_returns_non_200_json(
    client: TestClient,
):
    """Pre-first-chunk provider errors should not commit HTTP 200."""
    mock_provider.stream_response = _mock_pre_start_rate_limit
    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        "stream": True,
    }

    response = client.post("/v1/messages", json=payload)

    assert response.status_code == 429
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "type": "error",
        "error": {"type": "rate_limit_error", "message": "upstream is busy"},
    }
    mock_provider.stream_response = _mock_stream_response


def test_create_message_accepts_system_role_messages(client: TestClient):
    """Create message accepts latest-client system messages."""
    mock_provider.stream_response = _mock_stream_response
    _stream_response_calls.clear()
    payload = {
        "model": "claude-3-sonnet",
        "messages": [
            {"role": "user", "content": "context"},
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Hi"},
        ],
        "max_tokens": 100,
        "stream": True,
    }

    response = client.post("/v1/messages", json=payload)

    assert response.status_code == 200
    routed_request = _stream_response_calls[0][0][0]
    assert [message.role for message in routed_request.messages] == ["user", "user"]
    assert routed_request.system == "system prompt"


def test_model_mapping(client: TestClient):
    # Test Haiku mapping
    _stream_response_calls.clear()
    payload_haiku = {
        "model": "claude-3-haiku-20240307",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        "stream": True,
    }
    client.post("/v1/messages", json=payload_haiku)
    assert len(_stream_response_calls) == 1
    args = _stream_response_calls[0][0]
    kwargs = _stream_response_calls[0][1]
    assert args[0].model != "claude-3-haiku-20240307"
    assert kwargs["thinking_enabled"] is True


def test_error_fallbacks(client: TestClient):
    from providers.exceptions import (
        AuthenticationError,
        OverloadedError,
        RateLimitError,
    )

    base_payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
        "stream": True,
    }

    def _raise_auth(*args, **kwargs):
        raise AuthenticationError("Invalid Key")

    def _raise_rate_limit(*args, **kwargs):
        raise RateLimitError("Too Many Requests")

    def _raise_overloaded(*args, **kwargs):
        raise OverloadedError("Server Overloaded")

    # 1. Authentication Error (401)
    mock_provider.stream_response = _raise_auth
    response = client.post("/v1/messages", json=base_payload)
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"

    # 2. Rate Limit (429)
    mock_provider.stream_response = _raise_rate_limit
    response = client.post("/v1/messages", json=base_payload)
    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit_error"

    # 3. Overloaded (529)
    mock_provider.stream_response = _raise_overloaded
    response = client.post("/v1/messages", json=base_payload)
    assert response.status_code == 529
    assert response.json()["error"]["type"] == "overloaded_error"

    # Reset for subsequent tests
    mock_provider.stream_response = _mock_stream_response


def test_generic_exception_returns_500(client: TestClient):
    """Non-ProviderError exceptions are caught and returned as HTTPException(500)."""

    def _raise_runtime(*args, **kwargs):
        raise RuntimeError("unexpected crash")

    mock_provider.stream_response = _raise_runtime
    response = client.post(
        "/v1/messages",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "stream": True,
        },
    )
    assert response.status_code == 500
    mock_provider.stream_response = _mock_stream_response


def test_generic_exception_with_status_code(client: TestClient):
    """Unexpected errors always map to HTTP 500 (ignore ad-hoc status_code attrs)."""

    class ExceptionWithStatus(RuntimeError):
        def __init__(self, msg: str, status_code: int = 500):
            super().__init__(msg)
            self.status_code = status_code

    def _raise_with_status(*args, **kwargs):
        raise ExceptionWithStatus("bad gateway", 502)

    mock_provider.stream_response = _raise_with_status
    response = client.post(
        "/v1/messages",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "stream": True,
        },
    )
    assert response.status_code == 500
    mock_provider.stream_response = _mock_stream_response


def test_generic_exception_empty_message_returns_non_empty_detail(client: TestClient):
    """Exceptions with empty __str__ still return a readable HTTP detail."""

    class SilentError(RuntimeError):
        def __str__(self):
            return ""

    def _raise_silent(*args, **kwargs):
        raise SilentError()

    mock_provider.stream_response = _raise_silent
    response = client.post(
        "/v1/messages",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "stream": True,
        },
    )
    assert response.status_code == 500
    assert response.json()["detail"] != ""
    mock_provider.stream_response = _mock_stream_response


def test_count_tokens_endpoint(client: TestClient):
    """count_tokens endpoint returns token count."""
    response = client.post(
        "/v1/messages/count_tokens",
        json={"model": "test", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    assert "input_tokens" in response.json()


def test_stop_endpoint_no_workflow_no_cli_503(client: TestClient):
    """POST /stop without messaging workflow or cli_manager returns 503."""
    # Ensure no messaging workflow or cli_manager on app state
    if hasattr(app.state, "messaging_workflow"):
        delattr(app.state, "messaging_workflow")
    if hasattr(app.state, "cli_manager"):
        delattr(app.state, "cli_manager")
    response = client.post("/stop")
    assert response.status_code == 503
