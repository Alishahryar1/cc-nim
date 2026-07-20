"""Public commit-boundary behavior for canonical execution failures."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.anthropic.streaming import format_sse_event
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from tests.api.support import create_test_app

_PARTIAL_CONTENT = "PARTIAL_ASSISTANT_CONTENT"


class CanonicalFailureProvider:
    """Provider double that raises one request-correlated canonical failure."""

    def __init__(
        self,
        chunks: list[str],
        *,
        kind: FailureKind,
        status_code: int,
        message: str,
        retryable: bool,
        grouped: bool = False,
    ) -> None:
        self._chunks = chunks
        self._kind = kind
        self._status_code = status_code
        self._message = message
        self._retryable = retryable
        self._grouped = grouped
        self.preflight_stream = MagicMock()
        self.stream_kwargs: list[dict[str, Any]] = []

    async def stream_response(
        self,
        _request: object,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        self.stream_kwargs.append(kwargs)
        for chunk in self._chunks:
            yield chunk
        request_id = kwargs["request_id"]
        failure = ExecutionFailure(
            kind=self._kind,
            status_code=self._status_code,
            message=f"{self._message}\n\nRequest ID: {request_id}",
            retryable=self._retryable,
        )
        if self._grouped:
            raise ExceptionGroup(
                "provider stream and cleanup failed",
                [
                    RuntimeError("cleanup failed"),
                    ExceptionGroup("provider request failed", [failure]),
                ],
            )
        raise failure


def _messages_payload(*, stream: bool) -> dict[str, object]:
    return {
        "model": "nvidia_nim/test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
        "stream": stream,
    }


def _responses_payload() -> dict[str, object]:
    return {
        "model": "nvidia_nim/test-model",
        "input": "Hello",
        "max_output_tokens": 32,
    }


def _partial_anthropic_stream(*, close_block: bool) -> list[str]:
    chunks = [
        format_sse_event("message_start", {"type": "message_start", "message": {}}),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": _PARTIAL_CONTENT},
            },
        ),
    ]
    if close_block:
        chunks.append(
            format_sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": 0},
            )
        )
    return chunks


def _client_for(provider: CanonicalFailureProvider):
    app = create_test_app()
    return (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app),
    )


def _terminal_trace(trace_mock: MagicMock) -> dict[str, Any]:
    return dict(
        next(
            call.kwargs
            for call in trace_mock.call_args_list
            if call.kwargs.get("event")
            == "free_claude_code.api.response.terminal_execution_error"
        )
    )


def _grouped_rate_limit_provider(chunks: list[str]) -> CanonicalFailureProvider:
    return CanonicalFailureProvider(
        chunks,
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="upstream is busy",
        retryable=True,
        grouped=True,
    )


@pytest.mark.parametrize(
    ("path", "payload", "expected_type"),
    [
        ("/v1/messages", _messages_payload(stream=True), "rate_limit_error"),
        ("/v1/responses", _responses_payload(), "rate_limit_error"),
    ],
)
def test_grouped_pre_start_execution_failure_keeps_canonical_wire_error(
    path: str,
    payload: dict[str, object],
    expected_type: str,
) -> None:
    provider = _grouped_rate_limit_provider([])
    resolver_patch, client = _client_for(provider)

    with (
        resolver_patch,
        patch("free_claude_code.api.response_streams.trace_event") as trace_mock,
        client,
    ):
        response = client.post(path, json=payload)

    request_id = response.headers["request-id"]
    assert response.status_code == 429
    assert response.headers["x-should-retry"] == "false"
    error = response.json()["error"]
    assert error["type"] == expected_type
    assert error["message"] == f"upstream is busy\n\nRequest ID: {request_id}"
    trace = _terminal_trace(trace_mock)
    assert trace["status_code"] == 429
    assert trace["error_type"] == "rate_limit_error"
    assert trace["exc_type"] == "ExecutionFailure"
    assert trace["failure_kind"] == "rate_limit"


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/responses"])
def test_grouped_post_start_execution_failure_keeps_canonical_terminal_event(
    path: str,
) -> None:
    provider = _grouped_rate_limit_provider(_partial_anthropic_stream(close_block=True))
    payload = (
        _messages_payload(stream=True)
        if path == "/v1/messages"
        else _responses_payload()
    )
    resolver_patch, client = _client_for(provider)

    with (
        resolver_patch,
        patch("free_claude_code.api.response_streams.trace_event") as trace_mock,
        client,
    ):
        response = client.post(path, json=payload)

    request_id = response.headers["request-id"]
    events = parse_sse_text(response.text)
    if path == "/v1/messages":
        assert events[-1].event == "error"
        error = events[-1].data["error"]
    else:
        assert events[-1].event == "response.failed"
        error = events[-1].data["response"]["error"]
    assert response.status_code == 200
    assert error["type"] == "rate_limit_error"
    assert error["message"] == f"upstream is busy\n\nRequest ID: {request_id}"
    assert _terminal_trace(trace_mock)["failure_kind"] == "rate_limit"


def test_grouped_stream_false_execution_failure_discards_partial_content() -> None:
    provider = _grouped_rate_limit_provider(
        _partial_anthropic_stream(close_block=False)
    )
    resolver_patch, client = _client_for(provider)

    with (
        resolver_patch,
        patch("free_claude_code.api.response_streams.trace_event") as trace_mock,
        client,
    ):
        response = client.post("/v1/messages", json=_messages_payload(stream=False))

    request_id = response.headers["request-id"]
    assert response.status_code == 429
    assert response.headers["x-should-retry"] == "false"
    assert response.json()["error"] == {
        "type": "rate_limit_error",
        "message": f"upstream is busy\n\nRequest ID: {request_id}",
    }
    assert _PARTIAL_CONTENT not in response.text
    trace = _terminal_trace(trace_mock)
    assert trace["status_code"] == 429
    assert trace["error_type"] == "rate_limit_error"
    assert trace["exc_type"] == "ExecutionFailure"
    assert trace["failure_kind"] == "rate_limit"


def test_messages_pre_start_execution_failure_is_correlated_terminal_json() -> None:
    provider = CanonicalFailureProvider(
        [],
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="upstream is busy",
        retryable=True,
    )
    resolver_patch, client = _client_for(provider)

    with resolver_patch, client:
        response = client.post("/v1/messages", json=_messages_payload(stream=True))

    request_id = response.headers["request-id"]
    assert response.status_code == 429
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-should-retry"] == "false"
    assert "x-request-id" not in response.headers
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": f"upstream is busy\n\nRequest ID: {request_id}",
        },
        "request_id": request_id,
    }
    assert provider.stream_kwargs[0]["request_id"] == request_id


def test_messages_context_window_failure_triggers_client_compaction() -> None:
    provider = CanonicalFailureProvider(
        [],
        kind=FailureKind.CONTEXT_WINDOW_EXCEEDED,
        status_code=400,
        message="Provider input exceeds the model context window.",
        retryable=False,
    )
    resolver_patch, client = _client_for(provider)

    with (
        resolver_patch,
        patch("free_claude_code.api.response_streams.trace_event") as trace_mock,
        client,
    ):
        response = client.post("/v1/messages", json=_messages_payload(stream=True))

    request_id = response.headers["request-id"]
    assert response.status_code == 400
    assert response.headers["x-should-retry"] == "false"
    assert response.json() == {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": (
                "prompt is too long\n\n"
                "Provider input exceeds the model context window.\n\n"
                f"Request ID: {request_id}"
            ),
        },
        "request_id": request_id,
    }
    trace = _terminal_trace(trace_mock)
    assert trace["failure_kind"] == "context_window_exceeded"
    assert trace["status_code"] == 400
    assert trace["error_type"] == "invalid_request_error"
    assert trace["provider_retryable"] is False


def test_responses_pre_start_execution_failure_is_correlated_terminal_json() -> None:
    provider = CanonicalFailureProvider(
        [],
        kind=FailureKind.OVERLOADED,
        status_code=529,
        message="provider overloaded",
        retryable=True,
    )
    resolver_patch, client = _client_for(provider)

    with resolver_patch, client:
        response = client.post("/v1/responses", json=_responses_payload())

    request_id = response.headers["request-id"]
    assert response.status_code == 529
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-should-retry"] == "false"
    assert response.headers["x-request-id"] == request_id
    assert response.json() == {
        "error": {
            "message": f"provider overloaded\n\nRequest ID: {request_id}",
            "type": "overloaded_error",
            "param": None,
            "code": None,
        }
    }
    assert provider.stream_kwargs[0]["request_id"] == request_id


def test_messages_post_start_execution_failure_follows_closed_block() -> None:
    provider = CanonicalFailureProvider(
        _partial_anthropic_stream(close_block=True),
        kind=FailureKind.OVERLOADED,
        status_code=529,
        message="provider overloaded",
        retryable=True,
    )
    resolver_patch, client = _client_for(provider)

    with (
        resolver_patch,
        patch("free_claude_code.api.response_streams.trace_event") as trace_mock,
        client,
    ):
        response = client.post("/v1/messages", json=_messages_payload(stream=True))

    request_id = response.headers["request-id"]
    events = parse_sse_text(response.text)
    assert response.status_code == 200
    assert "x-should-retry" not in response.headers
    assert [event.event for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "error",
    ]
    assert events[-1].data["error"] == {
        "type": "overloaded_error",
        "message": f"provider overloaded\n\nRequest ID: {request_id}",
    }
    assert "message_stop" not in response.text
    assert _terminal_trace(trace_mock) == {
        "stage": "egress",
        "event": "free_claude_code.api.response.terminal_execution_error",
        "source": "api",
        "wire_api": "messages",
        "request_id": request_id,
        "status_code": 529,
        "error_type": "overloaded_error",
        "client_should_retry": False,
        "exc_type": "ExecutionFailure",
        "failure_kind": "overloaded",
        "provider_retryable": True,
    }


def test_responses_post_start_execution_failure_retains_id_after_block_close() -> None:
    provider = CanonicalFailureProvider(
        _partial_anthropic_stream(close_block=True),
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="upstream is busy",
        retryable=True,
    )
    resolver_patch, client = _client_for(provider)

    with (
        resolver_patch,
        patch("free_claude_code.api.response_streams.trace_event") as trace_mock,
        client,
    ):
        response = client.post("/v1/responses", json=_responses_payload())

    request_id = response.headers["request-id"]
    events = parse_sse_text(response.text)
    event_names = [event.event for event in events]
    created = events[0].data["response"]
    failed = events[-1].data["response"]
    assert response.status_code == 200
    assert response.headers["x-request-id"] == request_id
    assert "x-should-retry" not in response.headers
    assert event_names[0] == "response.created"
    assert "response.output_item.done" in event_names
    assert event_names.index("response.output_item.done") < event_names.index(
        "response.failed"
    )
    assert event_names[-1] == "response.failed"
    assert failed["id"] == created["id"]
    assert failed["status"] == "failed"
    assert failed["error"] == {
        "message": f"upstream is busy\n\nRequest ID: {request_id}",
        "type": "rate_limit_error",
        "param": None,
        "code": None,
    }
    assert _terminal_trace(trace_mock) == {
        "stage": "egress",
        "event": "free_claude_code.api.response.terminal_execution_error",
        "source": "api",
        "wire_api": "responses",
        "request_id": request_id,
        "status_code": 429,
        "error_type": "rate_limit_error",
        "client_should_retry": False,
        "exc_type": "ExecutionFailure",
        "failure_kind": "rate_limit",
        "provider_retryable": True,
    }


def test_messages_stream_false_discards_partial_content_on_execution_failure() -> None:
    provider = CanonicalFailureProvider(
        _partial_anthropic_stream(close_block=False),
        kind=FailureKind.RATE_LIMIT,
        status_code=429,
        message="upstream is busy",
        retryable=True,
    )
    resolver_patch, client = _client_for(provider)

    with resolver_patch, client:
        response = client.post("/v1/messages", json=_messages_payload(stream=False))

    request_id = response.headers["request-id"]
    assert response.status_code == 429
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-should-retry"] == "false"
    assert response.json()["request_id"] == request_id
    assert response.json()["error"] == {
        "type": "rate_limit_error",
        "message": f"upstream is busy\n\nRequest ID: {request_id}",
    }
    assert _PARTIAL_CONTENT not in response.text
