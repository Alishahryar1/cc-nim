from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from core.anthropic.stream_contracts import parse_sse_text
from core.anthropic.streaming import format_sse_event


class FakeProvider:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.preflight_stream = MagicMock()
        self.requests: list[Any] = []
        self.stream_kwargs: list[dict[str, Any]] = []

    async def stream_response(self, request_data, **_kwargs):
        self.requests.append(request_data)
        self.stream_kwargs.append(_kwargs)
        for chunk in self.chunks:
            yield chunk


@pytest.fixture
def responses_client():
    provider = FakeProvider(_anthropic_text_stream("Hello from provider"))
    app = create_app(lifespan_enabled=False)
    with (
        patch("api.dependencies.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        yield client, provider


def test_responses_probe_endpoints_return_204(
    responses_client: tuple[TestClient, FakeProvider],
) -> None:
    client, _provider = responses_client

    assert client.head("/v1/responses").status_code == 204
    assert client.options("/v1/responses").status_code == 204


def test_create_response_stream_routes_through_provider(
    responses_client: tuple[TestClient, FakeProvider],
) -> None:
    client, provider = responses_client

    response = client.post(
        "/v1/responses",
        json={
            "model": "nvidia_nim/test-model",
            "input": "Hello",
            "max_output_tokens": 32,
        },
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    events = parse_sse_text(response.text)
    assert events[0].event == "response.created"
    assert events[-1].event == "response.completed"
    assert events[-1].data["response"]["output"][0]["content"][0]["text"] == (
        "Hello from provider"
    )
    assert provider.preflight_stream.called
    routed = provider.requests[0]
    assert routed.model == "test-model"
    assert routed.messages[0].role == "user"
    assert routed.messages[0].content == "Hello"
    assert routed.max_tokens == 32


def test_create_response_stream_bypasses_local_message_optimizations() -> None:
    provider = FakeProvider(_anthropic_text_stream("Provider response"))
    app = create_app(lifespan_enabled=False)
    with (
        patch("api.dependencies.resolve_provider", return_value=provider),
        patch(
            "api.handlers.messages.try_optimizations",
            side_effect=AssertionError("Responses must not use message optimizations"),
        ),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "nvidia_nim/test-model",
                "input": "quota check",
            },
        )

    assert response.status_code == 200
    completed = parse_sse_text(response.text)[-1].data["response"]
    assert completed["output"][0]["content"][0]["text"] == "Provider response"
    assert provider.requests[0].messages[0].content == "quota check"


def test_create_response_stream_false_returns_openai_error(
    responses_client: tuple[TestClient, FakeProvider],
) -> None:
    client, provider = responses_client

    response = client.post(
        "/v1/responses",
        json={
            "model": "nvidia_nim/test-model",
            "input": "Hello",
            "stream": False,
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert "streaming only" in payload["error"]["message"]
    assert provider.requests == []


def test_create_response_stream_preserves_interleaved_reasoning_order() -> None:
    provider = FakeProvider(_anthropic_interleaved_reasoning_stream())
    app = create_app(lifespan_enabled=False)
    with (
        patch("api.dependencies.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "nvidia_nim/test-model",
                "input": "Use reasoning and tools",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "echo",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            },
        )

    assert response.status_code == 200
    events = parse_sse_text(response.text)
    assert "response.reasoning_text.delta" in [event.event for event in events]
    completed = events[-1].data["response"]
    assert [item["type"] for item in completed["output"]] == [
        "reasoning",
        "message",
        "function_call",
        "reasoning",
        "message",
    ]
    assert completed["output"][0]["content"][0]["text"] == "first thought"
    assert completed["output"][1]["content"][0]["text"] == "first answer"
    assert completed["output"][2]["arguments"] == '{"value":"FCC"}'
    assert completed["output"][3]["content"][0]["text"] == "second thought"
    assert completed["output"][4]["content"][0]["text"] == "final answer"


def test_create_response_tool_stream_emits_function_call() -> None:
    provider = FakeProvider(_anthropic_tool_stream())
    app = create_app(lifespan_enabled=False)
    with (
        patch("api.dependencies.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "nvidia_nim/test-model",
                "input": "Use echo",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "echo",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            },
        )

    assert response.status_code == 200
    events = parse_sse_text(response.text)
    completed = events[-1].data["response"]
    call = completed["output"][0]
    assert call["type"] == "function_call"
    assert call["call_id"] == "toolu_1"
    assert call["arguments"] == '{"value":"FCC"}'


def test_create_response_malformed_provider_function_call_fails_stream() -> None:
    provider = FakeProvider(
        _anthropic_tool_stream(partial_json='{"value":"FCC" "bad"}')
    )
    app = create_app(lifespan_enabled=False)
    with (
        patch("api.dependencies.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "nvidia_nim/test-model",
                "input": "Use echo",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "echo",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            },
        )

    assert response.status_code == 200
    events = parse_sse_text(response.text)
    assert events[-1].event == "response.failed"
    failed = events[-1].data["response"]
    assert failed["status"] == "failed"
    assert failed["output"] == []
    assert "replay-unsafe Responses output" in failed["error"]["message"]


def test_create_response_accepts_codex_namespace_tool_request() -> None:
    provider = FakeProvider(_anthropic_tool_stream(tool_name="mcp__node_repl__js"))
    app = create_app(lifespan_enabled=False)
    with (
        patch("api.dependencies.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "nvidia_nim/test-model",
                "input": "Use JS",
                "stream": True,
                "tools": [
                    {"type": "web_search", "external_web_access": True},
                    {"type": "image_generation", "output_format": "png"},
                    {
                        "type": "namespace",
                        "name": "mcp__node_repl",
                        "tools": [
                            {
                                "type": "function",
                                "name": "js",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"code": {"type": "string"}},
                                },
                            }
                        ],
                    },
                ],
            },
        )

    assert response.status_code == 200
    routed = provider.requests[0]
    assert [tool.name for tool in routed.tools] == ["mcp__node_repl__js"]
    completed = parse_sse_text(response.text)[-1].data["response"]
    call = completed["output"][0]
    assert call["namespace"] == "mcp__node_repl"
    assert call["name"] == "js"


def test_create_response_accepts_codex_custom_tool_request() -> None:
    provider = FakeProvider(
        _anthropic_tool_stream(
            tool_name="apply_patch",
            partial_json='{"input":"*** Begin Patch"}',
        )
    )
    app = create_app(lifespan_enabled=False)
    with (
        patch("api.dependencies.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "nvidia_nim/test-model",
                "input": "Use apply_patch",
                "stream": True,
                "tools": [
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "description": "Apply repo patches",
                        "format": {"type": "text"},
                    }
                ],
                "tool_choice": {"type": "custom", "name": "apply_patch"},
            },
        )

    assert response.status_code == 200
    routed = provider.requests[0]
    assert [tool.name for tool in routed.tools] == ["apply_patch"]
    assert routed.tool_choice == {"type": "tool", "name": "apply_patch"}
    events = parse_sse_text(response.text)
    assert "response.custom_tool_call_input.delta" in [event.event for event in events]
    completed = events[-1].data["response"]
    call = completed["output"][0]
    assert call["type"] == "custom_tool_call"
    assert call["name"] == "apply_patch"
    assert call["input"] == "*** Begin Patch"


def test_create_response_stream_provider_error_returns_response_failed() -> None:
    provider = FakeProvider(
        [
            format_sse_event(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": "provider failed",
                    },
                },
            )
        ]
    )
    app = create_app(lifespan_enabled=False)
    with (
        patch("api.dependencies.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "nvidia_nim/test-model",
                "input": "Hello",
                "stream": True,
            },
        )

    assert response.status_code == 200
    events = parse_sse_text(response.text)
    assert [event.event for event in events] == ["response.created", "response.failed"]
    failed = events[-1].data["response"]
    assert failed["id"] == events[0].data["response"]["id"]
    assert failed["status"] == "failed"
    assert failed["error"] == {
        "message": "provider failed",
        "type": "api_error",
        "param": None,
        "code": None,
    }


def test_create_response_replays_prior_reasoning_as_reasoning_content() -> None:
    provider = FakeProvider(_anthropic_text_stream("done"))
    app = create_app(lifespan_enabled=False)
    with (
        patch("api.dependencies.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "nvidia_nim/test-model",
                "input": [
                    {
                        "id": "rs_1",
                        "type": "reasoning",
                        "summary": [],
                        "content": [
                            {"type": "reasoning_text", "text": "Need the tool."}
                        ],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "echo",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "ok",
                    },
                    {
                        "id": "rs_2",
                        "type": "reasoning",
                        "summary": [
                            {"type": "summary_text", "text": "Use the result."}
                        ],
                    },
                    {"role": "user", "content": "continue"},
                ],
                "stream": True,
            },
        )

    assert response.status_code == 200
    routed = provider.requests[0]
    assert routed.messages[0].role == "assistant"
    assert routed.messages[0].reasoning_content == "Need the tool."
    assert routed.messages[0].content[0].type == "tool_use"
    assert routed.messages[1].role == "user"
    assert routed.messages[1].content[0].type == "tool_result"
    assert routed.messages[2].role == "assistant"
    assert routed.messages[2].content == ""
    assert routed.messages[2].reasoning_content == "Use the result."
    assert routed.messages[3].role == "user"
    assert routed.messages[3].content == "continue"


def test_create_response_quarantines_malformed_prior_function_call() -> None:
    provider = FakeProvider(_anthropic_text_stream("done"))
    app = create_app(lifespan_enabled=False)
    with (
        patch("api.dependencies.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "nvidia_nim/test-model",
                "input": [
                    {"role": "user", "content": "hello"},
                    {
                        "type": "function_call",
                        "call_id": "call_bad",
                        "name": "echo",
                        "arguments": "{",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_bad",
                        "output": "stale output",
                    },
                    {"role": "user", "content": "continue"},
                ],
                "stream": True,
            },
        )

    assert response.status_code == 200
    routed = provider.requests[0]
    assert [message.role for message in routed.messages] == ["user", "user"]
    assert routed.messages[0].content == "hello"
    assert routed.messages[1].content == "continue"
    completed = parse_sse_text(response.text)[-1].data["response"]
    assert completed["output"][0]["content"][0]["text"] == "done"


@pytest.mark.parametrize(
    ("reasoning", "expected_type", "expected_enabled"),
    [
        ({"effort": "none"}, "disabled", False),
        ({"effort": "low"}, "enabled", True),
    ],
)
def test_create_response_maps_reasoning_effort_to_thinking_request(
    reasoning: dict[str, str],
    expected_type: str,
    expected_enabled: bool,
) -> None:
    provider = FakeProvider(_anthropic_text_stream("done"))
    app = create_app(lifespan_enabled=False)
    with (
        patch("api.dependencies.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "nvidia_nim/test-model",
                "input": "Hello",
                "stream": True,
                "reasoning": reasoning,
            },
        )

    assert response.status_code == 200
    thinking = provider.requests[0].thinking
    assert thinking.type == expected_type
    assert thinking.enabled is expected_enabled


def test_create_response_maps_redacted_thinking_to_encrypted_reasoning() -> None:
    provider = FakeProvider(_anthropic_redacted_thinking_stream())
    app = create_app(lifespan_enabled=False)
    with (
        patch("api.dependencies.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "nvidia_nim/test-model",
                "input": "Continue",
                "stream": True,
            },
        )

    assert response.status_code == 200
    completed = parse_sse_text(response.text)[-1].data["response"]
    assert completed["output"] == [
        {
            "id": completed["output"][0]["id"],
            "type": "reasoning",
            "status": "completed",
            "summary": [],
            "encrypted_content": "opaque-redacted",
        }
    ]
    assert "content" not in completed["output"][0]


def test_create_response_unsupported_tool_returns_openai_error(
    responses_client: tuple[TestClient, FakeProvider],
) -> None:
    client, _provider = responses_client

    response = client.post(
        "/v1/responses",
        json={
            "model": "nvidia_nim/test-model",
            "input": "Hello",
            "tools": [{"type": "web_search_preview"}],
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert "Unsupported Responses tool type" in payload["error"]["message"]


def _anthropic_text_stream(text: str) -> list[str]:
    return [
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
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]


def _anthropic_tool_stream(
    tool_name: str = "echo", partial_json: str = '{"value":"FCC"}'
) -> list[str]:
    return [
        format_sse_event("message_start", {"type": "message_start", "message": {}}),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": tool_name,
                    "input": {},
                },
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": partial_json,
                },
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]


def _anthropic_reasoning_text_stream() -> list[str]:
    return [
        format_sse_event("message_start", {"type": "message_start", "message": {}}),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "thinking_delta",
                    "thinking": "provider reasoning",
                },
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "provider answer"},
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 1},
        ),
        format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]


def _anthropic_interleaved_reasoning_stream() -> list[str]:
    return [
        format_sse_event("message_start", {"type": "message_start", "message": {}}),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "first thought"},
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "first answer"},
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 1},
        ),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "echo",
                    "input": {},
                },
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"value":"FCC"}',
                },
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 2},
        ),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 3,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 3,
                "delta": {"type": "thinking_delta", "thinking": "second thought"},
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 3},
        ),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 4,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        format_sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 4,
                "delta": {"type": "text_delta", "text": "final answer"},
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 4},
        ),
        format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]


def _anthropic_redacted_thinking_stream() -> list[str]:
    return [
        format_sse_event("message_start", {"type": "message_start", "message": {}}),
        format_sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "redacted_thinking",
                    "data": "opaque-redacted",
                },
            },
        ),
        format_sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        ),
        format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        ),
        format_sse_event("message_stop", {"type": "message_stop"}),
    ]
