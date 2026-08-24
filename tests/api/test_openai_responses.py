from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.inference import (
    FinishReason,
    InferenceEvent,
    InferenceStreamLedger,
    MessageItem,
    MessageRole,
    ReasoningItem,
    ReplayArtifact,
    ReplayArtifactKind,
    ReplayArtifactOrigin,
    ReplayAttachment,
    ResponseStarted,
    TextContent,
    ToolCallItem,
    ToolCallKind,
    ToolChoiceMode,
    ToolResultItem,
)
from free_claude_code.core.reasoning import (
    ReasoningControl,
    ReasoningEffort,
    ReasoningPolicy,
)
from free_claude_code.core.replay_envelope import decode_replay_envelope
from tests.api.support import create_test_app
from tests.inference_support import reported_usage


class FakeProvider:
    def __init__(
        self,
        chunks: list[InferenceEvent],
        *,
        failure: Exception | None = None,
    ) -> None:
        self.chunks = chunks
        self.failure = failure
        self.preflight_stream = MagicMock()
        self.requests: list[Any] = []
        self.stream_kwargs: list[dict[str, Any]] = []

    async def stream_response(self, request_data, **_kwargs):
        self.requests.append(request_data)
        self.stream_kwargs.append(_kwargs)
        for chunk in self.chunks:
            yield chunk
        if self.failure is not None:
            raise self.failure


class PreStartFailingProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__([])

    async def stream_response(self, request_data, **_kwargs):
        self.requests.append(request_data)
        self.stream_kwargs.append(_kwargs)
        raise ExecutionFailure(
            kind=FailureKind.RATE_LIMIT,
            status_code=429,
            message="upstream is busy",
            retryable=True,
        )
        yield ResponseStarted("response_unreachable", "test-model")


class PostStartFailingProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__([ResponseStarted("response_test", "test-model")])

    async def stream_response(self, request_data, **_kwargs):
        self.requests.append(request_data)
        self.stream_kwargs.append(_kwargs)
        for chunk in self.chunks:
            yield chunk
        raise RuntimeError("socket closed")


@pytest.fixture
def responses_client():
    provider = FakeProvider(_anthropic_text_stream("Hello from provider"))
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
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
    assert response.headers["x-request-id"] == response.headers["request-id"]
    events = parse_sse_text(response.text)
    assert events[0].event == "response.created"
    assert events[-1].event == "response.completed"
    assert events[-1].data["response"]["output"][0]["content"][0]["text"] == (
        "Hello from provider"
    )
    assert provider.preflight_stream.called
    routed = provider.requests[0]
    assert routed.model == "nvidia_nim/test-model"
    assert routed.items == (
        MessageItem("turn_0", MessageRole.USER, (TextContent("Hello"),)),
    )
    assert routed.max_output_tokens == 32
    assert provider.stream_kwargs[0]["provider_model"] == "test-model"
    assert provider.stream_kwargs[0]["request_id"] == response.headers["request-id"]


def test_create_response_stream_preserves_output_limit_as_incomplete() -> None:
    provider = FakeProvider(
        _anthropic_text_stream("partial output", stop_reason="max_tokens")
    )
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "nvidia_nim/test-model",
                "input": "Keep working",
                "max_output_tokens": 32,
            },
        )

    assert response.status_code == 200
    events = parse_sse_text(response.text)
    assert events[-1].event == "response.incomplete"
    incomplete = events[-1].data["response"]
    assert incomplete["id"] == events[0].data["response"]["id"]
    assert incomplete["status"] == "incomplete"
    assert incomplete["incomplete_details"] == {"reason": "max_output_tokens"}
    assert incomplete["output"][0]["content"][0]["text"] == "partial output"


def test_create_response_preflight_rejection_stays_an_ordinary_http_error() -> None:
    provider = FakeProvider(_anthropic_text_stream("unused"))
    provider.preflight_stream.side_effect = InvalidRequestError("bad tool shape")
    app = create_test_app()

    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={"model": "nvidia_nim/test-model", "input": "Hello"},
        )

    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "bad tool shape",
        "type": "invalid_request_error",
        "param": None,
        "code": None,
    }
    assert "x-should-retry" not in response.headers
    assert provider.requests == []


def test_create_response_rejects_unknown_top_level_extensions_before_provider(
    responses_client: tuple[TestClient, FakeProvider],
) -> None:
    client, provider = responses_client

    response = client.post(
        "/v1/responses",
        json={
            "model": "nvidia_nim/test-model",
            "input": "Hello",
            "provider_extension": {"enabled": True},
        },
    )

    assert response.status_code == 400
    assert "provider_extension" in response.json()["error"]["message"]
    assert provider.requests == []


@pytest.mark.parametrize(
    "override",
    [
        {"store": True},
        {"previous_response_id": "resp_1"},
        {"include": []},
        {"prompt_cache_key": ""},
        {"reasoning": {"summary": "concise"}},
        {
            "tools": [{"type": "web_search"}],
            "tool_choice": "required",
        },
        {"tool_choice": {"type": "web_search"}},
        {"input": [{"type": "input_image", "image_url": "https://invalid"}]},
    ],
)
def test_invalid_responses_semantics_fail_before_provider_resolution(
    override: dict[str, object],
) -> None:
    provider = FakeProvider(_anthropic_text_stream("unused"))
    app = create_test_app()
    payload: dict[str, object] = {
        "model": "nvidia_nim/test-model",
        "input": "Hello",
    }
    payload.update(override)

    with (
        patch(
            "free_claude_code.api.routes.resolve_provider",
            return_value=provider,
        ) as resolve_provider,
        TestClient(app) as client,
    ):
        response = client.post("/v1/responses", json=payload)

    assert response.status_code == 400
    resolve_provider.assert_not_called()
    provider.preflight_stream.assert_not_called()
    assert provider.requests == []


def test_create_response_pre_start_provider_error_returns_openai_error() -> None:
    provider = PreStartFailingProvider()
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        patch("free_claude_code.api.response_streams.trace_event") as trace,
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "nvidia_nim/test-model",
                "input": "Hello",
            },
        )

    assert response.status_code == 429
    assert response.headers["x-should-retry"] == "false"
    assert response.headers["x-request-id"] == response.headers["request-id"]
    payload = response.json()
    assert payload["error"]["type"] == "rate_limit_error"
    assert payload["error"]["message"] == "upstream is busy"
    request_id = response.headers["request-id"]
    assert provider.stream_kwargs[0]["request_id"] == request_id
    terminal_trace = next(
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event")
        == "free_claude_code.api.response.terminal_execution_error"
    )
    assert terminal_trace["wire_api"] == "responses"
    assert terminal_trace["request_id"] == request_id
    assert terminal_trace["status_code"] == 429
    assert terminal_trace["error_type"] == "rate_limit_error"
    assert terminal_trace["client_should_retry"] is False
    assert terminal_trace["failure_kind"] == "rate_limit"
    assert terminal_trace["provider_retryable"] is True


def test_create_response_post_start_failure_preserves_response_id() -> None:
    provider = PostStartFailingProvider()
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "nvidia_nim/test-model",
                "input": "Hello",
            },
        )

    assert response.status_code == 200
    events = parse_sse_text(response.text)
    assert [event.event for event in events] == ["response.created", "response.failed"]
    assert events[-1].data["response"]["id"] == events[0].data["response"]["id"]
    assert events[-1].data["response"]["status"] == "failed"
    assert events[-1].data["response"]["error"]["message"] == "socket closed"


def test_create_response_stream_bypasses_local_message_optimizations() -> None:
    provider = FakeProvider(_anthropic_text_stream("Provider response"))
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        patch(
            "free_claude_code.api.handlers.messages.try_optimizations",
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
    assert provider.requests[0].items == (
        MessageItem("turn_0", MessageRole.USER, (TextContent("quota check"),)),
    )


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
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
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
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
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
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
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
    provider = FakeProvider(
        _anthropic_tool_stream(tool_name="js", namespace="mcp__node_repl")
    )
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
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
    assert [(tool.name, tool.namespace) for tool in routed.tools] == [
        ("js", "mcp__node_repl")
    ]
    completed = parse_sse_text(response.text)[-1].data["response"]
    call = completed["output"][0]
    assert call["namespace"] == "mcp__node_repl"
    assert call["name"] == "js"


def test_create_response_accepts_muse_code_request_shape() -> None:
    provider = FakeProvider(
        _anthropic_tool_stream(tool_name="read_file", namespace="muse")
    )
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "nvidia_nim/test-model",
                "input": "Read the file",
                "instructions": "Be concise.",
                "max_output_tokens": 64,
                "store": False,
                "stream": True,
                "reasoning": {"effort": "high", "summary": "auto"},
                "include": ["reasoning.encrypted_content"],
                "prompt_cache_key": "muse-session-1",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "muse",
                        "tools": [
                            {
                                "type": "function",
                                "name": "read_file",
                                "description": "Read one file.",
                                "strict": True,
                                "parameters": {
                                    "type": "object",
                                    "properties": {"path": {"type": "string"}},
                                    "required": ["path"],
                                    "additionalProperties": False,
                                },
                            }
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200
    routed = provider.requests[0]
    assert routed.max_output_tokens == 64
    assert routed.reasoning.control is ReasoningControl.ON
    assert routed.reasoning.effort is ReasoningEffort.HIGH
    assert provider.stream_kwargs[0]["reasoning"] == ReasoningPolicy(
        control=ReasoningControl.ON,
        effort=ReasoningEffort.HIGH,
    )
    assert [(tool.name, tool.namespace) for tool in routed.tools] == [
        ("read_file", "muse")
    ]
    completed = parse_sse_text(response.text)[-1].data["response"]
    call = completed["output"][0]
    assert call["namespace"] == "muse"
    assert call["name"] == "read_file"


def test_create_response_accepts_codex_custom_tool_request() -> None:
    provider = FakeProvider(
        _anthropic_tool_stream(
            tool_name="apply_patch",
            partial_json='{"input":"*** Begin Patch"}',
            kind=ToolCallKind.CUSTOM,
        )
    )
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
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
    assert routed.tool_choice is not None
    assert routed.tool_choice.mode is ToolChoiceMode.SPECIFIC
    assert routed.tool_choice.kind is ToolCallKind.CUSTOM
    assert routed.tool_choice.name == "apply_patch"
    events = parse_sse_text(response.text)
    assert "response.custom_tool_call_input.delta" in [event.event for event in events]
    completed = events[-1].data["response"]
    call = completed["output"][0]
    assert call["type"] == "custom_tool_call"
    assert call["name"] == "apply_patch"
    assert call["input"] == "*** Begin Patch"


def test_create_response_stream_provider_error_returns_response_failed() -> None:
    provider = FakeProvider(
        [ResponseStarted("response_test", "test-model")],
        failure=RuntimeError("provider failed"),
    )
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
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
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
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
    assert routed.items == (
        ReasoningItem("turn_0", "Need the tool."),
        ToolCallItem(
            turn_id="turn_0",
            call_id="call_1",
            kind=ToolCallKind.FUNCTION,
            name="echo",
            input={},
        ),
        ToolResultItem("turn_1", "call_1", "ok"),
        ReasoningItem("turn_2", "Use the result."),
        MessageItem("turn_3", MessageRole.USER, (TextContent("continue"),)),
    )


def test_create_response_quarantines_malformed_prior_function_call() -> None:
    provider = FakeProvider(_anthropic_text_stream("done"))
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
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
    assert routed.items == (
        MessageItem("turn_0", MessageRole.USER, (TextContent("hello"),)),
        MessageItem("turn_1", MessageRole.USER, (TextContent("continue"),)),
    )
    completed = parse_sse_text(response.text)[-1].data["response"]
    assert completed["output"][0]["content"][0]["text"] == "done"


@pytest.mark.parametrize(
    ("reasoning", "expected_policy"),
    [
        ({"effort": "none"}, ReasoningPolicy.off()),
        (
            {"effort": "low"},
            ReasoningPolicy(
                control=ReasoningControl.ON,
                effort=ReasoningEffort.LOW,
            ),
        ),
    ],
)
def test_create_response_preserves_and_resolves_reasoning_effort(
    reasoning: dict[str, str],
    expected_policy: ReasoningPolicy,
) -> None:
    provider = FakeProvider(_anthropic_text_stream("done"))
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
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
    routed = provider.requests[0]
    assert routed.reasoning.control is expected_policy.control
    assert routed.reasoning.effort is expected_policy.effort
    assert provider.stream_kwargs[0]["reasoning"] == expected_policy
    assert provider.preflight_stream.call_args.kwargs["reasoning"] == expected_policy


def test_create_response_maps_redacted_thinking_to_encrypted_reasoning() -> None:
    provider = FakeProvider(_anthropic_redacted_thinking_stream())
    app = create_test_app()
    with (
        patch("free_claude_code.api.routes.resolve_provider", return_value=provider),
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
    assert len(completed["output"]) == 1
    output = completed["output"][0]
    assert output["type"] == "reasoning"
    assert output["status"] == "completed"
    assert output["summary"] == []
    artifacts = decode_replay_envelope(
        output["encrypted_content"],
        attachment=ReplayAttachment.REASONING,
    )
    assert artifacts is not None
    assert len(artifacts) == 1
    assert artifacts[0].origin is ReplayArtifactOrigin.ANTHROPIC
    assert artifacts[0].kind is ReplayArtifactKind.REDACTED_THINKING
    assert artifacts[0].payload == "opaque-redacted"
    assert "content" not in output


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
    assert "tools[0].type" in payload["error"]["message"]


def _anthropic_text_stream(
    text: str,
    *,
    stop_reason: str = "end_turn",
) -> list[InferenceEvent]:
    ledger = InferenceStreamLedger("response_test", "test-model", 3)
    events: list[InferenceEvent] = [ledger.start_response()]
    events.extend(ledger.ensure_text_block())
    events.append(ledger.emit_text_delta(text))
    events.extend(ledger.close_all_blocks())
    finish_reason = (
        FinishReason.OUTPUT_LIMIT
        if stop_reason == "max_tokens"
        else FinishReason.END_TURN
    )
    events.extend(
        ledger.finish_events(
            finish_reason,
            reported_usage(input_tokens=3, output_tokens=4),
        )
    )
    return events


def _anthropic_tool_stream(
    tool_name: str = "echo",
    partial_json: str = '{"value":"FCC"}',
    *,
    kind: ToolCallKind = ToolCallKind.FUNCTION,
    namespace: str | None = None,
) -> list[InferenceEvent]:
    ledger = InferenceStreamLedger("response_test", "test-model", 3)
    events: list[InferenceEvent] = [ledger.start_response()]
    events.append(
        ledger.start_tool_block(
            0,
            "toolu_1",
            tool_name,
            kind=kind,
            namespace=namespace,
        )
    )
    events.append(ledger.emit_tool_delta(0, partial_json))
    events.append(ledger.stop_tool_block(0))
    events.extend(
        ledger.finish_events(
            FinishReason.TOOL_CALLS,
            reported_usage(input_tokens=3, output_tokens=4),
        )
    )
    return events


def _anthropic_reasoning_text_stream() -> list[InferenceEvent]:
    ledger = InferenceStreamLedger("response_test", "test-model", 3)
    events: list[InferenceEvent] = [ledger.start_response()]
    events.extend(ledger.ensure_reasoning_block())
    events.append(ledger.emit_reasoning_delta("provider reasoning"))
    events.extend(ledger.ensure_text_block())
    events.append(ledger.emit_text_delta("provider answer"))
    events.extend(ledger.close_all_blocks())
    events.extend(
        ledger.finish_events(
            FinishReason.END_TURN,
            reported_usage(input_tokens=3, output_tokens=4),
        )
    )
    return events


def _anthropic_interleaved_reasoning_stream() -> list[InferenceEvent]:
    ledger = InferenceStreamLedger("response_test", "test-model", 3)
    events: list[InferenceEvent] = [ledger.start_response()]
    events.extend(ledger.ensure_reasoning_block())
    events.append(ledger.emit_reasoning_delta("first thought"))
    events.extend(ledger.ensure_text_block())
    events.append(ledger.emit_text_delta("first answer"))
    events.extend(ledger.close_content_blocks())
    events.append(ledger.start_tool_block(0, "toolu_1", "echo"))
    events.append(ledger.emit_tool_delta(0, '{"value":"FCC"}'))
    events.append(ledger.stop_tool_block(0))
    events.extend(ledger.ensure_reasoning_block())
    events.append(ledger.emit_reasoning_delta("second thought"))
    events.extend(ledger.ensure_text_block())
    events.append(ledger.emit_text_delta("final answer"))
    events.extend(ledger.close_all_blocks())
    events.extend(
        ledger.finish_events(
            FinishReason.END_TURN,
            reported_usage(input_tokens=3, output_tokens=4),
        )
    )
    return events


def _anthropic_redacted_thinking_stream() -> list[InferenceEvent]:
    ledger = InferenceStreamLedger("response_test", "test-model", 3)
    events: list[InferenceEvent] = [ledger.start_response()]
    events.extend(
        ledger.emit_reasoning_artifact(
            ReplayArtifact(
                origin=ReplayArtifactOrigin.ANTHROPIC,
                kind=ReplayArtifactKind.REDACTED_THINKING,
                attachment=ReplayAttachment.REASONING,
                payload="opaque-redacted",
            )
        )
    )
    events.extend(
        ledger.finish_events(
            FinishReason.END_TURN,
            reported_usage(input_tokens=3, output_tokens=4),
        )
    )
    return events
