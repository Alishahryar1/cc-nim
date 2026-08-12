"""Unit tests for Antigravity CLI Provider client."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.providers.antigravity.auth import (
    ANTIGRAVITY_CLIENT_NAME,
    ANTIGRAVITY_GOOG_API_CLIENT,
    ANTIGRAVITY_USER_AGENT,
    AntigravityAuth,
)
from free_claude_code.providers.antigravity.client import (
    AntigravityProvider,
    _convert_anthropic_messages_to_gemini,
    _convert_anthropic_tools_to_gemini,
    _extract_error_message,
    _normalize_model_name,
)
from free_claude_code.providers.base import ProviderConfig


class MockMessage:
    def __init__(self, role: str, content: str | list[Any]):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(
        self,
        model: str = "antigravity-gemini-3.5-pro",
        messages: list[Any] | None = None,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: list[Any] | None = None,
        thinking: Any = None,
    ):
        self.model = model
        self.messages = messages or [MockMessage("user", "Hello Antigravity")]
        self.system = system
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tools = tools or []
        self.thinking = thinking
        self.stop_sequences = None
        self.extra_body = None


@pytest.fixture
def mock_auth():
    auth = MagicMock(spec=AntigravityAuth)
    auth.get_access_token_async = AsyncMock(return_value="mock_access_token_123")
    auth.get_project_id_async = AsyncMock(return_value="mock-project-456")
    return auth


@pytest.fixture
def provider_config():
    return ProviderConfig(
        api_key="auto_discovered",
        base_url="https://cloudcode-pa.googleapis.com",
    )


@pytest.fixture
def antigravity_provider(provider_config, mock_auth):
    return AntigravityProvider(provider_config, auth_manager=mock_auth)


def test_normalize_model_name():
    assert _normalize_model_name("models/gemini-2.5-pro") == "gemini-2.5-pro"
    assert _normalize_model_name("antigravity/gemini-3.5-pro") == "gemini-3.5-pro"
    assert _normalize_model_name("gemini-2.5-flash") == "gemini-2.5-flash"


def test_convert_anthropic_tools_to_gemini():
    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]
    gemini_tools = _convert_anthropic_tools_to_gemini(tools)
    assert len(gemini_tools) == 1
    decls = gemini_tools[0].get("functionDeclarations", [])
    assert len(decls) == 1
    assert decls[0]["name"] == "get_weather"
    assert decls[0]["description"] == "Get current weather"


def test_convert_anthropic_messages_to_gemini():
    messages = [
        MockMessage("user", "Hello"),
        MockMessage("assistant", "Hi there"),
    ]
    system = "You are a helpful assistant."

    contents, system_inst = _convert_anthropic_messages_to_gemini(
        messages, system=system
    )
    assert len(contents) == 2
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "Hello"
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0]["text"] == "Hi there"

    assert system_inst is not None
    assert system_inst["parts"][0]["text"] == "You are a helpful assistant."


def test_convert_anthropic_image_messages_to_gemini():
    messages = [
        MockMessage(
            "user",
            [
                {"type": "text", "text": "Look at this image"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    },
                },
            ],
        )
    ]
    contents, _ = _convert_anthropic_messages_to_gemini(messages)
    assert len(contents) == 1
    parts = contents[0]["parts"]
    assert len(parts) == 2
    assert parts[0]["text"] == "Look at this image"
    assert parts[1]["inlineData"]["mimeType"] == "image/png"
    assert (
        parts[1]["inlineData"]["data"]
        == "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


@pytest.mark.asyncio
async def test_list_model_ids(antigravity_provider):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "quotaBuckets": [
            {"modelId": "gemini-2.5-pro"},
            {"modelId": "gemini-3.5-pro"},
        ]
    }
    with patch.object(
        antigravity_provider._client, "post", AsyncMock(return_value=mock_resp)
    ):
        model_ids = await antigravity_provider.list_model_ids()
        assert "gemini-2.5-pro" in model_ids
        assert "antigravity/gemini-2.5-pro" in model_ids
        assert "gemini-3.5-pro" in model_ids


def test_build_request_headers(antigravity_provider):
    headers = antigravity_provider._build_request_headers("test_token_xyz")
    assert headers["User-Agent"] == ANTIGRAVITY_USER_AGENT
    assert headers["X-Client-Name"] == ANTIGRAVITY_CLIENT_NAME
    assert headers["X-Goog-Api-Client"] == ANTIGRAVITY_GOOG_API_CLIENT
    assert headers["Authorization"] == "Bearer test_token_xyz"
    assert headers["Accept"] == "text/event-stream"


def test_build_request_body_metadata_and_structure(antigravity_provider):
    req = MockRequest(
        system="System prompt",
        max_tokens=500,
        temperature=0.5,
    )
    body = antigravity_provider._build_request_body(
        req, model_name="gemini-3.5-pro", project_id="test-proj-789"
    )

    # Model and project invariant check
    assert body["model"] == "gemini-3.5-pro"
    assert body["project"] == "test-proj-789"

    # Request contents and config check
    req_dict = body["request"]
    assert len(req_dict["contents"]) == 1
    assert req_dict["systemInstruction"]["parts"][0]["text"] == "System prompt"
    assert req_dict["generationConfig"]["maxOutputTokens"] == 500
    assert req_dict["generationConfig"]["temperature"] == 0.5


@pytest.mark.asyncio
async def test_stream_response_text_success(antigravity_provider):
    req = MockRequest()

    sse_lines = [
        'data: {"candidates": [{"content": {"role": "model", "parts": [{"text": "Hello "}]}}]}\n\n',
        'data: {"candidates": [{"content": {"role": "model", "parts": [{"text": "world!"}]}, "finishReason": "STOP"}]}\n\n',
    ]

    async def mock_aiter_lines():
        for line in sse_lines:
            yield line

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.aiter_lines = mock_aiter_lines

    class MockStreamContext:
        async def __aenter__(self):
            return mock_resp

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch.object(
        antigravity_provider._client, "stream", return_value=MockStreamContext()
    ):
        events = [chunk async for chunk in antigravity_provider.stream_response(req)]
        full_stream = "".join(events)
        assert "event: message_start" in full_stream
        assert "Hello " in full_stream
        assert "world!" in full_stream
        assert "event: message_stop" in full_stream


@pytest.mark.asyncio
async def test_stream_response_thinking(antigravity_provider):
    req = MockRequest()

    sse_lines = [
        'data: {"candidates": [{"content": {"role": "model", "parts": [{"thought": "Thinking process..."}]}}]}\n\n',
        'data: {"candidates": [{"content": {"role": "model", "parts": [{"text": "Final answer"}]}, "finishReason": "STOP"}]}\n\n',
    ]

    async def mock_aiter_lines():
        for line in sse_lines:
            yield line

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.aiter_lines = mock_aiter_lines

    class MockStreamContext:
        async def __aenter__(self):
            return mock_resp

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch.object(
        antigravity_provider._client, "stream", return_value=MockStreamContext()
    ):
        events = [chunk async for chunk in antigravity_provider.stream_response(req)]
        full_stream = "".join(events)
        assert "Thinking process..." in full_stream
        assert "Final answer" in full_stream


@pytest.mark.asyncio
async def test_stream_response_tool_use(antigravity_provider):
    req = MockRequest()

    sse_lines = [
        'data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "read_file", "args": {"path": "main.py"}}}]}, "finishReason": "STOP"}]}\n\n',
    ]

    async def mock_aiter_lines():
        for line in sse_lines:
            yield line

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.aiter_lines = mock_aiter_lines

    class MockStreamContext:
        async def __aenter__(self):
            return mock_resp

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch.object(
        antigravity_provider._client, "stream", return_value=MockStreamContext()
    ):
        events = [chunk async for chunk in antigravity_provider.stream_response(req)]
        full_stream = "".join(events)
        assert "tool_use" in full_stream
        assert "read_file" in full_stream
        assert "main.py" in full_stream


@pytest.mark.asyncio
async def test_stream_response_duplicate_tool_calls_deduplicated(antigravity_provider):
    req = MockRequest()

    # Simulate Gemini streaming where chunk 1 delivers functionCall and chunk 2 repeats identical functionCall
    sse_lines = [
        'data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "Edit", "args": {"old_string": "foo", "new_string": "bar"}}}]}}]}\n\n',
        'data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "Edit", "args": {"old_string": "foo", "new_string": "bar"}}}]}, "finishReason": "STOP"}]}\n\n',
    ]

    async def mock_aiter_lines():
        for line in sse_lines:
            yield line

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.aiter_lines = mock_aiter_lines

    class MockStreamContext:
        async def __aenter__(self):
            return mock_resp

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch.object(
        antigravity_provider._client, "stream", return_value=MockStreamContext()
    ):
        events = [chunk async for chunk in antigravity_provider.stream_response(req)]
        full_stream = "".join(events)

        # Count occurrences of tool_use content_block_start
        tool_use_count = full_stream.count('"type": "tool_use"')
        assert tool_use_count == 1, f"Expected 1 tool_use block, got {tool_use_count}"
        assert "Edit" in full_stream
        assert "foo" in full_stream
        assert "bar" in full_stream


@pytest.mark.asyncio
async def test_stream_response_empty_args_then_populated_args_accumulated(
    antigravity_provider,
):
    req = MockRequest()

    # Simulate Gemini streaming where chunk 1 delivers functionCall with empty args {} and chunk 2 populates args
    sse_lines = [
        'data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "Bash", "args": {}}}]}}]}\n\n',
        'data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "Bash", "args": {"command": "ls -la"}}}]}, "finishReason": "STOP"}]}\n\n',
    ]

    async def mock_aiter_lines():
        for line in sse_lines:
            yield line

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.aiter_lines = mock_aiter_lines

    class MockStreamContext:
        async def __aenter__(self):
            return mock_resp

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch.object(
        antigravity_provider._client, "stream", return_value=MockStreamContext()
    ):
        events = [chunk async for chunk in antigravity_provider.stream_response(req)]
        full_stream = "".join(events)

        tool_use_count = full_stream.count('"type": "tool_use"')
        assert tool_use_count == 1, f"Expected 1 tool_use block, got {tool_use_count}"
        assert "Bash" in full_stream
        assert "ls -la" in full_stream


def test_extract_error_message_google_quota_exhausted():
    raw_json = """{
  "error": {
    "code": 429,
    "message": "Individual quota reached. Please upgrade your subscription to increase your limits. Resets in 157h4m32s.",
    "status": "RESOURCE_EXHAUSTED",
    "details": [
      {
        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
        "reason": "QUOTA_EXHAUSTED",
        "domain": "cloudcode-pa.googleapis.com",
        "metadata": {
          "uiMessage": "true",
          "model": "gemini-3.6-flash-high"
        }
      }
    ]
  }
}"""
    extracted = _extract_error_message(raw_json)
    assert "[QUOTA_EXHAUSTED]" in extracted
    assert (
        "Individual quota reached. Please upgrade your subscription to increase your limits."
        in extracted
    )


@pytest.mark.asyncio
async def test_stream_response_error_emits_sse_events(antigravity_provider):
    req = MockRequest()

    mock_resp = AsyncMock()
    mock_resp.status_code = 401
    mock_resp.text = '{"error": {"message": "Invalid credentials"}}'
    mock_resp.aread = AsyncMock()

    class MockStreamContext:
        async def __aenter__(self):
            return mock_resp

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch.object(
        antigravity_provider._client, "stream", return_value=MockStreamContext()
    ):
        events = [chunk async for chunk in antigravity_provider.stream_response(req)]
        full_stream = "".join(events)
        assert "event: message_start" in full_stream
        assert "Invalid credentials" in full_stream
        assert "event: message_stop" in full_stream


@pytest.mark.asyncio
async def test_stream_response_quota_exhausted_429(antigravity_provider):
    req = MockRequest()

    raw_429 = """{
  "error": {
    "code": 429,
    "message": "Individual quota reached. Please upgrade your subscription to increase your limits. Resets in 157h4m32s.",
    "status": "RESOURCE_EXHAUSTED",
    "details": [
      {
        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
        "reason": "QUOTA_EXHAUSTED",
        "domain": "cloudcode-pa.googleapis.com"
      }
    ]
  }
}"""
    mock_resp = AsyncMock()
    mock_resp.status_code = 429
    mock_resp.text = raw_429
    mock_resp.aread = AsyncMock()

    class MockStreamContext:
        async def __aenter__(self):
            return mock_resp

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch.object(
        antigravity_provider._client, "stream", return_value=MockStreamContext()
    ):
        events = [chunk async for chunk in antigravity_provider.stream_response(req)]
        full_stream = "".join(events)
        assert "event: message_start" in full_stream
        assert "[QUOTA_EXHAUSTED]" in full_stream
        assert "Individual quota reached." in full_stream
        assert "event: message_stop" in full_stream


@pytest.mark.asyncio
async def test_stream_response_empty_stream_emits_space(antigravity_provider):
    req = MockRequest()

    async def empty_lines():
        yield "data: {}\n"
        yield "data: [DONE]\n"

    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.aiter_lines = empty_lines

    class MockStreamContext:
        async def __aenter__(self):
            return mock_resp

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch.object(
        antigravity_provider._client, "stream", return_value=MockStreamContext()
    ):
        events = [chunk async for chunk in antigravity_provider.stream_response(req)]
        full_stream = "".join(events)
        assert "event: message_start" in full_stream
        assert "event: content_block_start" in full_stream
        assert "event: content_block_delta" in full_stream
        assert (
            '"text": " "' in full_stream
            or '"text_delta": " "' in full_stream
            or '"text"' in full_stream
        )
        assert "event: message_stop" in full_stream


def test_convert_anthropic_tool_result_success():
    messages = [
        MockMessage(
            "assistant",
            [
                {
                    "type": "tool_use",
                    "id": "call_123",
                    "name": "read_file",
                    "input": {"path": "test.txt"},
                }
            ],
        ),
        MockMessage(
            "user",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_123",
                    "content": [{"type": "text", "text": "file content line 1"}],
                    "is_error": False,
                }
            ],
        ),
    ]
    contents, _ = _convert_anthropic_messages_to_gemini(messages)
    assert len(contents) == 2
    fn_call_part = contents[0]["parts"][0]
    assert "functionCall" in fn_call_part
    assert fn_call_part["thought_signature"] == "skip_thought_signature_validator"
    assert "thought" not in fn_call_part

    tool_resp_part = contents[1]["parts"][0]
    assert "functionResponse" in tool_resp_part
    fn_resp = tool_resp_part["functionResponse"]
    assert fn_resp["name"] == "read_file"
    assert fn_resp["response"] == {"output": "file content line 1"}
    assert "error" not in fn_resp["response"]


def test_convert_anthropic_tool_result_error():
    messages = [
        MockMessage(
            "assistant",
            [
                {
                    "type": "tool_use",
                    "id": "call_456",
                    "name": "run_command",
                    "input": {"command": "invalid"},
                }
            ],
        ),
        MockMessage(
            "user",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_456",
                    "content": [{"type": "text", "text": "Command not found: invalid"}],
                    "is_error": True,
                }
            ],
        ),
    ]
    contents, _ = _convert_anthropic_messages_to_gemini(messages)
    assert len(contents) == 2
    tool_resp_part = contents[1]["parts"][0]
    assert "functionResponse" in tool_resp_part
    fn_resp = tool_resp_part["functionResponse"]
    assert fn_resp["name"] == "run_command"
    assert fn_resp["response"]["error"] == "Command not found: invalid"
    assert fn_resp["response"]["output"] == "Command not found: invalid"
