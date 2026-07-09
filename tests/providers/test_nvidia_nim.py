import json
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from httpx import Request, Response

from config.nim import NimSettings
from providers.defaults import NVIDIA_NIM_DEFAULT_BASE
from providers.exceptions import ProviderError
from providers.nvidia_nim import NvidiaNimProvider
from providers.nvidia_nim.tool_schema import NIM_TOOL_ARGUMENT_ALIASES_KEY


# Mock data classes
class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockTool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class MockBlock:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "test-model"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = ["STOP"]
        self.tools = []
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for k, v in kwargs.items():
            setattr(self, k, v)


def _input_json_deltas(events):
    deltas = []
    for event in events:
        if "event: content_block_delta" not in event:
            continue
        for line in event.splitlines():
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            delta = payload.get("delta", {})
            if delta.get("type") == "input_json_delta":
                deltas.append(delta.get("partial_json", ""))
    return deltas


def _tool_call_chunk(
    *,
    name,
    arguments,
    tool_id="call_1",
    index=0,
    finish_reason=None,
):
    mock_tc = MagicMock()
    mock_tc.index = index
    mock_tc.id = tool_id
    mock_tc.function.name = name
    mock_tc.function.arguments = arguments

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="", tool_calls=[mock_tc]),
            finish_reason=finish_reason,
        )
    ]
    mock_chunk.usage = None
    return mock_chunk


def _make_bad_request_error(message: str) -> openai.BadRequestError:
    response = Response(
        status_code=400,
        request=Request("POST", f"{NVIDIA_NIM_DEFAULT_BASE}/chat/completions"),
    )
    body = {"error": {"message": message, "type": "BadRequestError", "code": 400}}
    return openai.BadRequestError(message, response=response, body=body)


def _make_internal_server_error(message: str) -> openai.InternalServerError:
    response = Response(
        status_code=500,
        request=Request("POST", f"{NVIDIA_NIM_DEFAULT_BASE}/chat/completions"),
    )
    body = {
        "error": {
            "message": message,
            "type": "internal_server_error",
            "code": 500,
        }
    }
    return openai.InternalServerError(message, response=response, body=body)


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch("providers.transports.openai_chat.transport.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        # execute_with_retry should call through to the actual function
        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


@pytest.mark.asyncio
async def test_init(provider_config):
    """Test provider initialization."""
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI") as mock_openai:
        provider = NvidiaNimProvider(provider_config, nim_settings=NimSettings())
        assert provider._api_key == "test_key"
        assert provider._base_url == "https://test.api.nvidia.com/v1"
        mock_openai.assert_called_once()


@pytest.mark.asyncio
async def test_init_uses_configurable_timeouts():
    """Test that provider passes configurable read/write/connect timeouts to client."""
    from providers.base import ProviderConfig

    config = ProviderConfig(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI") as mock_openai:
        NvidiaNimProvider(config, nim_settings=NimSettings())
        call_kwargs = mock_openai.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


@pytest.mark.asyncio
async def test_build_request_body(provider_config):
    """Test request body construction."""
    provider = NvidiaNimProvider(provider_config, nim_settings=NimSettings())
    req = MockRequest()
    body = provider._build_request_body(req)

    assert body["model"] == "test-model"
    assert body["temperature"] == 0.5
    assert len(body["messages"]) == 2  # System + User
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "System prompt"

    assert "extra_body" in body
    ctk = body["extra_body"]["chat_template_kwargs"]
    assert ctk["thinking"] is True
    assert ctk["enable_thinking"] is True
    assert ctk["reasoning_budget"] == body["max_tokens"]
    assert "reasoning_budget" not in body["extra_body"]


@pytest.mark.asyncio
async def test_build_request_body_omits_reasoning_when_globally_disabled(
    provider_config,
):
    provider = NvidiaNimProvider(
        provider_config.model_copy(update={"enable_thinking": False}),
        nim_settings=NimSettings(),
    )
    req = MockRequest()
    body = provider._build_request_body(req)

    extra = body.get("extra_body", {})
    assert "chat_template_kwargs" not in extra
    assert "reasoning_budget" not in extra


@pytest.mark.asyncio
async def test_build_request_body_omits_reasoning_when_request_disables_thinking(
    provider_config,
):
    provider = NvidiaNimProvider(provider_config, nim_settings=NimSettings())
    req = MockRequest()
    req.thinking.enabled = False
    body = provider._build_request_body(req)

    extra = body.get("extra_body", {})
    assert "chat_template_kwargs" not in extra
    assert "reasoning_budget" not in extra


def test_preflight_and_build_request_issue_206_post_tool_text(nim_provider):
    """Regression: assistant message with tool_use then text plus tool results (GitHub #206)."""
    tool_id = "toolu_issue_206"
    req = MockRequest(
        messages=[
            MockMessage("user", "Use echo once."),
            MockMessage(
                "assistant",
                [
                    MockBlock(
                        type="tool_use",
                        id=tool_id,
                        name="echo_smoke",
                        input={"value": "FCC_206"},
                    ),
                    MockBlock(
                        type="text",
                        text="Commentary after the tool row.",
                    ),
                ],
            ),
            MockMessage(
                "user",
                [
                    MockBlock(
                        type="tool_result", tool_use_id=tool_id, content="FCC_206"
                    ),
                    MockBlock(type="text", text="What was echoed?"),
                ],
            ),
        ],
    )
    nim_provider.preflight_stream(req, thinking_enabled=False)
    body = nim_provider._build_request_body(req, thinking_enabled=False)
    assert "messages" in body
    assert any(m.get("role") == "tool" for m in body["messages"])


@pytest.mark.asyncio
async def test_stream_response_text(nim_provider):
    """Test streaming text response."""
    req = MockRequest()

    # Create mock chunks
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [
        MagicMock(
            delta=MagicMock(content="Hello", reasoning_content=""), finish_reason=None
        )
    ]
    mock_chunk1.usage = None

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [
        MagicMock(
            delta=MagicMock(content=" World", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk2.usage = MagicMock(completion_tokens=10)

    async def mock_stream():
        yield mock_chunk1
        yield mock_chunk2

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_response(req)]

        assert len(events) > 0
        assert "event: message_start" in events[0]

        text_content = ""
        for e in events:
            if "event: content_block_delta" in e and '"text_delta"' in e:
                for line in e.splitlines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "delta" in data and "text" in data["delta"]:
                            text_content += data["delta"]["text"]

        assert "Hello World" in text_content


@pytest.mark.asyncio
async def test_stream_response_thinking_reasoning_content(nim_provider):
    """Test streaming with native reasoning_content."""
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="Thinking..."),
            finish_reason=None,
        )
    ]
    mock_chunk.usage = None
    stop_chunk = MagicMock()
    stop_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content=None, tool_calls=None),
            finish_reason="stop",
        )
    ]
    stop_chunk.usage = None

    async def mock_stream():
        yield mock_chunk
        yield stop_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_response(req)]

        # Check for thinking_delta
        found_thinking = False
        for e in events:
            if (
                "event: content_block_delta" in e
                and '"thinking_delta"' in e
                and "Thinking..." in e
            ):
                found_thinking = True
        assert found_thinking


@pytest.mark.asyncio
async def test_stream_response_suppresses_thinking_when_disabled(provider_config):
    provider = NvidiaNimProvider(
        provider_config.model_copy(update={"enable_thinking": False}),
        nim_settings=NimSettings(),
    )
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="<think>secret</think>Answer", reasoning_content="Thinking..."
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in provider.stream_response(req)]

    event_text = "".join(events)
    assert "thinking_delta" not in event_text
    assert "Thinking..." not in event_text
    assert "secret" not in event_text
    assert "Answer" in event_text


def _make_bad_request_error(message: str) -> openai.BadRequestError:
    response = Response(status_code=400, request=Request("POST", "http://test"))
    body = {"error": {"message": message}}
    return openai.BadRequestError(message, response=response, body=body)


@pytest.mark.asyncio
async def test_stream_response_retries_without_chat_template(provider_config):
    provider = NvidiaNimProvider(
        provider_config,
        nim_settings=NimSettings(chat_template="custom_template"),
    )
    req = MockRequest(model="mistralai/mixtral-8x7b-instruct-v0.1")

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="OK", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2)

    async def mock_stream():
        yield mock_chunk

    first_error = _make_bad_request_error(
        "chat_template is not supported for Mistral tokenizers."
    )

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [first_error, mock_stream()]

        events = [e async for e in provider.stream_response(req)]

    assert mock_create.await_count == 2

    first_extra = mock_create.call_args_list[0].kwargs["extra_body"]
    second_extra = mock_create.call_args_list[1].kwargs["extra_body"]

    assert first_extra["chat_template"] == "custom_template"
    assert first_extra["chat_template_kwargs"] == {
        "thinking": True,
        "enable_thinking": True,
        "reasoning_budget": 100,
    }
    assert "reasoning_budget" not in first_extra

    assert "chat_template" not in second_extra
    assert "chat_template_kwargs" not in second_extra
    assert "reasoning_budget" not in second_extra

    event_text = "".join(events)
    assert "event: error" not in event_text
    assert "OK" in event_text


@pytest.mark.asyncio
async def test_stream_response_retries_without_chat_template_kwargs_issue_993(
    provider_config,
):
    provider = NvidiaNimProvider(provider_config, nim_settings=NimSettings())
    req = MockRequest(model="mistralai/mistral-small-4-119b-2603")

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="OK", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2)

    async def mock_stream():
        yield mock_chunk

    first_error = _make_bad_request_error(
        "chat_template is not supported for Mistral tokenizers."
    )

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [first_error, mock_stream()]

        events = [e async for e in provider.stream_response(req)]

    assert mock_create.await_count == 2

    first_extra = mock_create.call_args_list[0].kwargs["extra_body"]
    second_kwargs = mock_create.call_args_list[1].kwargs

    assert "chat_template" not in first_extra
    assert first_extra["chat_template_kwargs"] == {
        "thinking": True,
        "enable_thinking": True,
        "reasoning_budget": 100,
    }
    second_extra = second_kwargs.get("extra_body") or {}
    assert "chat_template" not in second_extra
    assert "chat_template_kwargs" not in second_extra

    event_text = "".join(events)
    assert "event: error" not in event_text
    assert "OK" in event_text


@pytest.mark.asyncio
async def test_stream_response_does_not_retry_unrelated_bad_request(provider_config):
    provider = NvidiaNimProvider(
        provider_config,
        nim_settings=NimSettings(chat_template="custom_template"),
    )
    req = MockRequest(model="mistralai/mixtral-8x7b-instruct-v0.1")

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = _make_bad_request_error("unrelated bad request")

        with pytest.raises(ProviderError) as exc_info:
            [e async for e in provider.stream_response(req)]

    assert mock_create.await_count == 1
    assert "Invalid request sent to provider" in exc_info.value.message


@pytest.mark.asyncio
async def test_tool_call_stream(nim_provider):
    """Test streaming tool calls."""
    req = MockRequest()

    # Mock tool call delta
    mock_tc = MagicMock()
    mock_tc.index = 0
    mock_tc.id = "call_1"
    mock_tc.function.name = "search"
    mock_tc.function.arguments = '{"q": "test"}'

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="", tool_calls=[mock_tc]),
            finish_reason=None,
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_response(req)]

        starts = [
            e for e in events if "event: content_block_start" in e and '"tool_use"' in e
        ]
        assert len(starts) == 1
        assert "search" in starts[0]


@pytest.mark.asyncio
async def test_stream_response_restores_aliased_tool_arguments(nim_provider):
    """NIM-safe argument aliases are restored before Anthropic SSE emission."""
    req = MockRequest(
        tools=[
            MockTool(
                "Grep",
                "Search file contents",
                {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "-A": {"type": "number"},
                        "type": {"type": "string"},
                    },
                    "required": ["pattern"],
                },
            )
        ]
    )
    mock_chunk = _tool_call_chunk(
        name="Grep",
        arguments=json.dumps({"pattern": "needle", "-A": 2, "_fcc_arg_type": "py"}),
    )

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_response(req)]

    await_args = mock_create.await_args
    assert await_args is not None
    create_kwargs = await_args.kwargs
    assert NIM_TOOL_ARGUMENT_ALIASES_KEY not in create_kwargs
    properties = create_kwargs["tools"][0]["function"]["parameters"]["properties"]
    assert "-A" in properties
    assert "type" not in properties
    assert "_fcc_arg_A" not in properties
    assert "_fcc_arg_type" in properties

    deltas = _input_json_deltas(events)
    assert len(deltas) == 1
    assert json.loads(deltas[0]) == {"pattern": "needle", "-A": 2, "type": "py"}
    assert "_fcc_arg_type" not in deltas[0]


@pytest.mark.asyncio
async def test_stream_response_buffers_chunked_aliased_tool_arguments(nim_provider):
    """Chunked aliased args are emitted once as restored Claude Code args."""
    req = MockRequest(
        tools=[
            MockTool(
                "Grep",
                "Search file contents",
                {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "type": {"type": "string"},
                    },
                    "required": ["pattern"],
                },
            )
        ]
    )
    first_chunk = _tool_call_chunk(
        name="Grep",
        arguments='{"pattern": "needle", ',
        tool_id="call_chunked",
    )
    second_chunk = _tool_call_chunk(
        name=None,
        arguments='"_fcc_arg_type": "py"}',
        tool_id="call_chunked",
    )

    async def mock_stream():
        yield first_chunk
        yield second_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_response(req)]

    deltas = _input_json_deltas(events)
    assert len(deltas) == 1
    assert json.loads(deltas[0]) == {"pattern": "needle", "type": "py"}


@pytest.mark.asyncio
async def test_stream_response_restores_nested_aliased_tool_arguments(nim_provider):
    req = MockRequest(
        tools=[
            MockTool(
                "NotionLike",
                "Nested type schema",
                {
                    "type": "object",
                    "properties": {
                        "parent": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "id": {"type": "string"},
                            },
                            "required": ["type", "id"],
                        }
                    },
                    "required": ["parent"],
                },
            )
        ]
    )
    mock_chunk = _tool_call_chunk(
        name="NotionLike",
        arguments=json.dumps(
            {"parent": {"_fcc_arg_type": "page_id", "id": "page_123"}}
        ),
    )

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_response(req)]

    deltas = _input_json_deltas(events)
    assert len(deltas) == 1
    assert json.loads(deltas[0]) == {"parent": {"type": "page_id", "id": "page_123"}}


@pytest.mark.asyncio
async def test_stream_response_task_tool_still_forces_background_false(nim_provider):
    req = MockRequest(
        tools=[
            MockTool(
                "Task",
                "Run a subagent",
                {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "prompt": {"type": "string"},
                        "run_in_background": {"type": "boolean"},
                    },
                    "required": ["description", "prompt"],
                },
            )
        ]
    )
    mock_chunk = _tool_call_chunk(
        name="Task",
        arguments=json.dumps(
            {
                "description": "Inspect",
                "prompt": "Read the marker",
                "run_in_background": True,
            }
        ),
        tool_id="call_task",
    )

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_response(req)]

    deltas = _input_json_deltas(events)
    assert len(deltas) == 1
    assert json.loads(deltas[0])["run_in_background"] is False


@pytest.mark.asyncio
async def test_stream_response_retries_without_reasoning_budget(nim_provider):
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Recovered", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5)

    async def mock_stream():
        yield mock_chunk

    error = _make_bad_request_error("Unsupported field: reasoning_budget")

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [error, mock_stream()]

        events = [e async for e in nim_provider.stream_response(req)]

    assert mock_create.await_count == 2
    first_call = mock_create.await_args_list[0].kwargs
    second_call = mock_create.await_args_list[1].kwargs
    assert (
        first_call["extra_body"]["chat_template_kwargs"]["reasoning_budget"]
        == first_call["max_tokens"]
    )
    assert "reasoning_budget" not in second_call["extra_body"]
    assert "reasoning_budget" not in second_call["extra_body"]["chat_template_kwargs"]
    assert second_call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert any("Recovered" in event for event in events)
    assert any("message_stop" in event for event in events)


@pytest.mark.asyncio
async def test_stream_response_retries_without_budget_for_thinking_token_error(
    nim_provider,
):
    req = MockRequest(model="meta/llama-3.3-70b-instruct")

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Recovered", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5)

    async def mock_stream():
        yield mock_chunk

    error = _make_internal_server_error(
        "ValueError: thinking_token_budget is set but reasoning_config is not "
        "configured. Please set --reasoning-config to use thinking_token_budget."
    )

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [error, mock_stream()]

        events = [e async for e in nim_provider.stream_response(req)]

    assert mock_create.await_count == 2
    first_call = mock_create.await_args_list[0].kwargs
    second_call = mock_create.await_args_list[1].kwargs
    assert (
        first_call["extra_body"]["chat_template_kwargs"]["reasoning_budget"]
        == first_call["max_tokens"]
    )
    assert "reasoning_budget" not in second_call["extra_body"]
    assert "reasoning_budget" not in second_call["extra_body"]["chat_template_kwargs"]
    assert second_call["extra_body"]["chat_template_kwargs"]["thinking"] is True
    assert second_call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert any("Recovered" in event for event in events)
    assert any("message_stop" in event for event in events)


@pytest.mark.asyncio
async def test_stream_response_retries_without_reasoning_content(nim_provider):
    req = MockRequest(
        system=None,
        messages=[
            MockMessage(
                "assistant",
                [
                    MockBlock(type="thinking", thinking="Need the tool."),
                    MockBlock(
                        type="tool_use",
                        id="toolu_reasoning",
                        name="echo_smoke",
                        input={"value": "FCC_TOOL"},
                    ),
                ],
            ),
            MockMessage(
                "user",
                [
                    MockBlock(
                        type="tool_result",
                        tool_use_id="toolu_reasoning",
                        content="result",
                    )
                ],
            ),
        ],
    )

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Recovered", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5)

    async def mock_stream():
        yield mock_chunk

    error = _make_bad_request_error("Unsupported field: reasoning_content")

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [error, mock_stream()]

        events = [e async for e in nim_provider.stream_response(req)]

    assert mock_create.await_count == 2
    first_call = mock_create.await_args_list[0].kwargs
    second_call = mock_create.await_args_list[1].kwargs
    assert first_call["messages"][0]["reasoning_content"] == "Need the tool."
    assert "reasoning_content" not in second_call["messages"][0]
    assert second_call["messages"][0]["tool_calls"][0]["id"] == "toolu_reasoning"
    assert any("Recovered" in event for event in events)
    assert any("message_stop" in event for event in events)


@pytest.mark.asyncio
async def test_stream_response_bad_request_without_reasoning_budget_does_not_retry(
    nim_provider,
):
    req = MockRequest()
    error = _make_bad_request_error("Unsupported field: top_k")

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = error

        with pytest.raises(ProviderError) as exc_info:
            [e async for e in nim_provider.stream_response(req)]

    assert mock_create.await_count == 1
    assert "Invalid request sent to provider" in exc_info.value.message


@pytest.mark.asyncio
async def test_stream_response_unrelated_internal_error_does_not_downgrade(
    nim_provider,
):
    req = MockRequest()
    error = _make_internal_server_error("unrelated internal provider failure")

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = error

        with pytest.raises(ProviderError) as exc_info:
            [e async for e in nim_provider.stream_response(req)]

    assert mock_create.await_count == 1
    assert "Provider API request failed" in exc_info.value.message


@pytest.mark.asyncio
async def test_stream_response_internal_reasoning_content_error_does_not_downgrade(
    nim_provider,
):
    req = MockRequest()
    error = _make_internal_server_error(
        "reasoning_content could not be processed by the upstream model"
    )

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = error

        with pytest.raises(ProviderError) as exc_info:
            [e async for e in nim_provider.stream_response(req)]

    assert mock_create.await_count == 1
    assert "Provider API request failed" in exc_info.value.message
