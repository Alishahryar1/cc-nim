"""Tests for LM Studio OpenAI-compatible chat completions provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.lmstudio import LMStudioProvider
from tests.stream_contract import assert_canonical_stream_error_envelope


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "lmstudio-community/qwen2.5-7b-instruct"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.stream = True
        self.tools = []
        self.tool_choice = None
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for key, value in kwargs.items():
            setattr(self, key, value)

    def model_dump(self, exclude_none=True):
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "system": self.system,
            "stream": self.stream,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "extra_body": self.extra_body,
            "thinking": {"enabled": self.thinking.enabled} if self.thinking else None,
        }


@pytest.fixture
def lmstudio_config():
    return ProviderConfig(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch("providers.openai_compat.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


@pytest.fixture
def lmstudio_provider(lmstudio_config):
    return LMStudioProvider(lmstudio_config)


def test_init(lmstudio_config):
    """Test provider initialization."""
    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = LMStudioProvider(lmstudio_config)
        assert provider._base_url == "http://localhost:1234/v1"
        assert provider._provider_name == "LMSTUDIO"


def test_init_uses_configurable_timeouts():
    """Test that provider passes configurable read/write/connect timeouts to client."""
    config = ProviderConfig(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        LMStudioProvider(config)
        mock_openai.assert_called_once()
        timeout = mock_openai.call_args[1]["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


def test_init_base_url_strips_trailing_slash():
    """Config with base_url trailing slash is stored without it."""
    config = ProviderConfig(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1/",
        rate_limit=10,
        rate_window=60,
    )
    with patch("openai.AsyncOpenAI"):
        provider = LMStudioProvider(config)
        assert provider._base_url == "http://localhost:1234/v1"


@pytest.mark.asyncio
async def test_stream_response(lmstudio_provider):
    """Test streaming OpenAI chat completions response."""
    req = MockRequest()

    async def mock_iter_chunks():
        # OpenAI style chunks
        # Chunk 1: Role
        delta1 = MagicMock()
        delta1.role = "assistant"
        delta1.content = None
        delta1.reasoning_content = None
        delta1.tool_calls = None
        yield MagicMock(
            choices=[MagicMock(delta=delta1, finish_reason=None)], usage=None
        )

        # Chunk 2: Content
        delta2 = MagicMock()
        delta2.role = None
        delta2.content = "Hello"
        delta2.reasoning_content = None
        delta2.tool_calls = None
        yield MagicMock(
            choices=[MagicMock(delta=delta2, finish_reason=None)], usage=None
        )

        # Chunk 3: Content
        delta3 = MagicMock()
        delta3.role = None
        delta3.content = " World"
        delta3.reasoning_content = None
        delta3.tool_calls = None
        yield MagicMock(
            choices=[MagicMock(delta=delta3, finish_reason=None)], usage=None
        )

    # We need to mock the underlying OpenAI client's request execution
    with patch.object(
        lmstudio_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=mock_iter_chunks(),
    ) as mock_create:
        events = [event async for event in lmstudio_provider.stream_response(req)]

    mock_create.assert_called_once()
    full_text = "".join(events)
    assert "message_start" in full_text
    assert "Hello" in full_text
    assert "World" in full_text
    assert "message_stop" in full_text


@pytest.mark.asyncio
async def test_stream_error_status_code(lmstudio_provider):
    """Exception during stream creation is yielded as an SSE API error."""
    req = MockRequest()

    with patch.object(
        lmstudio_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        side_effect=Exception("Connection failed"),
    ):
        events = [
            event
            async for event in lmstudio_provider.stream_response(
                req, request_id="TEST_ID"
            )
        ]

    full_text = "".join(events)
    assert_canonical_stream_error_envelope(
        events, user_message_substr="Connection failed"
    )
    assert "TEST_ID" in full_text
    assert "Connection failed" in full_text


@pytest.mark.asyncio
async def test_stream_response_omits_thinking_when_globally_disabled(lmstudio_config):
    provider = LMStudioProvider(
        lmstudio_config.model_copy(update={"enable_thinking": False})
    )
    req = MockRequest()

    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=AsyncMock(),
    ) as mock_create:

        async def empty_aiter():
            if False:
                yield None

        mock_create.return_value.__aiter__.return_value = empty_aiter()
        [e async for e in provider.stream_response(req)]

    _, kwargs = mock_create.call_args
    assert "thinking" not in str(kwargs)


@pytest.mark.asyncio
async def test_cleanup(lmstudio_provider):
    """Test that cleanup closes the client."""
    lmstudio_provider._client.close = AsyncMock()

    await lmstudio_provider.cleanup()

    lmstudio_provider._client.close.assert_called_once()


def test_build_request_body_disabled_thinking_strips_native_thinking_history(
    lmstudio_config,
):
    """Disabled thinking must omit prior assistant thinking/redacted blocks in JSON."""
    provider = LMStudioProvider(
        lmstudio_config.model_copy(update={"enable_thinking": False})
    )
    req = MockRequest(
        system=None,
        messages=[
            MockMessage("user", "hi"),
            MockMessage(
                "assistant",
                [
                    {"type": "thinking", "thinking": "t"},
                    {"type": "redacted_thinking", "data": "opaque"},
                ],
            ),
        ],
    )
    # OpenAIChatTransport doesn't have _build_request_body directly,
    # it uses _prepare_create_body which uses AnthropicToOpenAIConverter.
    # We test that the final body passed to create is correct.
    body = provider._build_request_body(req)
    assert body["messages"][1]["content"] == " "
    assert "redacted_thinking" not in str(body)
