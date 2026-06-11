"""Tests for the Hugging Face Inference Providers (OpenAI-compatible) adapter."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.huggingface import HUGGINGFACE_DEFAULT_BASE, HuggingFaceProvider


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "meta-llama/Llama-3.3-70B-Instruct"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def huggingface_config():
    return ProviderConfig(
        api_key="test_huggingface_key",
        base_url=HUGGINGFACE_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""

    @asynccontextmanager
    async def _slot():
        yield

    with patch("providers.openai_compat.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def huggingface_provider(huggingface_config):
    return HuggingFaceProvider(huggingface_config)


def test_init(huggingface_config):
    """Test provider initialization."""
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = HuggingFaceProvider(huggingface_config)
        assert provider._api_key == "test_huggingface_key"
        assert provider._base_url == HUGGINGFACE_DEFAULT_BASE
        mock_openai.assert_called_once()


def test_default_base_url_constant():
    assert HUGGINGFACE_DEFAULT_BASE == "https://router.huggingface.co/v1"


def test_build_request_body_basic(huggingface_provider):
    """Basic request body conversion attaches system message from Claude request."""
    req = MockRequest()
    body = huggingface_provider._build_request_body(req)

    assert body["model"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert body["messages"][0]["role"] == "system"
    assert body["max_tokens"] == 100


def test_build_request_body_keeps_provider_suffixed_model(huggingface_provider):
    """Hub model ids with an explicit ``:provider`` suffix pass through unchanged."""
    req = MockRequest(model="deepseek-ai/DeepSeek-V3:fireworks-ai")
    body = huggingface_provider._build_request_body(req)

    assert body["model"] == "deepseek-ai/DeepSeek-V3:fireworks-ai"


def test_build_request_body_global_disable_blocks_reasoning_mapping():
    provider = HuggingFaceProvider(
        ProviderConfig(
            api_key="test_huggingface_key",
            base_url=HUGGINGFACE_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
            enable_thinking=False,
        )
    )
    req = MockRequest()
    body = provider._build_request_body(req)

    roles = [m.get("role") for m in body.get("messages", [])]
    assert "assistant_reasoning_content" not in roles


def test_build_request_body_preserves_caller_extra_body(huggingface_provider):
    req = MockRequest(extra_body={"custom_flag": True})

    body = huggingface_provider._build_request_body(req)

    eb = body.get("extra_body")
    assert isinstance(eb, dict)
    assert eb.get("custom_flag") is True


def test_build_request_body_merges_extra_body_with_caller_precedence(
    huggingface_provider,
):
    """Caller extra_body keys merge into (not replace) any converter-set entry."""
    with patch(
        "providers.huggingface.request.build_base_request_body"
    ) as mock_convert:
        mock_convert.return_value = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"converter_key": 1, "shared_key": "converter"},
        }
        req = MockRequest(extra_body={"caller_key": 2, "shared_key": "caller"})
        body = huggingface_provider._build_request_body(req)

    assert body["extra_body"] == {
        "converter_key": 1,
        "caller_key": 2,
        "shared_key": "caller",
    }


@pytest.mark.asyncio
async def test_stream_response_text(huggingface_provider):
    """Text content deltas are emitted as text blocks."""
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello back!",
                reasoning_content=None,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        huggingface_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in huggingface_provider.stream_response(req)]

        assert any(
            '"text_delta"' in event and "Hello back!" in event for event in events
        )


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(huggingface_provider):
    """reasoning_content deltas are emitted as thinking blocks."""
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                reasoning_content="Thinking...",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        huggingface_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in huggingface_provider.stream_response(req)]

        assert any(
            '"thinking_delta"' in event and "Thinking..." in event for event in events
        )


_ROUTER_MAX_TOKENS_400 = (
    "Error code: 400 - {'message': '`max_tokens` must be less than or equal to "
    "`32768`, the maximum value for `max_tokens` is less than the "
    "`context_window` for this model', 'type': 'invalid_request_error', "
    "'param': 'max_tokens'}"
)


class _Stub400(Exception):
    status_code = 400


def test_retry_body_clamps_max_tokens_to_router_limit(huggingface_provider):
    body = {"model": "m", "messages": [], "max_tokens": 81920}

    retry_body = huggingface_provider._get_retry_request_body(
        _Stub400(_ROUTER_MAX_TOKENS_400), body
    )

    assert retry_body is not None
    assert retry_body["max_tokens"] == 32768
    assert body["max_tokens"] == 81920  # original body untouched


def test_retry_body_none_when_max_tokens_already_within_limit(huggingface_provider):
    body = {"model": "m", "messages": [], "max_tokens": 100}

    retry_body = huggingface_provider._get_retry_request_body(
        _Stub400(_ROUTER_MAX_TOKENS_400), body
    )

    assert retry_body is None


def test_retry_body_none_for_non_400_errors(huggingface_provider):
    body = {"model": "m", "messages": [], "max_tokens": 81920}

    retry_body = huggingface_provider._get_retry_request_body(
        RuntimeError(_ROUTER_MAX_TOKENS_400), body
    )

    assert retry_body is None


def test_retry_body_none_for_unrelated_400_errors(huggingface_provider):
    body = {"model": "m", "messages": [], "max_tokens": 81920}

    retry_body = huggingface_provider._get_retry_request_body(
        _Stub400("model not found"), body
    )

    assert retry_body is None


def test_retry_body_resamples_once_on_tool_use_failed(huggingface_provider):
    body = {"model": "m", "messages": [], "max_tokens": 100}
    error = _Stub400(
        "Error code: 400 - {'message': 'Failed to call a function. Please adjust "
        "your prompt.', 'type': 'invalid_request_error', 'code': 'tool_use_failed'}"
    )

    retry_body = huggingface_provider._get_retry_request_body(error, body)

    assert retry_body == body
    assert retry_body is not body  # cloned, not the same object


@pytest.mark.asyncio
async def test_cleanup(huggingface_provider):
    huggingface_provider._client = AsyncMock()

    await huggingface_provider.cleanup()

    huggingface_provider._client.close.assert_called_once()
