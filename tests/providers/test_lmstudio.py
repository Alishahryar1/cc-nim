"""Tests for LM Studio (OpenAI-compatible chat completions) provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from providers.base import ProviderConfig
from providers.exceptions import InvalidRequestError
from providers.lmstudio import LMStudioProvider
from providers.lmstudio.client import LMSTUDIO_DEFAULT_BASE


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
        self.tools = []
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def lmstudio_config():
    return ProviderConfig(
        api_key="lm-studio",
        base_url=LMSTUDIO_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""

    @asynccontextmanager
    async def _slot():
        yield

    with patch("providers.transports.openai_chat.transport.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def lmstudio_provider(lmstudio_config):
    return LMStudioProvider(lmstudio_config)


def test_init(lmstudio_config):
    """Test provider initialization."""
    with patch("providers.transports.openai_chat.transport.AsyncOpenAI") as mock_openai:
        provider = LMStudioProvider(lmstudio_config)
        assert provider._api_key == "lm-studio"
        assert provider._base_url == LMSTUDIO_DEFAULT_BASE
        assert provider._provider_name == "LMSTUDIO"
        mock_openai.assert_called_once()


def test_default_base_url_constant():
    assert LMSTUDIO_DEFAULT_BASE == "http://localhost:1234/v1"


def test_build_request_body_basic(lmstudio_provider):
    req = MockRequest()
    body = lmstudio_provider._build_request_body(req)

    assert body["model"] == "lmstudio-community/qwen2.5-7b-instruct"
    assert body["messages"][0]["role"] == "system"


def test_build_request_body_never_replays_prior_thinking(lmstudio_provider):
    """Mistral-family templates have no assistant reasoning field; prior-turn
    thinking must never be replayed regardless of the enable_thinking setting."""
    req = MockRequest(
        messages=[
            MockMessage("user", "hi"),
            MockMessage(
                "assistant",
                [{"type": "thinking", "thinking": "prior reasoning", "signature": "s"}],
            ),
        ]
    )
    body = lmstudio_provider._build_request_body(req)

    roles = [m.get("role") for m in body.get("messages", [])]
    assert "assistant_reasoning_content" not in roles
    assert "prior reasoning" not in str(body)


@pytest.mark.asyncio
async def test_stream_response_text(lmstudio_provider):
    """Text content deltas are emitted as text blocks (via the OpenAI chat transport)."""
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Hello back!", reasoning_content=None, tool_calls=None
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5, prompt_tokens=10)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        lmstudio_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in lmstudio_provider.stream_response(req)]

        assert any(
            '"text_delta"' in event and "Hello back!" in event for event in events
        )


@pytest.mark.asyncio
async def test_cleanup(lmstudio_provider):
    lmstudio_provider._client = AsyncMock()
    await lmstudio_provider.cleanup()


# --- Context-budget preflight (new: guards against LM Studio's silent
# mid-stream truncation when a prompt exceeds the loaded model's context) ---


def test_preflight_context_budget_noop_when_context_length_unknown(lmstudio_provider):
    """No LM Studio /api/v0/models data available -> preflight is a no-op (fail open)."""
    with patch.object(lmstudio_provider, "_loaded_context_length", return_value=None):
        lmstudio_provider._preflight_context_budget(MockRequest())  # must not raise


def test_preflight_context_budget_allows_request_under_budget(lmstudio_provider):
    with patch.object(
        lmstudio_provider, "_loaded_context_length", return_value=100_000
    ):
        req = MockRequest(messages=[MockMessage("user", "hi")], system=None, tools=[])
        lmstudio_provider._preflight_context_budget(req)  # must not raise


def test_preflight_context_budget_rejects_request_over_90_percent(lmstudio_provider):
    with (
        patch.object(lmstudio_provider, "_loaded_context_length", return_value=1000),
        patch(
            "providers.lmstudio.client.get_token_count",
            return_value=901,
        ),
        pytest.raises(InvalidRequestError, match="prompt is too long"),
    ):
        lmstudio_provider._preflight_context_budget(MockRequest())


def test_loaded_context_length_reads_max_across_loaded_models(lmstudio_provider):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "data": [
            {"state": "loaded", "loaded_context_length": 40960},
            {"state": "loaded", "loaded_context_length": 8192},
            {"state": "not-loaded", "loaded_context_length": 999999},
        ]
    }
    with patch(
        "providers.lmstudio.client.httpx.get", return_value=response
    ) as mock_get:
        value = lmstudio_provider._loaded_context_length()

    assert value == 40960
    mock_get.assert_called_once()
    assert mock_get.call_args[0][0] == "http://localhost:1234/api/v0/models"


def test_loaded_context_length_fails_open_on_error(lmstudio_provider):
    with patch(
        "providers.lmstudio.client.httpx.get",
        side_effect=httpx.ConnectError("refused"),
    ):
        assert lmstudio_provider._loaded_context_length() is None


def test_loaded_context_length_is_cached_within_ttl(lmstudio_provider):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "data": [{"state": "loaded", "loaded_context_length": 40960}]
    }
    with patch(
        "providers.lmstudio.client.httpx.get", return_value=response
    ) as mock_get:
        first = lmstudio_provider._loaded_context_length()
        second = lmstudio_provider._loaded_context_length()

    assert first == second == 40960
    mock_get.assert_called_once()
