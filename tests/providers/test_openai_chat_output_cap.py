"""Tests for OpenAI-compatible output-token cap recovery (issue #955).

Covers the pure parse/clamp helpers and the transport behavior that clamps
``max_completion_tokens``/``max_tokens`` to the upstream maximum, retries once,
and learns the cap so later requests clamp proactively.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.groq import GROQ_DEFAULT_BASE, GroqProvider
from providers.transports.openai_chat.output_cap import (
    clamp_output_tokens,
    parse_output_token_cap,
)


class _BadRequest(Exception):
    """Stand-in for openai.BadRequestError (status_code + optional JSON body)."""

    def __init__(self, message: str, body: object | None = None):
        super().__init__(message)
        self.status_code = 400
        self.body = body


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_parse_cap_from_groq_message():
    error = _BadRequest(
        "max_completion_tokens must be less than or equal to 40960, the maximum "
        "value for max_completion_tokens is less than the context_window for this model"
    )
    assert parse_output_token_cap(error) == 40960


@pytest.mark.parametrize(
    "message,expected",
    [
        ("max_tokens: maximum value is 8192", 8192),
        ("max_tokens must not exceed 16000", 16000),
        ("`max_completion_tokens` <= 4096 required", 4096),
        ("max_tokens at most 2048 allowed", 2048),
        ("maximum allowed value of 32768 for max_tokens", 32768),
    ],
)
def test_parse_cap_various_phrasings(message, expected):
    assert parse_output_token_cap(_BadRequest(message)) == expected


def test_parse_cap_reads_json_body():
    error = _BadRequest(
        "invalid request",
        body={"error": {"param": "max_completion_tokens", "message": "<= 12000"}},
    )
    assert parse_output_token_cap(error) == 12000


def test_parse_cap_ignores_non_400():
    error = _BadRequest("max_tokens must be less than or equal to 40960")
    error.status_code = 500
    assert parse_output_token_cap(error) is None


def test_parse_cap_ignores_unrelated_400():
    assert parse_output_token_cap(_BadRequest("temperature must be <= 2")) is None


def test_parse_cap_returns_none_without_number():
    assert (
        parse_output_token_cap(_BadRequest("max_tokens is larger than allowed")) is None
    )


def test_clamp_reduces_max_completion_tokens():
    assert clamp_output_tokens({"max_completion_tokens": 64000}, 40960) == {
        "max_completion_tokens": 40960
    }


def test_clamp_reduces_max_tokens():
    assert clamp_output_tokens({"max_tokens": 100000}, 8192) == {"max_tokens": 8192}


def test_clamp_noop_when_within_cap_returns_none():
    assert clamp_output_tokens({"max_completion_tokens": 1000}, 40960) is None


def test_clamp_does_not_mutate_input():
    body = {"max_tokens": 99999, "model": "m"}
    clamped = clamp_output_tokens(body, 8192)
    assert body["max_tokens"] == 99999
    assert clamped is not None
    assert clamped["max_tokens"] == 8192


def test_clamp_ignores_bool_values():
    assert clamp_output_tokens({"max_tokens": True}, 8192) is None


# --------------------------------------------------------------------------- #
# Transport integration (via GroqProvider, which uses max_completion_tokens)
# --------------------------------------------------------------------------- #


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, max_tokens=64000):
        self.model = "llama-3.3-70b-versatile"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = max_tokens
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.thinking = MagicMock()
        self.thinking.enabled = False


@pytest.fixture(autouse=True)
def mock_rate_limiter():
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
def groq_provider():
    return GroqProvider(
        ProviderConfig(
            api_key="test_groq_key",
            base_url=GROQ_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
            enable_thinking=False,
        )
    )


@pytest.mark.asyncio
async def test_create_stream_clamps_and_learns_on_cap_rejection(groq_provider):
    body = groq_provider._build_request_body(MockRequest(max_tokens=64000))
    assert body["max_completion_tokens"] == 64000
    model = body["model"]

    error = _BadRequest("max_completion_tokens must be less than or equal to 40960")
    create = AsyncMock(side_effect=[error, object()])

    with patch.object(groq_provider._client.chat.completions, "create", create):
        _stream, used_body = await groq_provider._create_stream(body)

    assert create.call_count == 2
    assert create.call_args_list[1].kwargs["max_completion_tokens"] == 40960
    assert used_body["max_completion_tokens"] == 40960
    assert groq_provider._model_output_caps[model] == 40960


@pytest.mark.asyncio
async def test_learned_cap_clamps_next_request_without_a_400(groq_provider):
    body = groq_provider._build_request_body(MockRequest(max_tokens=64000))
    model = body["model"]
    groq_provider._model_output_caps[model] = 40960

    create = AsyncMock(return_value=object())
    with patch.object(groq_provider._client.chat.completions, "create", create):
        _stream, used_body = await groq_provider._create_stream(body)

    assert create.call_count == 1
    assert create.call_args.kwargs["max_completion_tokens"] == 40960
    assert used_body["max_completion_tokens"] == 40960


@pytest.mark.asyncio
async def test_unrelated_400_is_not_clamped_and_propagates(groq_provider):
    body = groq_provider._build_request_body(MockRequest(max_tokens=100))
    create = AsyncMock(side_effect=_BadRequest("messages: invalid role 'wizard'"))

    with (
        patch.object(groq_provider._client.chat.completions, "create", create),
        pytest.raises(Exception, match="wizard"),
    ):
        await groq_provider._create_stream(body)

    assert create.call_count == 1
    assert groq_provider._model_output_caps == {}
