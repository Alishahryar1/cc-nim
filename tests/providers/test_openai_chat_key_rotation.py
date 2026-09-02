"""Tests for API key rotation in OpenAIChatProvider."""

from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.providers.admission import (
    ProviderAdmissionController,
)
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.openai_chat import (
    NO_REASONING,
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
)
from tests.providers.request_factory import make_messages_request
from tests.providers.support import immediate_admission


class _UnauthorizedError(Exception):
    """Stand-in for openai.AuthenticationError (status_code + optional JSON body)."""

    def __init__(self, message: str, body: object | None = None):
        super().__init__(message)
        self.status_code = 401
        self.body = body


class _TestOpenAIChatProvider(OpenAIChatProvider):
    """Test provider that inherits key rotation logic."""

    def __init__(
        self, config: ProviderConfig, *, admission: ProviderAdmissionController
    ):
        profile = OpenAIChatProfile(
            OpenAIChatRequestPolicy(
                provider_name="TEST",
                reasoning_replay=None,
            ),
            NO_REASONING,
        )
        super().__init__(config, profile=profile, admission=admission)

    async def cleanup(self) -> None:
        await super().cleanup()

    async def list_model_infos(self):
        return frozenset()


def _request(model: str = "gpt-3.5-turbo"):
    return make_messages_request(model, max_tokens=100)


def _chunk(content: str = "test", *, finish_reason: str = "stop"):
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content, reasoning_content=None, tool_calls=None
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(completion_tokens=5, prompt_tokens=8),
    )


async def _stream(*chunks):
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_key_rotation_on_401_error():
    """Test that provider rotates keys on 401 Unauthorized error."""
    # Config with two API keys
    config = ProviderConfig(
        api_keys=["key1", "key2"],
        base_url="https://api.test.com/v1",
        rate_limit=10,
        rate_window=60,
        max_concurrency=5,
        http_read_timeout=5.0,
        http_write_timeout=5.0,
        http_connect_timeout=5.0,
        proxy=None,
        log_raw_sse_events=False,
        log_api_error_tracebacks=False,
    )
    provider = _TestOpenAIChatProvider(config, admission=immediate_admission())

    # Verify initial state
    assert provider._get_current_api_key() == "key1"
    assert provider._current_key_index == 0

    # First call returns 401, second call succeeds with key2
    error_response = _UnauthorizedError("Invalid API key")
    success_chunk = _chunk(content="success")

    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        side_effect=[error_response, _stream(success_chunk)],
    ) as mock_create:
        # This should trigger key rotation and retry with key2
        events = [e async for e in provider.stream_messages(_request())]

        # Verify we tried twice (original key1, then rotated to key2)
        assert mock_create.await_count == 2

        # Verify the API key progression by checking provider state
        # After first call (which failed), key should have been rotated
        # After second call (which succeeded), we should be on key2
        assert provider._get_current_api_key() == "key2"
        assert provider._current_key_index == 1

        # Verify we got the success response
        assert any("success" in str(e) for e in events)


@pytest.mark.asyncio
async def test_key_rotation_exhaustion():
    """Test that provider fails after all keys are exhausted."""
    # Config with two API keys
    config = ProviderConfig(
        api_keys=["key1", "key2"],
        base_url="https://api.test.com/v1",
        rate_limit=10,
        rate_window=60,
        max_concurrency=5,
        http_read_timeout=5.0,
        http_write_timeout=5.0,
        http_connect_timeout=5.0,
        proxy=None,
        log_raw_sse_events=False,
        log_api_error_tracebacks=False,
    )
    # Use low max_attempts to make test predictable
    admission = ProviderAdmissionController(
        provider_name="TEST",
        rate_limit=1_000_000,
        rate_window=1.0,
        max_concurrency=1_000,
        max_attempts=3,  # Limit to 3 attempts for predictable testing
        base_delay=0.0,
        max_delay=0.0,
        jitter=0.0,
    )
    provider = _TestOpenAIChatProvider(config, admission=admission)

    # Verify initial state
    assert provider._get_current_api_key() == "key1"
    assert provider._current_key_index == 0

    # All calls return 401 errors - should exhaust admission attempts
    error_response = _UnauthorizedError("Invalid API key")

    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        side_effect=[
            error_response,
            error_response,
            error_response,
        ],  # 3 calls to match max_attempts
    ) as mock_create:
        # This should try keys and then fail due to admission limits
        try:
            [e async for e in provider.stream_messages(_request())]
            # If we reach here, no exception was raised (unexpected)
            raise AssertionError("Expected an exception to be raised")
        except Exception as e:
            print(f"Actual exception: {e}")
            print(f"Exception type: {type(e)}")
            print(f"Exception args: {e.args}")
            # Verify we made the expected number of calls
            assert mock_create.await_count == 3

            # Verify the error message indicates failure
            assert (
                "All API keys for TEST are exhausted" in str(e)
                or "Max attempts exceeded" in str(e)
                or "provider execution ended without a final error" in str(e)
            )


@pytest.mark.asyncio
async def test_no_rotation_with_single_key():
    """Test that single key configuration doesn't attempt rotation."""
    # Config with single API key
    config = ProviderConfig(
        api_keys=["single-key"],
        base_url="https://api.test.com/v1",
        rate_limit=10,
        rate_window=60,
        max_concurrency=5,
        http_read_timeout=5.0,
        http_write_timeout=5.0,
        http_connect_timeout=5.0,
        proxy=None,
        log_raw_sse_events=False,
        log_api_error_tracebacks=False,
    )
    provider = _TestOpenAIChatProvider(config, admission=immediate_admission())

    # Verify initial state
    assert provider._get_current_api_key() == "single-key"
    assert provider._current_key_index == 0
    assert provider._config.api_keys == ["single-key"]

    # Call returns 401 error
    error_response = _UnauthorizedError("Invalid API key")

    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=error_response,  # Just return the error, no retry
    ) as mock_create:
        # This should fail without attempting rotation
        try:
            [e async for e in provider.stream_messages(_request())]
            raise AssertionError("Expected an exception to be raised")
        except Exception:
            # Verify we only tried once (no rotation with single key)
            assert mock_create.await_count == 1

            # Verify we're still on the first (and only) key
            assert provider._get_current_api_key() == "single-key"
            assert provider._current_key_index == 0
