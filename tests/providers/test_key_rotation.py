"""Tests for API key rotation functionality in base provider."""

from free_claude_code.providers.base import BaseProvider, ProviderConfig


class _TestProvider(BaseProvider):
    """Test provider implementation for key rotation testing."""

    def preflight_messages(
        self,
        request,
        *,
        reasoning=None,
    ) -> None:
        pass

    def preflight_responses(
        self,
        request,
        *,
        reasoning=None,
    ) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    async def list_model_infos(self):
        return frozenset()

    def stream_messages(
        self,
        request,
        input_tokens=0,
        *,
        request_id=None,
        response_model=None,
        reasoning=None,
    ):
        # Dummy implementation for testing
        async def dummy_stream():
            yield ""

        return dummy_stream()

    def stream_responses(
        self,
        request,
        input_tokens=0,
        *,
        request_id=None,
        response_model=None,
        reasoning=None,
    ):
        # Dummy implementation for testing
        async def dummy_stream():
            yield ""

        return dummy_stream()


def test_single_api_key():
    """Test that single API key works correctly."""
    config = ProviderConfig(
        api_keys=["single-key"],
        base_url="https://test.example.com",
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
    provider = _TestProvider(config)

    # Should return the single key
    assert provider._get_current_api_key() == "single-key"
    assert provider._current_key_index == 0

    # Rotating should not change anything with single key
    provider._rotate_api_key()
    assert provider._get_current_api_key() == "single-key"
    assert provider._current_key_index == 0


def test_multiple_api_keys_rotation():
    """Test that multiple API keys rotate correctly."""
    config = ProviderConfig(
        api_keys=["key1", "key2", "key3"],
        base_url="https://test.example.com",
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
    provider = _TestProvider(config)

    # Should start with first key
    assert provider._get_current_api_key() == "key1"
    assert provider._current_key_index == 0

    # Rotate to second key
    provider._rotate_api_key()
    assert provider._get_current_api_key() == "key2"
    assert provider._current_key_index == 1

    # Rotate to third key
    provider._rotate_api_key()
    assert provider._get_current_api_key() == "key3"
    assert provider._current_key_index == 2

    # Rotate back to first key
    provider._rotate_api_key()
    assert provider._get_current_api_key() == "key1"
    assert provider._current_key_index == 0


def test_key_failure_tracking():
    """Test that key failures are tracked correctly."""
    config = ProviderConfig(
        api_keys=["key1", "key2", "key3"],
        base_url="https://test.example.com",
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
    provider = _TestProvider(config)

    # Initially no failures
    assert provider._key_failure_count == {}
    assert not provider._is_key_exhausted(0)

    # Mark key 0 as failed once
    provider._mark_key_failed()
    assert provider._key_failure_count == {0: 1}
    assert not provider._is_key_exhausted(0)  # Default threshold is 3

    # Mark key 0 as failed twice more
    provider._mark_key_failed()
    provider._mark_key_failed()
    assert provider._key_failure_count == {0: 3}
    assert provider._is_key_exhausted(0)  # Now exhausted

    # Other keys should not be exhausted
    assert not provider._is_key_exhausted(1)
    assert not provider._is_key_exhausted(2)


def test_get_next_available_key_index():
    """Test finding the next available key index."""
    config = ProviderConfig(
        api_keys=["key1", "key2", "key3", "key4"],
        base_url="https://test.example.com",
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
    provider = _TestProvider(config)

    # All keys available, should return current index (0)
    assert provider._get_next_available_key_index() == 0

    # Exhaust key 0
    provider._key_failure_count[0] = 3  # At threshold
    assert provider._get_next_available_key_index() == 1  # Should skip to key 1

    # Exhaust keys 0 and 1
    provider._key_failure_count[1] = 3
    assert provider._get_next_available_key_index() == 2  # Should skip to key 2

    # Exhaust keys 0, 1, and 2
    provider._key_failure_count[2] = 3
    assert provider._get_next_available_key_index() == 3  # Should skip to key 3

    # Exhaust all keys
    provider._key_failure_count[3] = 3
    assert provider._get_next_available_key_index() is None  # No available keys


def test_attempt_with_key_rotation():
    """Test attempting to find a usable key through rotation."""
    config = ProviderConfig(
        api_keys=["key1", "key2", "key3"],
        base_url="https://test.example.com",
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
    provider = _TestProvider(config)

    # All keys available, should succeed without rotation
    assert provider._attempt_with_key_rotation()
    assert provider._current_key_index == 0  # Should remain unchanged

    # Exhaust current key
    provider._key_failure_count[0] = 3
    # When we attempt rotation, it will find key 1 usable (not exhausted)
    # Failure counts should NOT be reset during rotation (preserves exhaustion detection)
    assert provider._attempt_with_key_rotation()  # Should find key 1
    assert provider._current_key_index == 1  # Should have rotated
    # Key 0 should still be exhausted (failure count preserved)
    assert provider._key_failure_count[0] == 3

    # Exhaust the new current key (key 1)
    provider._key_failure_count[1] = 3
    # When we attempt rotation, it will find key 2 usable (not exhausted)
    # Failure counts should NOT be reset during rotation
    assert provider._attempt_with_key_rotation()  # Should find key 2
    assert provider._current_key_index == 2  # Should have rotated
    # Keys 0 and 1 should still be exhausted (failure counts preserved)
    assert provider._key_failure_count[0] == 3
    assert provider._key_failure_count[1] == 3

    # Exhaust the new current key (key 2)
    provider._key_failure_count[2] = 3
    # When we attempt rotation, all keys are exhausted, so no usable key should be found
    assert not provider._attempt_with_key_rotation()  # Should NOT find any usable key
    assert (
        provider._current_key_index == 2
    )  # Should have rotated through all keys and ended at key 2
    # All keys should be exhausted (failure counts preserved)
    assert provider._key_failure_count[0] == 3
    assert provider._key_failure_count[1] == 3
    assert provider._key_failure_count[2] == 3

    # Only when there are no keys should it return False
    config_no_keys = ProviderConfig(
        api_keys=[],
        base_url="https://test.example.com",
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
    provider_no_keys = _TestProvider(config_no_keys)
    assert not provider_no_keys._attempt_with_key_rotation()


def test_no_api_keys():
    """Test behavior when no API keys are configured."""
    config = ProviderConfig(
        api_keys=[],
        base_url="https://test.example.com",
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
    provider = _TestProvider(config)

    # Should return None for no keys
    assert provider._get_current_api_key() is None

    # Rotating should do nothing
    provider._rotate_api_key()  # Should not raise
    assert provider._get_current_api_key() is None

    # Failure tracking should do nothing
    provider._mark_key_failed()  # Should not raise
    assert provider._key_failure_count == {}

    # Exhaustion checks should return False/None
    assert not provider._is_key_exhausted(0)
    assert provider._get_next_available_key_index() is None
    assert not provider._attempt_with_key_rotation()
