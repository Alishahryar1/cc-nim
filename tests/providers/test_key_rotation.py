"""API-key rotation pool behavior."""

from __future__ import annotations

import httpx
import openai

from providers.key_rotation import (
    ApiKeyRotationMode,
    ApiKeyRotationPool,
    is_limit_error,
)


def test_round_robin_mode_cycles_keys_per_request() -> None:
    pool = ApiKeyRotationPool(" key-a, key-b ,key-c ", ApiKeyRotationMode.ROUND_ROBIN)

    assert [pool.key_for_new_request() for _ in range(5)] == [
        "key-a",
        "key-b",
        "key-c",
        "key-a",
        "key-b",
    ]


def test_failover_mode_keeps_first_key_until_limit_error() -> None:
    pool = ApiKeyRotationPool("key-a,key-b,key-c", ApiKeyRotationMode.FAILOVER_ON_LIMIT)

    assert pool.key_for_new_request() == "key-a"
    assert pool.key_for_new_request() == "key-a"
    assert pool.next_key_after_limit("key-a") == "key-b"
    assert pool.next_key_after_limit("key-b") == "key-c"
    assert pool.next_key_after_limit("key-c") is None


def test_single_key_has_no_failover_candidate() -> None:
    pool = ApiKeyRotationPool("only-key", ApiKeyRotationMode.FAILOVER_ON_LIMIT)

    assert pool.key_for_new_request() == "only-key"
    assert pool.next_key_after_limit("only-key") is None


def test_limit_error_detection_supports_openai_and_httpx_429() -> None:
    request = httpx.Request("POST", "https://example.test/v1/messages")
    response = httpx.Response(429, request=request)

    assert is_limit_error(
        httpx.HTTPStatusError("limit", request=request, response=response)
    )
    assert is_limit_error(
        openai.RateLimitError(
            "limit",
            response=response,
            body={},
        )
    )
