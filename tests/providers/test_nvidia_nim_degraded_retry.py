"""NVIDIA Cloud Function deployment failures use provider-owned retry semantics."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import openai
import pytest

from free_claude_code.config.nim import NimSettings
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.failure_policy import (
    overloaded_provider_failure,
    retryable_upstream_status,
)
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.open_router import OpenRouterProvider
from free_claude_code.providers.rate_limit import (
    DEFAULT_UPSTREAM_MAX_RETRIES,
    UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS,
    ProviderRateLimiter,
)
from tests.providers.request_factory import make_messages_request

_FUNCTION_ID = "87ea0ddc-cff1-4bca-bf8b-3bd98a35ddd0"
_DEGRADED_DETAIL = f"Function id '{_FUNCTION_ID}': DEGRADED function cannot be invoked"


def _config(base_url: str) -> ProviderConfig:
    return ProviderConfig(
        api_key="test_key",
        base_url=base_url,
        rate_limit=1_000_000,
        rate_window=1,
        max_concurrency=1_000,
        http_read_timeout=30.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )


def _limiter() -> ProviderRateLimiter:
    return ProviderRateLimiter(
        rate_limit=1_000_000,
        rate_window=1.0,
        max_concurrency=1_000,
    )


def _bad_request(
    detail: str = _DEGRADED_DETAIL,
    *,
    body_extra: dict[str, str] | None = None,
) -> openai.BadRequestError:
    request = httpx.Request(
        "POST", "https://integrate.api.nvidia.com/v1/chat/completions"
    )
    response = httpx.Response(400, request=request)
    body: dict[str, object] = {
        "status": 400,
        "title": "Bad Request",
        "detail": detail,
    }
    if body_extra is not None:
        body.update(body_extra)
    return openai.BadRequestError("Bad Request", response=response, body=body)


def _successful_stream(text: str = "Recovered"):
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(
            delta=MagicMock(content=text, reasoning_content=""),
            finish_reason="stop",
        )
    ]
    chunk.usage = None

    async def stream():
        yield chunk

    return stream()


def _nim(limiter: ProviderRateLimiter) -> NvidiaNimProvider:
    return NvidiaNimProvider(
        _config("https://integrate.api.nvidia.com/v1"),
        nim_settings=NimSettings(),
        rate_limiter=limiter,
    )


@pytest.mark.asyncio
async def test_degraded_function_retries_unchanged_request_then_succeeds() -> None:
    limiter = _limiter()
    provider = _nim(limiter)

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=[_bad_request(), _successful_stream()],
        ) as create,
        patch.object(limiter, "extend_reactive_block") as extend_block,
        patch(
            "free_claude_code.providers.rate_limit.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep,
    ):
        events = [
            event
            async for event in provider.stream_response(
                make_messages_request(), request_id="req_recovered"
            )
        ]

    assert create.await_count == 2
    assert create.call_args_list[0].kwargs == create.call_args_list[1].kwargs
    extend_block.assert_called_once()
    sleep.assert_awaited_once()
    event_text = "".join(events)
    assert "Recovered" in event_text
    assert "event: message_stop" in event_text
    assert "event: error" not in event_text


@pytest.mark.asyncio
async def test_degraded_function_exhaustion_is_detailed_redacted_overload() -> None:
    limiter = _limiter()
    provider = _nim(limiter)
    error = _bad_request(
        body_extra={
            "authorization": "Bearer NIM_AUTH_SECRET",
            "api_key": "NIM_API_SECRET",
        }
    )

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ) as create,
        patch.object(limiter, "extend_reactive_block") as extend_block,
        patch(
            "free_claude_code.providers.rate_limit.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep,
        patch("free_claude_code.providers.openai_chat.provider.trace_event") as trace,
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        [
            event
            async for event in provider.stream_response(
                make_messages_request(), request_id="req_degraded"
            )
        ]

    assert create.await_count == UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS
    assert extend_block.call_count == DEFAULT_UPSTREAM_MAX_RETRIES
    assert sleep.await_count == DEFAULT_UPSTREAM_MAX_RETRIES

    failure = exc_info.value
    assert failure.kind is FailureKind.OVERLOADED
    assert failure.status_code == 529
    assert failure.retryable is True
    assert "Upstream provider NIM returned HTTP 400." in failure.message
    assert _DEGRADED_DETAIL in failure.message
    assert "Request ID: req_degraded" in failure.message
    assert "NIM_AUTH_SECRET" not in failure.message
    assert "NIM_API_SECRET" not in failure.message

    error_traces = [
        call.kwargs
        for call in trace.call_args_list
        if call.kwargs.get("event") == "provider.response.error"
    ]
    assert error_traces[-1]["exc_type"] == "BadRequestError"
    assert error_traces[-1]["failure_kind"] == "overloaded"
    assert error_traces[-1]["status_code"] == 529
    assert error_traces[-1]["provider_retryable"] is True
    assert "error_message" not in error_traces[-1]


@pytest.mark.parametrize(
    "detail",
    [
        "Unsupported field: top_k",
        "Validation failed: DEGRADED function cannot be invoked",
        f"Function id '{_FUNCTION_ID}': DEGRADING function cannot be invoked",
        f"Function id '{_FUNCTION_ID}': DEGRADED function is waiting",
    ],
)
@pytest.mark.asyncio
async def test_unrelated_nim_bad_request_is_not_retried(detail: str) -> None:
    limiter = _limiter()
    provider = _nim(limiter)

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=_bad_request(detail),
        ) as create,
        patch.object(limiter, "extend_reactive_block") as extend_block,
        patch(
            "free_claude_code.providers.rate_limit.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep,
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        [event async for event in provider.stream_response(make_messages_request())]

    assert create.await_count == 1
    extend_block.assert_not_called()
    sleep.assert_not_awaited()
    assert exc_info.value.kind is FailureKind.INVALID_REQUEST
    assert exc_info.value.status_code == 400
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_degraded_wording_remains_non_retryable_for_other_providers() -> None:
    limiter = _limiter()
    provider = OpenRouterProvider(
        _config("https://openrouter.ai/api/v1"), rate_limiter=limiter
    )
    error = _bad_request()

    assert retryable_upstream_status(error) is None

    with (
        patch.object(
            provider._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=error,
        ) as create,
        patch.object(limiter, "extend_reactive_block") as extend_block,
        patch(
            "free_claude_code.providers.rate_limit.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep,
        pytest.raises(ExecutionFailure) as exc_info,
    ):
        [event async for event in provider.stream_response(make_messages_request())]

    assert create.await_count == 1
    extend_block.assert_not_called()
    sleep.assert_not_awaited()
    assert exc_info.value.kind is FailureKind.INVALID_REQUEST
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_limiter_override_preserves_raw_exception_after_exhaustion() -> None:
    limiter = _limiter()
    errors = (_bad_request(), _bad_request())
    attempts = 0

    async def fail() -> None:
        nonlocal attempts
        error = errors[attempts]
        attempts += 1
        raise error

    override = Mock(return_value=overloaded_provider_failure())
    with (
        patch.object(limiter, "extend_reactive_block"),
        patch(
            "free_claude_code.providers.rate_limit.asyncio.sleep",
            new_callable=AsyncMock,
        ),
        pytest.raises(openai.BadRequestError) as exc_info,
    ):
        await limiter.execute_with_retry(
            fail,
            provider_failure_override=override,
            max_retries=1,
        )

    assert attempts == 2
    assert override.call_count == 2
    assert exc_info.value is errors[-1]
