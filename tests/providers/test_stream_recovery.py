"""Provider-owned stream retry and holdback policy."""

import httpx
import openai

from free_claude_code.providers.stream_recovery import (
    EARLY_TRANSPARENT_MAX_RETRIES,
    EARLY_TRANSPARENT_TOTAL_ATTEMPTS,
    MIDSTREAM_RECOVERY_ATTEMPTS,
    RecoveryController,
    RecoveryFailureAction,
    RecoveryHoldbackBuffer,
    is_retryable_stream_error,
)


def _statusless_openai_api_error(
    message: str, body: object | None = None
) -> openai.APIError:
    return openai.APIError(
        message,
        request=httpx.Request("POST", "https://provider.test/messages"),
        body=body,
    )


def test_early_transparent_retry_total_attempts_is_five() -> None:
    assert EARLY_TRANSPARENT_TOTAL_ATTEMPTS == 5
    assert EARLY_TRANSPARENT_MAX_RETRIES == 4


def test_midstream_recovery_attempts_total_is_five() -> None:
    assert MIDSTREAM_RECOVERY_ATTEMPTS == 5


def test_retryable_stream_error_classifies_transport_and_http_status() -> None:
    assert is_retryable_stream_error(httpx.ReadError("cut off"))

    request = httpx.Request("GET", "https://example.test")
    assert is_retryable_stream_error(
        httpx.HTTPStatusError(
            "server error", request=request, response=httpx.Response(503)
        )
    )
    assert not is_retryable_stream_error(
        httpx.HTTPStatusError(
            "bad request", request=request, response=httpx.Response(400)
        )
    )


def test_stream_retry_preserves_timeout_scope() -> None:
    request = httpx.Request("POST", "https://provider.test/messages")

    assert is_retryable_stream_error(httpx.ReadTimeout("read", request=request))
    assert not is_retryable_stream_error(
        httpx.ConnectTimeout("connect", request=request)
    )
    assert not is_retryable_stream_error(httpx.WriteTimeout("write", request=request))
    assert not is_retryable_stream_error(httpx.PoolTimeout("pool", request=request))


def test_retryable_stream_error_classifies_statusless_api_error_body_status() -> None:
    assert is_retryable_stream_error(
        _statusless_openai_api_error(
            "stream embedded error",
            {"error": {"message": "internal failure", "code": 500}},
        )
    )


def test_retryable_stream_error_classifies_statusless_internal_error_type() -> None:
    assert is_retryable_stream_error(
        _statusless_openai_api_error(
            "stream embedded error",
            {"error": {"message": "internal failure", "type": "internal_server_error"}},
        )
    )


def test_retryable_stream_error_classifies_resource_exhausted_text() -> None:
    assert is_retryable_stream_error(
        _statusless_openai_api_error(
            "ResourceExhausted: limit reached while generating response",
            {"error": {"message": "ResourceExhausted: limit reached"}},
        )
    )


def test_retryable_stream_error_does_not_retry_bad_request_status() -> None:
    request = httpx.Request("POST", "https://provider.test/messages")
    assert not is_retryable_stream_error(
        openai.BadRequestError(
            "bad request",
            response=httpx.Response(400, request=request),
            body={"error": {"message": "bad request"}},
        )
    )


def test_recovery_controller_advances_early_retry_and_discards_holdback() -> None:
    controller = RecoveryController(provider_name="TEST", request_id="REQ")

    assert controller.push("hidden") == []
    decision = controller.advance_failure(
        httpx.ReadError("early cutoff"),
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
    )

    assert decision.action == RecoveryFailureAction.EARLY_RETRY
    assert decision.early_retry_attempt == 1
    assert controller.early_retries == 1
    assert not controller.committed
    assert not controller.has_buffered
    assert controller.flush() == []


def test_recovery_controller_retries_statusless_transient_api_error() -> None:
    controller = RecoveryController(provider_name="TEST", request_id="REQ")

    decision = controller.advance_failure(
        _statusless_openai_api_error(
            "ResourceExhausted: limit reached while generating response",
            {"error": {"message": "ResourceExhausted: limit reached"}},
        ),
        stream_opened=True,
        generated_output=False,
        complete_tool_salvageable=False,
    )

    assert decision.action == RecoveryFailureAction.EARLY_RETRY
    assert decision.retryable
    assert decision.early_retry_attempt == 1


def test_recovery_controller_respects_early_retry_limit() -> None:
    controller = RecoveryController(provider_name="TEST", request_id=None)

    for attempt in range(1, EARLY_TRANSPARENT_MAX_RETRIES + 1):
        decision = controller.advance_failure(
            httpx.ReadError("cutoff"),
            stream_opened=True,
            generated_output=False,
            complete_tool_salvageable=False,
        )
        assert decision.action == RecoveryFailureAction.EARLY_RETRY
        assert decision.early_retry_attempt == attempt

    decision = controller.advance_failure(
        httpx.ReadError("cutoff"),
        stream_opened=True,
        generated_output=False,
        complete_tool_salvageable=False,
    )

    assert decision.action == RecoveryFailureAction.FINAL_ERROR
    assert controller.early_retries == EARLY_TRANSPARENT_MAX_RETRIES


def test_recovery_controller_classifies_midstream_recovery_after_commit() -> None:
    controller = RecoveryController(provider_name="TEST", request_id=None)

    assert controller.push("event: content_block_delta\n\n") == []
    assert controller.flush() == ["event: content_block_delta\n\n"]
    decision = controller.advance_failure(
        httpx.ReadError("midstream cutoff"),
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
    )

    assert decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY
    assert decision.retryable
    assert decision.committed
    assert controller.flush_uncommitted(decision) == []


def test_recovery_controller_flushes_uncommitted_midstream_decision() -> None:
    controller = RecoveryController(provider_name="TEST", request_id=None)

    assert controller.push("event: content_block_delta\n\n") == []
    decision = controller.advance_failure(
        httpx.ReadError("midstream cutoff"),
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=True,
    )

    assert decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY
    assert not decision.committed
    assert decision.has_buffered
    assert not controller.committed
    assert controller.has_buffered

    assert controller.flush_uncommitted(decision) == ["event: content_block_delta\n\n"]
    assert not decision.committed
    assert decision.has_buffered
    assert controller.committed
    assert not controller.has_buffered


def test_recovery_controller_non_retryable_error_is_final() -> None:
    request = httpx.Request("POST", "https://example.test/messages")
    error = httpx.HTTPStatusError(
        "bad request",
        request=request,
        response=httpx.Response(400, request=request),
    )
    controller = RecoveryController(provider_name="TEST", request_id=None)

    decision = controller.advance_failure(
        error,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
    )

    assert decision.action == RecoveryFailureAction.FINAL_ERROR
    assert not decision.retryable
    assert controller.early_retries == 0


def test_holdback_buffers_until_delay_then_commits() -> None:
    now = [10.0]
    holdback = RecoveryHoldbackBuffer(holdback_seconds=0.75, now=lambda: now[0])

    assert holdback.push("event: content_block_start\n\n") == []
    now[0] += 0.74
    assert holdback.push("event: content_block_delta\n\n") == []
    assert not holdback.committed

    now[0] += 0.01
    flushed = holdback.push("event: content_block_stop\n\n")
    assert flushed == [
        "event: content_block_start\n\n",
        "event: content_block_delta\n\n",
        "event: content_block_stop\n\n",
    ]
    assert holdback.committed
    assert holdback.push("event: message_stop\n\n") == ["event: message_stop\n\n"]


def test_holdback_flushes_at_internal_buffer_cap() -> None:
    holdback = RecoveryHoldbackBuffer(max_bytes=5, now=lambda: 1.0)

    assert holdback.push("ab") == []
    assert holdback.push("cde") == ["ab", "cde"]
    assert holdback.committed


def test_holdback_discard_drops_uncommitted_events() -> None:
    holdback = RecoveryHoldbackBuffer(now=lambda: 1.0)

    assert holdback.push("hidden") == []
    holdback.discard()

    assert holdback.flush() == []
