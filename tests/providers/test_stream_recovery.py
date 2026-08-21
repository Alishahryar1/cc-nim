"""Provider stream commit-boundary and recovery policy."""

import httpx
import openai

from free_claude_code.providers.stream_recovery import (
    RecoveryController,
    RecoveryFailureAction,
    RecoveryHoldbackBuffer,
    TruncatedProviderStreamError,
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


def test_retryable_stream_error_classifies_protocol_transport_and_status() -> None:
    assert is_retryable_stream_error(
        TruncatedProviderStreamError("missing terminal marker")
    )
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


def test_early_retry_discards_uncommitted_holdback() -> None:
    controller = RecoveryController()

    assert controller.push("hidden") == []
    decision = controller.advance_failure(
        httpx.ReadError("early cutoff"),
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=2,
    )

    assert decision.action == RecoveryFailureAction.EARLY_RETRY
    assert decision.retryable
    assert decision.has_buffered
    assert not controller.committed
    assert not controller.has_buffered
    assert controller.flush() == []


def test_early_retry_requires_remaining_execution_budget() -> None:
    controller = RecoveryController()
    assert controller.push("hidden") == []

    decision = controller.advance_failure(
        httpx.ReadError("early cutoff"),
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=0,
    )

    assert decision.action == RecoveryFailureAction.FINAL_ERROR
    assert decision.retryable
    assert controller.has_buffered


def test_last_attempt_is_reserved_for_partial_output_recovery() -> None:
    controller = RecoveryController()
    assert controller.push("partial") == []

    decision = controller.advance_failure(
        httpx.ReadError("early cutoff"),
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=1,
    )

    assert decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY
    assert decision.has_buffered
    assert controller.has_buffered


def test_create_failure_is_owned_by_admission_not_stream_recovery() -> None:
    decision = RecoveryController().advance_failure(
        httpx.ConnectError("connect failed"),
        stream_opened=False,
        generated_output=False,
        complete_tool_salvageable=False,
        attempts_remaining=1,
    )

    assert decision.action == RecoveryFailureAction.FINAL_ERROR
    assert decision.retryable


def test_statusless_transient_api_error_allows_early_retry() -> None:
    decision = RecoveryController().advance_failure(
        _statusless_openai_api_error(
            "ResourceExhausted: limit reached while generating response",
            {"error": {"message": "ResourceExhausted: limit reached"}},
        ),
        stream_opened=True,
        generated_output=False,
        complete_tool_salvageable=False,
        attempts_remaining=1,
    )

    assert decision.action == RecoveryFailureAction.EARLY_RETRY
    assert decision.retryable


def test_committed_output_allows_midstream_recovery() -> None:
    controller = RecoveryController()

    assert controller.push("event: content_block_delta\n\n") == []
    assert controller.flush() == ["event: content_block_delta\n\n"]
    decision = controller.advance_failure(
        httpx.ReadError("midstream cutoff"),
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=1,
    )

    assert decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY
    assert decision.retryable
    assert decision.committed
    assert controller.flush_uncommitted(decision) == []


def test_uncommitted_complete_tool_can_be_salvaged() -> None:
    controller = RecoveryController()

    assert controller.push("event: content_block_delta\n\n") == []
    decision = controller.advance_failure(
        httpx.ReadError("midstream cutoff"),
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=True,
        attempts_remaining=0,
    )

    assert decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY
    assert not decision.committed
    assert decision.has_buffered
    assert controller.flush_uncommitted(decision) == ["event: content_block_delta\n\n"]
    assert controller.committed
    assert not controller.has_buffered


def test_non_retryable_error_is_final() -> None:
    request = httpx.Request("POST", "https://example.test/messages")
    error = httpx.HTTPStatusError(
        "bad request",
        request=request,
        response=httpx.Response(400, request=request),
    )

    decision = RecoveryController().advance_failure(
        error,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
        attempts_remaining=2,
    )

    assert decision.action == RecoveryFailureAction.FINAL_ERROR
    assert not decision.retryable


def test_holdback_buffers_until_delay_then_commits() -> None:
    now = [10.0]
    holdback = RecoveryHoldbackBuffer(holdback_seconds=0.75, now=lambda: now[0])

    assert holdback.push("event: content_block_start\n\n") == []
    now[0] += 0.74
    assert holdback.push("event: content_block_delta\n\n") == []
    assert not holdback.committed

    now[0] += 0.01
    assert holdback.push("event: content_block_stop\n\n") == [
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
