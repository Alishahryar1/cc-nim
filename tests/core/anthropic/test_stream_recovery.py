"""Unit tests for resilient stream recovery helpers."""

import httpx

from core.anthropic.streaming import (
    EARLY_TRANSPARENT_MAX_RETRIES,
    EARLY_TRANSPARENT_TOTAL_ATTEMPTS,
    MIDSTREAM_RECOVERY_ATTEMPTS,
    RecoveryController,
    RecoveryFailureAction,
    RecoveryHoldbackBuffer,
    ToolSchema,
    accept_tool_json_repair,
    continuation_suffix,
    is_retryable_stream_error,
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


def test_stream_recovery_session_advances_early_retry_and_discards_holdback() -> None:
    session = RecoveryController(provider_name="TEST", request_id="REQ")

    assert session.push("hidden") == []
    decision = session.advance_failure(
        httpx.ReadError("early cutoff"),
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
    )

    assert decision.action == RecoveryFailureAction.EARLY_RETRY
    assert decision.early_retry_attempt == 1
    assert session.early_retries == 1
    assert not session.committed
    assert not session.has_buffered
    assert session.flush() == []


def test_stream_recovery_session_respects_early_retry_limit() -> None:
    session = RecoveryController(provider_name="TEST", request_id=None)

    for attempt in range(1, EARLY_TRANSPARENT_MAX_RETRIES + 1):
        decision = session.advance_failure(
            httpx.ReadError("cutoff"),
            stream_opened=True,
            generated_output=False,
            complete_tool_salvageable=False,
        )
        assert decision.action == RecoveryFailureAction.EARLY_RETRY
        assert decision.early_retry_attempt == attempt

    decision = session.advance_failure(
        httpx.ReadError("cutoff"),
        stream_opened=True,
        generated_output=False,
        complete_tool_salvageable=False,
    )

    assert decision.action == RecoveryFailureAction.FINAL_ERROR
    assert session.early_retries == EARLY_TRANSPARENT_MAX_RETRIES


def test_stream_recovery_session_classifies_midstream_recovery_after_commit() -> None:
    session = RecoveryController(provider_name="TEST", request_id=None)

    assert session.push("event: content_block_delta\n\n") == []
    assert session.flush() == ["event: content_block_delta\n\n"]
    decision = session.advance_failure(
        httpx.ReadError("midstream cutoff"),
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
    )

    assert decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY
    assert decision.retryable
    assert decision.committed
    assert session.flush_uncommitted(decision) == []


def test_stream_recovery_session_flushes_uncommitted_midstream_decision() -> None:
    session = RecoveryController(provider_name="TEST", request_id=None)

    assert session.push("event: content_block_delta\n\n") == []
    decision = session.advance_failure(
        httpx.ReadError("midstream cutoff"),
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=True,
    )

    assert decision.action == RecoveryFailureAction.MIDSTREAM_RECOVERY
    assert not decision.committed
    assert decision.has_buffered
    assert not session.committed
    assert session.has_buffered

    assert session.flush_uncommitted(decision) == ["event: content_block_delta\n\n"]
    assert not decision.committed
    assert decision.has_buffered
    assert session.committed
    assert not session.has_buffered


def test_stream_recovery_session_non_retryable_error_is_final() -> None:
    request = httpx.Request("POST", "https://example.test/messages")
    error = httpx.HTTPStatusError(
        "bad request",
        request=request,
        response=httpx.Response(400, request=request),
    )
    session = RecoveryController(provider_name="TEST", request_id=None)

    decision = session.advance_failure(
        error,
        stream_opened=True,
        generated_output=True,
        complete_tool_salvageable=False,
    )

    assert decision.action == RecoveryFailureAction.FINAL_ERROR
    assert not decision.retryable
    assert session.early_retries == 0


def test_continuation_suffix_trims_overlap() -> None:
    assert continuation_suffix("hello wor", "world") == "ld"
    assert continuation_suffix("alpha", "alpha beta") == " beta"
    assert continuation_suffix("", "fresh") == "fresh"


def test_tool_json_repair_requires_append_only_schema_valid_json() -> None:
    schemas = {
        "Echo": ToolSchema(
            name="Echo",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
                "additionalProperties": False,
            },
        )
    }

    accepted = accept_tool_json_repair(
        '{"message":',
        '"ok"}',
        tool_name="Echo",
        schemas=schemas,
    )
    assert accepted is not None
    assert accepted.suffix == '"ok"}'
    assert accepted.parsed_input == {"message": "ok"}

    assert (
        accept_tool_json_repair(
            '{"message":',
            "1}",
            tool_name="Echo",
            schemas=schemas,
        )
        is None
    )


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
