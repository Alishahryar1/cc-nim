"""Tests that API and SSE logging avoid raw sensitive payloads by default."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from api import provider_execution, request_errors
from api.handlers import MessagesHandler, TokenCountHandler
from api.models.anthropic import Message, MessagesRequest
from config.settings import Settings
from core.anthropic import AnthropicStreamLedger


@pytest.mark.asyncio
async def test_create_message_skips_full_payload_debug_log_by_default():
    settings = Settings()
    assert settings.log_raw_api_payloads is False
    mock_provider = MagicMock()

    async def fake_stream(*_a, **_kw):
        yield "event: ping\ndata: {}\n\n"

    mock_provider.stream_response = fake_stream
    service = MessagesHandler(settings, provider_getter=lambda _: mock_provider)

    request = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=10,
        messages=[Message(role="user", content="secret-user-text")],
    )

    with patch.object(provider_execution.logger, "debug") as mock_debug:
        await service.create(request)

    full_payload_calls = [
        c
        for c in mock_debug.call_args_list
        if c.args and str(c.args[0]) == "FULL_PAYLOAD [{}]: {}"
    ]
    assert not full_payload_calls


@pytest.mark.asyncio
async def test_create_message_logs_full_payload_when_opt_in():
    settings = Settings()
    settings.log_raw_api_payloads = True
    mock_provider = MagicMock()

    async def fake_stream(*_a, **_kw):
        yield "event: ping\ndata: {}\n\n"

    mock_provider.stream_response = fake_stream
    service = MessagesHandler(settings, provider_getter=lambda _: mock_provider)
    request = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=10,
        messages=[Message(role="user", content="visible")],
    )

    with patch.object(provider_execution.logger, "debug") as mock_debug:
        await service.create(request)

    keys = [c.args[0] for c in mock_debug.call_args_list if c.args]
    assert any(k == "FULL_PAYLOAD [{}]: {}" for k in keys)


def test_stream_ledger_default_debug_has_no_serialized_json_content():
    with patch("core.anthropic.streaming.emitter.logger.debug") as mock_debug:
        ledger = AnthropicStreamLedger("msg_x", "m", 1, log_raw_events=False)
        ledger.message_start()

    assert mock_debug.call_count == 0


def test_stream_ledger_raw_logging_includes_event_body_when_enabled():
    with patch("core.anthropic.streaming.emitter.logger.debug") as mock_debug:
        ledger = AnthropicStreamLedger("msg_x", "m", 1, log_raw_events=True)
        ledger.message_start()

    assert mock_debug.call_count == 1
    message = str(mock_debug.call_args)
    assert "message_start" in message
    assert "role" in message


def _flatten_log_calls(mock_log) -> str:
    parts: list[str] = []
    for call in mock_log.call_args_list:
        parts.extend(str(arg) for arg in call.args)
        parts.append(repr(call.kwargs))
    return " ".join(parts)


@pytest.mark.asyncio
async def test_create_message_unexpected_error_default_logs_exclude_exception_text():
    settings = Settings()
    assert settings.log_api_error_tracebacks is False
    secret = "upstream-secret-token-abc"

    mock_provider = MagicMock()

    def stream_boom(*_a, **_kw):
        raise RuntimeError(secret)

    mock_provider.stream_response = stream_boom
    service = MessagesHandler(settings, provider_getter=lambda _: mock_provider)
    request = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=10,
        messages=[Message(role="user", content="hi")],
    )

    with (
        patch.object(request_errors.logger, "error") as log_err,
        pytest.raises(HTTPException),
    ):
        await service.create(request)

    blob = _flatten_log_calls(log_err)
    assert secret not in blob
    assert "RuntimeError" in blob


@pytest.mark.asyncio
async def test_create_message_unexpected_error_always_returns_500():
    """Non-provider failures must not leak arbitrary status_code attributes."""

    class WeirdError(Exception):
        status_code = 418

    settings = Settings()
    mock_provider = MagicMock()

    def stream_boom(*_a, **_kw):
        raise WeirdError("no")

    mock_provider.stream_response = stream_boom
    service = MessagesHandler(settings, provider_getter=lambda _: mock_provider)
    request = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=10,
        messages=[Message(role="user", content="hi")],
    )

    with pytest.raises(HTTPException) as excinfo:
        await service.create(request)

    assert excinfo.value.status_code == 500


def test_parse_cli_event_error_logs_metadata_by_default():
    """CLI parser must not log raw error text unless LOG_RAW_CLI_DIAGNOSTICS is on."""
    from messaging.event_parser import parse_cli_event

    secret = "user-secret-parser-leak-xyz"
    with patch("messaging.event_parser.logger.info") as log_info:
        parse_cli_event(
            {"type": "error", "error": {"message": secret}}, log_raw_cli=False
        )
    flat = " ".join(str(c) for c in log_info.call_args_list)
    assert secret not in flat
    assert "message_chars" in flat


def test_parse_cli_event_error_logs_text_when_log_raw_cli_enabled():
    from messaging.event_parser import parse_cli_event

    secret = "visible-cli-parser-msg"
    with patch("messaging.event_parser.logger.info") as log_info:
        parse_cli_event(
            {"type": "error", "error": {"message": secret}}, log_raw_cli=True
        )
    flat = " ".join(str(c) for c in log_info.call_args_list)
    assert secret in flat


def test_count_tokens_unexpected_error_default_logs_exclude_exception_text():
    settings = Settings()
    assert settings.log_api_error_tracebacks is False
    secret = "count-tokens-leak-xyz"

    def boom(*_a, **_kw):
        raise ValueError(secret)

    service = TokenCountHandler(
        settings,
        token_counter=boom,
    )
    from api.models.anthropic import TokenCountRequest

    req = TokenCountRequest(
        model="claude-3-haiku-20240307",
        messages=[Message(role="user", content="x")],
    )

    with (
        patch.object(request_errors.logger, "error") as log_err,
        pytest.raises(HTTPException),
    ):
        service.count(req)

    blob = _flatten_log_calls(log_err)
    assert secret not in blob
    assert "ValueError" in blob
