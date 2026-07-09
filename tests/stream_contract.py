"""Shared assertions for canonical provider streaming error envelopes."""

from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    text_content,
)


def assert_canonical_stream_error_envelope(
    events: list[str], *, user_message_substr: str
) -> None:
    """Native transports emit message_start → text error → message_stop."""
    blob = "".join(events)
    assert "event: error\ndata:" not in blob
    parsed = parse_sse_text(blob)
    assert_anthropic_stream_contract(parsed, allow_error=False)
    assert user_message_substr in text_content(parsed)
