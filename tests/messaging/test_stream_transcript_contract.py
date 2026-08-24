"""Messaging-specific assertions built on neutral Anthropic stream contracts."""

from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    has_tool_use,
    parse_sse_text,
)
from free_claude_code.core.inference import (
    FinishReason,
    InferenceEvent,
    InferenceStreamLedger,
)
from free_claude_code.messaging.event_parser import parse_cli_event
from free_claude_code.messaging.transcript import RenderCtx, TranscriptBuffer
from tests.inference_support import present_anthropic, reported_usage


def test_thinking_tool_text_and_transcript_order_contract() -> None:
    builder = InferenceStreamLedger("response_contract", "contract-model")
    canonical: list[InferenceEvent] = [builder.start_response()]
    canonical.extend(builder.ensure_reasoning_block())
    canonical.append(builder.emit_reasoning_delta("inspect first"))
    canonical.extend(builder.close_content_blocks())
    canonical.append(builder.start_tool_block(0, "toolu_1", "Read"))
    canonical.append(builder.emit_tool_delta(0, '{"file":"README.md"}'))
    canonical.append(builder.stop_tool_block(0))
    canonical.extend(builder.ensure_text_block())
    canonical.append(builder.emit_text_delta("done"))
    canonical.extend(builder.close_all_blocks())
    canonical.extend(
        builder.finish_events(
            FinishReason.END_TURN,
            reported_usage(input_tokens=0, output_tokens=20),
        )
    )

    events = parse_sse_text("".join(present_anthropic(canonical)))
    assert_anthropic_stream_contract(events)
    assert has_tool_use(events)

    transcript = TranscriptBuffer()
    for event in events:
        for parsed in parse_cli_event(event.data):
            transcript.apply(parsed)
    rendered = transcript.render(_render_ctx(), limit_chars=3900, status=None)
    assert (
        rendered.find("inspect first")
        < rendered.find("Tool call:")
        < rendered.find("done")
    )


def _render_ctx() -> RenderCtx:
    return RenderCtx(
        bold=lambda s: f"*{s}*",
        code_inline=lambda s: f"`{s}`",
        escape_code=lambda s: s,
        escape_text=lambda s: s,
        render_markdown=lambda s: s,
    )
