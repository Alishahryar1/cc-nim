"""Test fixtures for the provider-neutral inference stream contract."""

from collections.abc import AsyncIterator, Iterable

from free_claude_code.core.anthropic import AnthropicEventPresenter
from free_claude_code.core.inference import (
    FinishReason,
    InferenceEvent,
    InferenceStreamLedger,
    InferenceUsage,
    TokenMeasurement,
    UsageSource,
)


def reported_usage(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
    reasoning_output_tokens: int | None = None,
) -> InferenceUsage:
    """Build cumulative upstream-reported usage for a test stream."""

    return InferenceUsage(
        input_tokens=TokenMeasurement(input_tokens, UsageSource.REPORTED),
        cache_read_input_tokens=(
            TokenMeasurement(cache_read_input_tokens, UsageSource.REPORTED)
            if cache_read_input_tokens is not None
            else None
        ),
        cache_creation_input_tokens=(
            TokenMeasurement(cache_creation_input_tokens, UsageSource.REPORTED)
            if cache_creation_input_tokens is not None
            else None
        ),
        output_tokens=TokenMeasurement(output_tokens, UsageSource.REPORTED),
        reasoning_output_tokens=(
            TokenMeasurement(reasoning_output_tokens, UsageSource.REPORTED)
            if reasoning_output_tokens is not None
            else None
        ),
    )


def text_event_stream(
    text: str,
    *,
    model: str = "test-model",
    input_tokens: int = 3,
    output_tokens: int = 4,
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
) -> list[InferenceEvent]:
    """Build one complete canonical text response."""

    ledger = InferenceStreamLedger("response_test", model, input_tokens)
    events: list[InferenceEvent] = [ledger.start_response()]
    events.extend(ledger.ensure_text_block())
    if text:
        events.append(ledger.emit_text_delta(text))
    events.extend(ledger.close_all_blocks())
    events.extend(
        ledger.finish_events(
            FinishReason.END_TURN,
            reported_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
            ),
        )
    )
    return events


def present_anthropic(events: Iterable[InferenceEvent]) -> list[str]:
    """Render canonical test events at the Anthropic protocol boundary."""

    presenter = AnthropicEventPresenter()
    chunks: list[str] = []
    for event in events:
        chunks.extend(presenter.present(event))
    return chunks


async def collect_anthropic(
    events: AsyncIterator[InferenceEvent],
) -> list[str]:
    """Collect a canonical provider stream through the Anthropic presenter."""

    presenter = AnthropicEventPresenter()
    chunks: list[str] = []
    async for event in events:
        chunks.extend(presenter.present(event))
    return chunks
