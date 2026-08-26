from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    estimate_responses_input_tokens,
)


def test_responses_token_estimate_counts_only_input_bearing_fields() -> None:
    base = OpenAIResponsesRequest(
        model="provider/model",
        input="hello",
        instructions="be concise",
        tools=[
            {
                "type": "function",
                "name": "lookup",
                "description": "look up one value",
                "parameters": {"type": "object"},
            }
        ],
        metadata={"large-unrelated-value": "x" * 20_000},
    )
    without_metadata = base.model_copy(update={"metadata": None})

    assert estimate_responses_input_tokens(base) == (
        estimate_responses_input_tokens(without_metadata)
    )


def test_responses_token_estimate_increases_with_input_content() -> None:
    short = OpenAIResponsesRequest(model="provider/model", input="hello")
    long = OpenAIResponsesRequest(
        model="provider/model",
        input="hello " * 1_000,
    )

    assert estimate_responses_input_tokens(short) > 0
    assert estimate_responses_input_tokens(long) > estimate_responses_input_tokens(
        short
    )
