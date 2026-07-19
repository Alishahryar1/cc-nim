"""Map Anthropic stop reasons to OpenAI Chat Completions finish reasons."""


def finish_reason_from_stop_reason(
    stop_reason: str | None, *, has_tool_calls: bool
) -> str:
    """Translate an Anthropic ``stop_reason`` into an OpenAI ``finish_reason``."""
    if has_tool_calls or stop_reason == "tool_use":
        return "tool_calls"
    if stop_reason == "max_tokens":
        return "length"
    # end_turn, stop_sequence, and unknown/None all map to the normal stop.
    return "stop"
