"""Neutral Anthropic continuation and tool-repair helpers."""

from free_claude_code.core.anthropic.streaming import (
    ToolSchema,
    accept_tool_json_repair,
    continuation_suffix,
)


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
