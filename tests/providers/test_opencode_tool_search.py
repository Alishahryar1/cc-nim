from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.opencode.tool_search import normalize_tool_search


def test_nested_search_arguments_keep_constraints_and_literal_data() -> None:
    tool: JsonObject = {
        "type": "tool_search",
        "execution": "client",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/filter"},
                },
            },
            "required": ["query"],
            "additionalProperties": False,
            "$defs": {
                "filter": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": ["file", "directory"]},
                        "count": {
                            "anyOf": [
                                {"type": "integer", "minimum": 1},
                                {"type": "string", "enum": ["all"]},
                            ]
                        },
                    },
                    "required": ["kind"],
                    "additionalProperties": False,
                }
            },
            "examples": [{"properties": {"literal": "untouched"}}],
        },
    }
    original = deepcopy(tool)
    normalized = normalize_tool_search(tool)
    schema = normalized["parameters"]
    assert isinstance(schema, dict)
    validator = Draft202012Validator(schema)
    assert validator.is_valid({"query": "files", "filters": None})
    assert validator.is_valid(
        {"query": "files", "filters": [{"kind": "file", "count": None}]}
    )
    assert validator.is_valid(
        {"query": "files", "filters": [{"kind": "file", "count": 2}]}
    )
    assert not validator.is_valid(
        {"query": "files", "filters": [{"kind": "file", "count": 0}]}
    )
    assert not validator.is_valid(
        {"query": "files", "filters": [{"kind": "unknown", "count": None}]}
    )
    assert not validator.is_valid({"query": "files", "filters": [{"kind": "file"}]})
    assert schema["examples"] == [{"properties": {"literal": "untouched"}}]
    assert normalize_tool_search(normalized) == normalized
    assert tool == original


@pytest.mark.parametrize(
    "tool",
    [
        {"type": "tool_search"},
        {"type": "tool_search", "execution": "server"},
        {"type": "tool_search", "execution": "client"},
        {
            "type": "function",
            "name": "search",
            "strict": False,
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        },
    ],
)
def test_other_tool_definitions_are_preserved(tool: JsonObject) -> None:
    assert normalize_tool_search(tool) == tool
