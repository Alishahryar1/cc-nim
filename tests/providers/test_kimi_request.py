"""Tests for providers/kimi/request.py — Moonshot-flavored schema sanitization."""

from typing import Any, cast

from providers.kimi.request import (
    _inline_refs,
    _sanitize_tool_parameters,
    _sanitize_tools,
)


class TestSanitizeToolParameters:
    def test_inlines_local_ref_and_drops_defs(self):
        schema = {
            "type": "object",
            "properties": {
                "color": {
                    "$ref": "#/$defs/LabelColor",
                    "description": "Sibling description.",
                },
            },
            "$defs": {
                "LabelColor": {
                    "description": "Target description.",
                    "type": "object",
                    "properties": {
                        "backgroundColor": {"type": "string"},
                    },
                },
            },
        }

        out = _sanitize_tool_parameters(schema)

        color = out["properties"]["color"]
        assert "$ref" not in color
        assert color["type"] == "object"
        assert color["properties"]["backgroundColor"]["type"] == "string"
        assert "$defs" not in out

    def test_sibling_description_does_not_overwrite_target(self):
        schema = {
            "properties": {
                "color": {
                    "$ref": "#/$defs/LabelColor",
                    "description": "Sibling description.",
                },
            },
            "$defs": {
                "LabelColor": {
                    "description": "Target description.",
                    "type": "object",
                },
            },
        }

        out = _sanitize_tool_parameters(schema)

        # Target wins; only one description survives so no Moonshot conflict.
        assert out["properties"]["color"]["description"] == "Target description."

    def test_sibling_keys_kept_when_target_lacks_them(self):
        schema = {
            "properties": {
                "x": {
                    "$ref": "#/$defs/Plain",
                    "description": "Only on sibling.",
                },
            },
            "$defs": {"Plain": {"type": "string"}},
        }

        out = _sanitize_tool_parameters(schema)

        assert out["properties"]["x"]["type"] == "string"
        assert out["properties"]["x"]["description"] == "Only on sibling."

    def test_definitions_alias_also_dropped(self):
        schema = {
            "properties": {"x": {"$ref": "#/definitions/Foo"}},
            "definitions": {"Foo": {"type": "string"}},
        }

        out = _sanitize_tool_parameters(schema)

        assert "definitions" not in out
        assert out["properties"]["x"]["type"] == "string"

    def test_unresolvable_ref_drops_ref_keeps_siblings(self):
        schema = {
            "properties": {
                "x": {"$ref": "#/$defs/Missing", "type": "string"},
            },
        }

        out = _sanitize_tool_parameters(schema)

        assert "$ref" not in out["properties"]["x"]
        assert out["properties"]["x"]["type"] == "string"

    def test_circular_ref_does_not_recurse_forever(self):
        schema = {
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {"child": {"$ref": "#/$defs/Node"}},
                },
            },
            "properties": {"root": {"$ref": "#/$defs/Node"}},
        }

        out = _sanitize_tool_parameters(schema)

        # Outer expansion happens once; inner self-ref breaks cycle.
        root = out["properties"]["root"]
        assert root["type"] == "object"
        assert "$ref" not in root["properties"]["child"]

    def test_does_not_mutate_input(self):
        schema = {
            "properties": {"x": {"$ref": "#/$defs/Foo"}},
            "$defs": {"Foo": {"type": "string"}},
        }

        _sanitize_tool_parameters(schema)

        assert schema["properties"]["x"] == {"$ref": "#/$defs/Foo"}
        assert "$defs" in schema

    def test_non_dict_input_passes_through(self):
        assert _sanitize_tool_parameters(cast(Any, [])) == []
        assert _sanitize_tool_parameters(cast(Any, "x")) == "x"


class TestInlineRefs:
    def test_nested_lists_are_walked(self):
        root = {"$defs": {"S": {"type": "string"}}}
        node = {"oneOf": [{"$ref": "#/$defs/S"}, {"type": "integer"}]}

        out = _inline_refs(node, root)

        assert out == {"oneOf": [{"type": "string"}, {"type": "integer"}]}

    def test_passthrough_for_non_container(self):
        assert _inline_refs(42, {}) == 42
        assert _inline_refs(None, {}) is None


class TestSanitizeTools:
    def test_walks_tools_function_parameters(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "create_label",
                    "parameters": {
                        "properties": {
                            "color": {
                                "$ref": "#/$defs/LabelColor",
                                "description": "sibling",
                            },
                        },
                        "$defs": {
                            "LabelColor": {
                                "description": "target",
                                "type": "object",
                            },
                        },
                    },
                },
            },
        ]

        out = _sanitize_tools(tools)

        params = out[0]["function"]["parameters"]
        assert "$defs" not in params
        assert "$ref" not in params["properties"]["color"]

    def test_tool_without_function_passes_through(self):
        tools = [{"type": "other", "name": "x"}]
        assert _sanitize_tools(tools) == tools

    def test_function_without_parameters_unchanged(self):
        tools = [{"type": "function", "function": {"name": "noop"}}]
        assert _sanitize_tools(tools) == tools

    def test_empty_or_none_tools(self):
        assert _sanitize_tools([]) == []
        assert _sanitize_tools(cast(Any, None)) == []
