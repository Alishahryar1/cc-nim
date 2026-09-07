"""OpenCode requires explicit values for client-executed tool-search arguments."""

from collections.abc import Mapping

from free_claude_code.core.json_types import JsonObject, JsonValue


def normalize_tool_search(tool: JsonObject) -> JsonObject:
    """Represent omitted search arguments as null without changing function tools."""
    if tool.get("type") != "tool_search" or tool.get("execution") != "client":
        return tool
    parameters = tool.get("parameters")
    if not isinstance(parameters, Mapping):
        return tool
    return {**tool, "parameters": _explicit_arguments(parameters)}


def _explicit_arguments(schema: JsonValue) -> JsonValue:
    if not isinstance(schema, Mapping):
        return schema
    result = dict(schema)
    for keyword in ("properties", "$defs", "definitions"):
        children = schema.get(keyword)
        if isinstance(children, Mapping):
            result[keyword] = {
                name: _explicit_arguments(child) for name, child in children.items()
            }
    for keyword in ("items", "additionalProperties"):
        if keyword in schema:
            result[keyword] = _explicit_arguments(schema[keyword])
    for keyword in ("anyOf", "oneOf", "allOf", "prefixItems"):
        children = schema.get(keyword)
        if isinstance(children, list):
            result[keyword] = [_explicit_arguments(child) for child in children]
    properties = result.get("properties")
    if isinstance(properties, Mapping):
        required = schema.get("required")
        names = required if isinstance(required, list) else []
        result["properties"] = {
            name: child if name in names else {"anyOf": [child, {"type": "null"}]}
            for name, child in properties.items()
        }
        result["required"] = [
            *names,
            *(name for name in properties if name not in names),
        ]
    return result
