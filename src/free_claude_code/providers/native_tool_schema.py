"""Schema helpers shared by provider-native textual tool-call dialects."""

import json
from collections.abc import Mapping
from typing import Any

import jsonschema


def openai_tool_schemas(body: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index declared OpenAI function schemas by tool name."""
    tools = body.get("tools")
    if not isinstance(tools, list):
        return {}

    schemas: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = function.get("parameters")
        schemas[name] = dict(parameters) if isinstance(parameters, Mapping) else {}
    return schemas


def schema_type(schema: Mapping[str, Any]) -> str | None:
    """Resolve the single useful JSON type from a simple or union schema."""
    declared = schema.get("type")
    if isinstance(declared, str):
        return declared
    if isinstance(declared, list):
        non_null = [
            item for item in declared if isinstance(item, str) and item != "null"
        ]
        if len(non_null) == 1:
            return non_null[0]

    for keyword in ("oneOf", "anyOf"):
        alternatives = schema.get(keyword)
        if not isinstance(alternatives, list):
            continue
        types = {
            nested_type
            for alternative in alternatives
            if isinstance(alternative, Mapping)
            if (nested_type := schema_type(alternative)) not in (None, "null")
        }
        if len(types) == 1:
            return types.pop()
    return None


def coerce_text_argument(value: str, schema: Mapping[str, Any]) -> Any:
    """Decode a textual native-tool argument according to its JSON schema.

    ``ValueError`` means the text cannot represent the declared type. Unknown or
    invalid schemas intentionally leave the value as text; final schema validation
    remains the authority.
    """
    enum_values = schema.get("enum")
    if isinstance(enum_values, list):
        for enum_value in enum_values:
            if str(enum_value) == value:
                return enum_value
    if "const" in schema and str(schema["const"]) == value:
        return schema["const"]

    value_type = schema_type(schema)
    stripped = value.strip()
    try:
        if value_type == "integer":
            return int(stripped)
        if value_type == "number":
            number = json.loads(stripped)
            if isinstance(number, int | float) and not isinstance(number, bool):
                return number
            raise ValueError
        if value_type == "boolean":
            if stripped.lower() == "true":
                return True
            if stripped.lower() == "false":
                return False
            raise ValueError
        if value_type == "null":
            if stripped.lower() == "null":
                return None
            raise ValueError
        if value_type == "array":
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
            raise ValueError
        if value_type == "object":
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError
    except json.JSONDecodeError as error:
        raise ValueError from error
    return value


def arguments_match_schema(arguments: dict[str, Any], schema: dict[str, Any]) -> bool:
    """Return whether arguments satisfy a valid schema.

    Upstream tool declarations are trusted request input. If one contains an
    invalid JSON Schema, native normalization must not invent stricter semantics
    than the ordinary structured-tool path, so validation is skipped.
    """
    try:
        validator_type = jsonschema.validators.validator_for(schema)
        validator_type.check_schema(schema)
        validator_type(schema).validate(arguments)
    except jsonschema.exceptions.SchemaError:
        return True
    except jsonschema.exceptions.ValidationError:
        return False
    return True
