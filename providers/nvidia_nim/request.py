"""Request builder for NVIDIA NIM provider."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from loguru import logger

from config.nim import NimSettings
from core.anthropic import (
    ReasoningReplayMode,
    build_base_request_body,
    set_if_not_none,
)
from core.anthropic.conversion import OpenAIConversionError
from providers.exceptions import InvalidRequestError

_SCHEMA_VALUE_KEYS = frozenset(
    {
        "additionalProperties",
        "additionalItems",
        "unevaluatedProperties",
        "unevaluatedItems",
        "items",
        "contains",
        "propertyNames",
        "if",
        "then",
        "else",
        "not",
    }
)
_SCHEMA_LIST_KEYS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SCHEMA_MAP_KEYS = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
)
NIM_TOOL_ARGUMENT_ALIASES_KEY = "_fcc_nim_tool_argument_aliases"
_NIM_TOOL_PARAMETER_ALIAS_PREFIX = "_fcc_arg_"
_NIM_UNSAFE_TOOL_PARAMETER_NAMES = frozenset({"type"})


def _clone_strip_extra_body(
    body: dict[str, Any],
    strip: Callable[[dict[str, Any]], bool],
) -> dict[str, Any] | None:
    """Deep-clone ``body`` and remove fields via ``strip`` on ``extra_body`` only.

    Returns ``None`` when there is no ``extra_body`` dict or ``strip`` reports no change.
    """
    cloned_body = deepcopy(body)
    extra_body = cloned_body.get("extra_body")
    if not isinstance(extra_body, dict):
        return None
    if not strip(extra_body):
        return None
    if not extra_body:
        cloned_body.pop("extra_body", None)
    return cloned_body


def _strip_reasoning_budget_fields(extra_body: dict[str, Any]) -> bool:
    removed = extra_body.pop("reasoning_budget", None) is not None
    chat_template_kwargs = extra_body.get("chat_template_kwargs")
    if (
        isinstance(chat_template_kwargs, dict)
        and chat_template_kwargs.pop("reasoning_budget", None) is not None
    ):
        removed = True
    return removed


def _strip_chat_template_field(extra_body: dict[str, Any]) -> bool:
    return extra_body.pop("chat_template", None) is not None


def _strip_chat_template_kwargs_field(extra_body: dict[str, Any]) -> bool:
    return extra_body.pop("chat_template_kwargs", None) is not None


def _strip_message_reasoning_content(body: dict[str, Any]) -> bool:
    removed = False
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if (
            isinstance(message, dict)
            and message.pop("reasoning_content", None) is not None
        ):
            removed = True
    return removed


def _sanitize_nim_schema_node(value: Any) -> tuple[bool, Any]:
    """Remove boolean JSON Schema subschemas that hosted NIM rejects."""
    if isinstance(value, bool):
        return False, None
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in _SCHEMA_VALUE_KEYS:
                keep, sanitized_item = _sanitize_nim_schema_node(item)
                if keep:
                    sanitized[key] = sanitized_item
            elif key in _SCHEMA_LIST_KEYS and isinstance(item, list):
                sanitized_items: list[Any] = []
                for schema_item in item:
                    keep, sanitized_item = _sanitize_nim_schema_node(schema_item)
                    if keep:
                        sanitized_items.append(sanitized_item)
                if sanitized_items:
                    sanitized[key] = sanitized_items
            elif key in _SCHEMA_MAP_KEYS and isinstance(item, dict):
                sanitized_map: dict[str, Any] = {}
                for map_key, schema_item in item.items():
                    keep, sanitized_item = _sanitize_nim_schema_node(schema_item)
                    if keep:
                        sanitized_map[map_key] = sanitized_item
                sanitized[key] = sanitized_map
            else:
                sanitized[key] = item
        return True, sanitized
    if isinstance(value, list):
        sanitized_items = []
        for item in value:
            keep, sanitized_item = _sanitize_nim_schema_node(item)
            if keep:
                sanitized_items.append(sanitized_item)
        return True, sanitized_items
    return True, value


def _needs_nim_tool_parameter_alias(name: str) -> bool:
    return name in _NIM_UNSAFE_TOOL_PARAMETER_NAMES


def _make_nim_tool_parameter_alias(name: str, reserved: set[str]) -> str:
    safe_tail = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in name
    ).strip("_")
    if not safe_tail:
        safe_tail = "arg"
    candidate = f"{_NIM_TOOL_PARAMETER_ALIAS_PREFIX}{safe_tail}"
    alias = candidate
    suffix = 2
    while alias in reserved:
        alias = f"{candidate}_{suffix}"
        suffix += 1
    reserved.add(alias)
    return alias


def _collect_nim_tool_property_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            for property_name, property_schema in properties.items():
                if isinstance(property_name, str):
                    names.add(property_name)
                names.update(_collect_nim_tool_property_names(property_schema))
        for key, item in value.items():
            if key != "properties":
                names.update(_collect_nim_tool_property_names(item))
    elif isinstance(value, list):
        for item in value:
            names.update(_collect_nim_tool_property_names(item))
    return names


def _alias_nim_schema_property_names(
    value: Any,
    *,
    reserved: set[str],
    alias_to_original: dict[str, str],
    original_to_alias: dict[str, str],
) -> Any:
    if isinstance(value, list):
        return [
            _alias_nim_schema_property_names(
                item,
                reserved=reserved,
                alias_to_original=alias_to_original,
                original_to_alias=original_to_alias,
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return value

    local_aliases: dict[str, str] = {}
    aliased_value: dict[str, Any] = {}
    properties = value.get("properties")
    if isinstance(properties, dict):
        aliased_properties: dict[str, Any] = {}
        for property_name, property_schema in properties.items():
            aliased_schema = _alias_nim_schema_property_names(
                property_schema,
                reserved=reserved,
                alias_to_original=alias_to_original,
                original_to_alias=original_to_alias,
            )
            if isinstance(property_name, str) and _needs_nim_tool_parameter_alias(
                property_name
            ):
                alias = original_to_alias.get(property_name)
                if alias is None:
                    alias = _make_nim_tool_parameter_alias(property_name, reserved)
                    alias_to_original[alias] = property_name
                    original_to_alias[property_name] = alias
                local_aliases[property_name] = alias
                aliased_properties[alias] = aliased_schema
            else:
                aliased_properties[property_name] = aliased_schema
        aliased_value["properties"] = aliased_properties

    for key, item in value.items():
        if key == "properties":
            continue
        if key == "required" and isinstance(item, list):
            aliased_value[key] = [
                local_aliases.get(required_item, required_item)
                if isinstance(required_item, str)
                else required_item
                for required_item in item
            ]
            continue
        aliased_value[key] = _alias_nim_schema_property_names(
            item,
            reserved=reserved,
            alias_to_original=alias_to_original,
            original_to_alias=original_to_alias,
        )
    return aliased_value


def _alias_nim_tool_parameters(
    parameters: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    alias_to_original: dict[str, str] = {}
    original_to_alias: dict[str, str] = {}
    reserved = _collect_nim_tool_property_names(parameters)
    aliased_parameters = _alias_nim_schema_property_names(
        parameters,
        reserved=reserved,
        alias_to_original=alias_to_original,
        original_to_alias=original_to_alias,
    )
    if not alias_to_original:
        return parameters, {}
    return aliased_parameters, alias_to_original


def _sanitize_nim_tool_schemas(body: dict[str, Any]) -> None:
    """Sanitize only tool parameter schemas, preserving tool calls/history."""
    tools = body.get("tools")
    if not isinstance(tools, list):
        return

    tool_argument_aliases: dict[str, dict[str, str]] = {}
    sanitized_tools: list[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            sanitized_tools.append(tool)
            continue
        sanitized_tool = dict(tool)
        function = tool.get("function")
        if isinstance(function, dict):
            sanitized_function = dict(function)
            parameters = function.get("parameters")
            if isinstance(parameters, dict):
                _, sanitized_parameters = _sanitize_nim_schema_node(parameters)
                sanitized_parameters, argument_aliases = _alias_nim_tool_parameters(
                    sanitized_parameters
                )
                sanitized_function["parameters"] = sanitized_parameters
                tool_name = function.get("name")
                if argument_aliases and isinstance(tool_name, str) and tool_name:
                    tool_argument_aliases[tool_name] = argument_aliases
            sanitized_tool["function"] = sanitized_function
        sanitized_tools.append(sanitized_tool)

    body["tools"] = sanitized_tools
    if tool_argument_aliases:
        body[NIM_TOOL_ARGUMENT_ALIASES_KEY] = tool_argument_aliases
    else:
        body.pop(NIM_TOOL_ARGUMENT_ALIASES_KEY, None)


def nim_tool_argument_aliases_from_body(
    body: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """Return validated private NIM tool argument aliases from a built body."""
    raw_aliases = body.get(NIM_TOOL_ARGUMENT_ALIASES_KEY)
    if not isinstance(raw_aliases, dict):
        return {}

    aliases: dict[str, dict[str, str]] = {}
    for tool_name, tool_aliases in raw_aliases.items():
        if not isinstance(tool_name, str) or not isinstance(tool_aliases, dict):
            continue
        sanitized_aliases = {
            alias: original
            for alias, original in tool_aliases.items()
            if isinstance(alias, str) and isinstance(original, str)
        }
        if sanitized_aliases:
            aliases[tool_name] = sanitized_aliases
    return aliases


def body_without_nim_tool_argument_aliases(body: dict[str, Any]) -> dict[str, Any]:
    """Return a request body with private alias metadata stripped before upstream I/O."""
    if NIM_TOOL_ARGUMENT_ALIASES_KEY not in body:
        return body
    upstream_body = dict(body)
    upstream_body.pop(NIM_TOOL_ARGUMENT_ALIASES_KEY, None)
    return upstream_body


def _set_extra(
    extra_body: dict[str, Any], key: str, value: Any, ignore_value: Any = None
) -> None:
    if key in extra_body:
        return
    if value is None:
        return
    if ignore_value is not None and value == ignore_value:
        return
    extra_body[key] = value


def clone_body_without_reasoning_budget(body: dict[str, Any]) -> dict[str, Any] | None:
    """Clone a request body and strip only reasoning_budget fields."""
    return _clone_strip_extra_body(body, _strip_reasoning_budget_fields)


def clone_body_without_chat_template(body: dict[str, Any]) -> dict[str, Any] | None:
    """Clone a request body and strip only chat_template."""
    return _clone_strip_extra_body(body, _strip_chat_template_field)


def clone_body_without_chat_template_kwargs(
    body: dict[str, Any],
) -> dict[str, Any] | None:
    """Clone a request body and strip only chat_template_kwargs."""
    return _clone_strip_extra_body(body, _strip_chat_template_kwargs_field)


def clone_body_without_reasoning_content(body: dict[str, Any]) -> dict[str, Any] | None:
    """Clone a request body and strip assistant message ``reasoning_content`` fields."""
    cloned_body = deepcopy(body)
    if not _strip_message_reasoning_content(cloned_body):
        return None
    return cloned_body


def build_request_body(
    request_data: Any, nim: NimSettings, *, thinking_enabled: bool
) -> dict:
    """Build OpenAI-format request body from Anthropic request."""
    logger.debug(
        "NIM_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    try:
        body = build_base_request_body(
            request_data,
            reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
            if thinking_enabled
            else ReasoningReplayMode.DISABLED,
        )
    except OpenAIConversionError as exc:
        raise InvalidRequestError(str(exc)) from exc

    _sanitize_nim_tool_schemas(body)

    # NIM-specific max_tokens: cap against nim.max_tokens
    max_tokens = body.get("max_tokens") or getattr(request_data, "max_tokens", None)
    if max_tokens is None:
        max_tokens = nim.max_tokens
    elif nim.max_tokens:
        max_tokens = min(max_tokens, nim.max_tokens)
    set_if_not_none(body, "max_tokens", max_tokens)

    # NIM-specific temperature/top_p: fall back to NIM defaults if request didn't set
    if body.get("temperature") is None and nim.temperature is not None:
        body["temperature"] = nim.temperature
    if body.get("top_p") is None and nim.top_p is not None:
        body["top_p"] = nim.top_p

    # NIM-specific stop sequences fallback
    if "stop" not in body and nim.stop:
        body["stop"] = nim.stop

    if nim.presence_penalty != 0.0:
        body["presence_penalty"] = nim.presence_penalty
    if nim.frequency_penalty != 0.0:
        body["frequency_penalty"] = nim.frequency_penalty
    if nim.seed is not None:
        body["seed"] = nim.seed

    body["parallel_tool_calls"] = nim.parallel_tool_calls

    # Handle non-standard parameters via extra_body
    extra_body: dict[str, Any] = {}
    request_extra = getattr(request_data, "extra_body", None)
    if request_extra:
        extra_body.update(request_extra)

    if thinking_enabled:
        chat_template_kwargs = extra_body.setdefault(
            "chat_template_kwargs", {"thinking": True, "enable_thinking": True}
        )
        if isinstance(chat_template_kwargs, dict):
            chat_template_kwargs.setdefault("reasoning_budget", max_tokens)

    req_top_k = getattr(request_data, "top_k", None)
    top_k = req_top_k if req_top_k is not None else nim.top_k
    _set_extra(extra_body, "top_k", top_k, ignore_value=-1)
    _set_extra(extra_body, "min_p", nim.min_p, ignore_value=0.0)
    _set_extra(
        extra_body, "repetition_penalty", nim.repetition_penalty, ignore_value=1.0
    )
    _set_extra(extra_body, "min_tokens", nim.min_tokens, ignore_value=0)
    _set_extra(extra_body, "chat_template", nim.chat_template)
    _set_extra(extra_body, "request_id", nim.request_id)
    _set_extra(extra_body, "ignore_eos", nim.ignore_eos)

    if extra_body:
        body["extra_body"] = extra_body

    logger.debug(
        "NIM_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
