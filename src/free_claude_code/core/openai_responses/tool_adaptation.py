"""Provider-selected adaptations of native Responses tool representations."""

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

from free_claude_code.core.json_types import JsonObject, JsonValue

from .errors import ResponsesConversionError
from .models import OpenAIResponsesRequest
from .tool_search import normalize_tool_search
from .tools import (
    ResponsesToolIdentity,
    custom_tool_description,
    custom_tool_input_schema,
    custom_tool_input_text,
    custom_tool_input_text_from_arguments,
    flatten_responses_tool_name,
    optional_str,
    required_str,
)


@dataclass(frozen=True, slots=True)
class ResponsesToolPolicy:
    custom_tools_as_functions: bool = False
    explicit_search_parameters: bool = False
    text_only_web_search: bool = False


@dataclass(frozen=True, slots=True)
class _DefinitionEdit:
    original: JsonObject
    adapted: JsonObject
    namespace: str | None
    scope: str | None


class ResponsesToolAdapter:
    """Prepare one request and retain only the information needed to undo edits."""

    def __init__(
        self, request: OpenAIResponsesRequest, policy: ResponsesToolPolicy
    ) -> None:
        self.request = request.model_copy(deep=True)
        self._policy = policy
        self._identities: dict[tuple[str | None, str], ResponsesToolIdentity] = {}
        self._edits: list[_DefinitionEdit] = []
        if policy == ResponsesToolPolicy():
            return
        if self.request.tools:
            self.request.tools = cast(list[JsonObject], self._tools(self.request.tools))
        if isinstance(self.request.input, list):
            self.request.input = [self._input(item) for item in self.request.input]
        if policy.custom_tools_as_functions:
            self.request.tool_choice = self._choice(self.request.tool_choice)

    def _register(self, identity: ResponsesToolIdentity, wire_name: str) -> None:
        key = (identity.namespace, wire_name)
        existing = self._identities.get(key)
        if existing is not None and existing != identity:
            raise ResponsesConversionError("Tool names collide after conversion.")
        self._identities[key] = identity

    def _tools(
        self, tools: JsonValue, namespace: str | None = None, scope: str | None = None
    ) -> JsonValue:
        if not isinstance(tools, list):
            return tools
        return [self._tool(tool, namespace, scope) for tool in tools]

    def _tool(
        self, value: JsonValue, namespace: str | None, scope: str | None
    ) -> JsonValue:
        if not isinstance(value, dict):
            return value
        tool = value
        kind = tool.get("type")
        if kind == "namespace":
            return {
                **tool,
                "tools": self._tools(tool.get("tools"), _name(tool), scope),
            }
        if self._policy.custom_tools_as_functions and kind in {"custom", "function"}:
            nested = tool.get(str(kind))
            source = nested if isinstance(nested, dict) else tool
            name = _name(source)
            identity = ResponsesToolIdentity(
                kind="custom" if kind == "custom" else "function",
                name=name,
                namespace=namespace or _namespace(source),
            )
            wire_name = (
                flatten_responses_tool_name(name, namespace=identity.namespace)
                if kind == "custom"
                else name
            )
            self._register(identity, wire_name)
            if kind == "custom":
                tool = {
                    **{key: child for key, child in tool.items() if key != "custom"},
                    **source,
                    "type": "function",
                    "name": wire_name,
                    "parameters": custom_tool_input_schema(),
                    "strict": False,
                }
                tool.pop("format", None)
                if description := custom_tool_description(source):
                    tool["description"] = description
        if self._policy.explicit_search_parameters:
            tool = normalize_tool_search(tool)
        if self._policy.text_only_web_search and kind in {
            "web_search",
            "web_search_preview",
        }:
            content_types = tool.get("search_content_types")
            if content_types is not None:
                if not isinstance(content_types, list) or "text" not in content_types:
                    raise ResponsesConversionError(
                        "The selected provider supports text web search only."
                    )
                tool = {
                    key: child
                    for key, child in tool.items()
                    if key != "search_content_types"
                }
        if tool != value:
            self._edits.append(_DefinitionEdit(value, tool, namespace, scope))
        return tool

    def _input(self, item: JsonValue) -> JsonValue:
        if not isinstance(item, dict):
            return item
        kind = item.get("type")
        if kind == "tool_search_output":
            return {**item, "tools": self._tools(item.get("tools"), scope=_scope(item))}
        if not self._policy.custom_tools_as_functions:
            return item
        if kind in {"custom_tool_call", "function_call"}:
            name = _name(item)
            identity = ResponsesToolIdentity(
                kind="custom" if kind == "custom_tool_call" else "function",
                name=name,
                namespace=_namespace(item),
            )
            wire_name = (
                flatten_responses_tool_name(name, namespace=identity.namespace)
                if kind == "custom_tool_call"
                else name
            )
            self._register(identity, wire_name)
            if kind == "custom_tool_call":
                return {
                    **{key: value for key, value in item.items() if key != "input"},
                    "type": "function_call",
                    "name": wire_name,
                    "arguments": json.dumps(
                        {"input": custom_tool_input_text(item.get("input"))},
                        ensure_ascii=False,
                    ),
                }
        if kind == "custom_tool_call_output":
            return {**item, "type": "function_call_output"}
        return item

    def _choice(self, choice: JsonValue) -> JsonValue:
        if not isinstance(choice, dict):
            return choice
        if choice.get("type") == "custom":
            return {
                **choice,
                "type": "function",
                "name": flatten_responses_tool_name(
                    _name(choice), namespace=_namespace(choice)
                ),
            }
        children = choice.get("tools")
        if isinstance(children, list):
            return {**choice, "tools": [self._choice(tool) for tool in children]}
        return choice

    def _custom(self, item: Mapping[str, JsonValue]) -> ResponsesToolIdentity | None:
        name = item.get("name")
        if not isinstance(name, str):
            return None
        namespace = _namespace(item)
        if namespace is not None:
            identity = self._identities.get((namespace, name))
        else:
            candidates = {
                identity
                for (_, wire_name), identity in self._identities.items()
                if wire_name == name
            }
            identity = next(iter(candidates)) if len(candidates) == 1 else None
        return identity if identity is not None and identity.kind == "custom" else None

    def restore_item(self, value: JsonValue) -> JsonValue:
        if not isinstance(value, dict):
            return value
        if value.get("type") == "tool_search_output":
            return {
                **value,
                "tools": self.restore_tools(value.get("tools"), scope=_scope(value)),
            }
        if (
            value.get("type") != "function_call"
            or (identity := self._custom(value)) is None
        ):
            return value
        arguments = value.get("arguments")
        item: JsonObject = {
            **{key: child for key, child in value.items() if key != "arguments"},
            "type": "custom_tool_call",
            "name": identity.name,
            "input": custom_tool_input_text_from_arguments(arguments)
            if isinstance(arguments, str)
            else "",
        }
        if identity.namespace is not None:
            item["namespace"] = identity.namespace
        return item

    def restore_tools(
        self, tools: JsonValue, namespace: str | None = None, scope: str | None = None
    ) -> JsonValue:
        if not isinstance(tools, list):
            return tools
        result: list[JsonValue] = []
        for tool in tools:
            if isinstance(tool, dict):
                if tool.get("type") == "namespace":
                    tool = {
                        **tool,
                        "tools": self.restore_tools(
                            tool.get("tools"), _name(tool), scope
                        ),
                    }
                else:
                    edits = [
                        edit
                        for edit in self._edits
                        if edit.namespace == namespace
                        and edit.adapted.get("type") == tool.get("type")
                        and edit.adapted.get("name") == tool.get("name")
                    ]
                    scoped = [edit for edit in edits if edit.scope == scope]
                    if scoped:
                        edits = scoped
                    if edits:
                        incoming = tool
                        edit = max(
                            edits,
                            key=lambda edit: sum(
                                key in incoming and incoming[key] == value
                                for key, value in edit.adapted.items()
                            ),
                        )
                        tool = dict(tool)
                        for key in edit.original.keys() | edit.adapted.keys():
                            if (key in edit.original) != (
                                key in edit.adapted
                            ) or edit.original.get(key) != edit.adapted.get(key):
                                if key in edit.original:
                                    tool[key] = deepcopy(edit.original[key])
                                else:
                                    tool.pop(key, None)
            result.append(tool)
        return result

    def restore_choice(self, value: JsonValue) -> JsonValue:
        if not isinstance(value, dict):
            return value
        if (
            value.get("type") == "function"
            and (identity := self._custom(value)) is not None
        ):
            value = {**value, "type": "custom", "name": identity.name}
            if identity.namespace is not None:
                value["namespace"] = identity.namespace
        children = value.get("tools")
        if isinstance(children, list):
            value = {
                **value,
                "tools": [self.restore_choice(tool) for tool in children],
            }
        return value

    def event_adapter(self) -> ResponsesToolEventAdapter | None:
        if self._edits or any(
            identity.kind == "custom" for identity in self._identities.values()
        ):
            return ResponsesToolEventAdapter(self)
        return None


class ResponsesToolEventAdapter:
    """Undo one request's tool edits with fresh event state for each attempt."""

    def __init__(self, tools: ResponsesToolAdapter) -> None:
        self._tools = tools
        self._custom_items: set[str] = set()
        self._sequence = 0

    def feed(
        self, event_type: str, payload: JsonObject
    ) -> Iterable[tuple[str, JsonObject]]:
        data = deepcopy(payload)
        sequence = data.get("sequence_number")
        if isinstance(sequence, int) and not isinstance(sequence, bool):
            self._sequence = max(self._sequence, sequence)
        original_item = data.get("item")
        item = self._tools.restore_item(original_item)
        if isinstance(item, dict):
            data["item"] = item
            if (
                isinstance(original_item, dict)
                and original_item.get("type") == "function_call"
                and item.get("type") == "custom_tool_call"
            ):
                if isinstance(item_id := item.get("id"), str):
                    self._custom_items.add(item_id)
                if event_type == "response.output_item.done":
                    coordinates = {
                        "item_id": item.get("id"),
                        "output_index": data.get("output_index"),
                    }
                    if item["input"]:
                        yield self._emit(
                            "response.custom_tool_call_input.delta",
                            {**coordinates, "delta": item["input"]},
                        )
                    yield self._emit(
                        "response.custom_tool_call_input.done",
                        {**coordinates, "input": item["input"]},
                    )
        if (
            event_type
            in {
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
            }
            and data.get("item_id") in self._custom_items
        ):
            return
        response = data.get("response")
        if isinstance(response, dict):
            if isinstance(output := response.get("output"), list):
                response["output"] = [
                    self._tools.restore_item(value) for value in output
                ]
            if "tools" in response:
                response["tools"] = self._tools.restore_tools(response["tools"])
            if "tool_choice" in response:
                response["tool_choice"] = self._tools.restore_choice(
                    response["tool_choice"]
                )
        yield self._emit(event_type, data)

    def _emit(self, event_type: str, payload: JsonObject) -> tuple[str, JsonObject]:
        payload = {**payload, "type": event_type, "sequence_number": self._sequence}
        self._sequence += 1
        return event_type, payload


def _name(value: Mapping[str, JsonValue]) -> str:
    return required_str(value.get("name"), "tool.name")


def _namespace(value: Mapping[str, JsonValue]) -> str | None:
    return optional_str(value.get("namespace"))


def _scope(value: Mapping[str, JsonValue]) -> str | None:
    return optional_str(value.get("call_id")) or optional_str(value.get("id"))
