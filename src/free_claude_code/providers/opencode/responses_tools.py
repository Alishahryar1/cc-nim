"""Adapt native Responses tools to OpenCode's supported tool formats."""

import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.failures import ExecutionFailure
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.openai_responses import (
    OpenAIResponsesRequest,
    ResponsesStreamFailure,
    convert_tools,
    custom_tool_input_text_from_arguments,
    responses_tool_name_to_anthropic_name,
)
from free_claude_code.providers.openai_responses.presentation import (
    NativeResponsesPresenter,
)

from .tool_search import normalize_tool_search


@dataclass(frozen=True, slots=True)
class _CustomTool:
    name: str
    namespace: str | None
    definition: JsonObject


class OpenCodeResponsesTools:
    def __init__(self, request: OpenAIResponsesRequest) -> None:
        self.customs: dict[tuple[str | None, str], _CustomTool] = {}
        self.request = request.model_copy(deep=True)
        if self.request.tools:
            self.request.tools = self._tools(self.request.tools)
        if isinstance(self.request.input, list):
            self.request.input = [self._input(item) for item in self.request.input]
        self.request.tool_choice = self._choice(self.request.tool_choice)

    def _tools(
        self, tools: list[JsonObject], namespace: str | None = None
    ) -> list[JsonObject]:
        result: list[JsonObject] = []
        names: set[str] = set()
        for tool in tools:
            tool = normalize_tool_search(tool)
            children = tool.get("tools")
            if tool.get("type") == "namespace" and isinstance(children, list):
                tool = {
                    **tool,
                    "tools": self._tools(
                        [
                            dict(child)
                            for child in children
                            if isinstance(child, Mapping)
                        ],
                        _name(tool),
                    ),
                }
            elif tool.get("type") == "custom":
                source = tool.get("custom")
                source = dict(source) if isinstance(source, Mapping) else tool
                name = _name(source)
                wrapped = (
                    [{"type": "namespace", "name": namespace, "tools": [tool]}]
                    if namespace
                    else [tool]
                )
                # Reuse the shared custom-tool schema and grammar description.
                converted_tools = convert_tools(wrapped)
                assert converted_tools is not None
                converted = converted_tools[0]
                wire_name = converted["name"]
                identity = (namespace, wire_name)
                if identity in self.customs:
                    raise InvalidRequestError(
                        "Custom tool names collide after conversion."
                    )
                self.customs[identity] = _CustomTool(name, namespace, tool)
                tool = {
                    "type": "function",
                    "name": wire_name,
                    "parameters": converted["input_schema"],
                    "description": converted.get("description", ""),
                    "strict": False,
                }
            elif tool.get("type") in {"web_search", "web_search_preview"}:
                content_types = tool.get("search_content_types")
                if content_types is not None:
                    if (
                        not isinstance(content_types, list)
                        or "text" not in content_types
                    ):
                        raise InvalidRequestError(
                            "OpenCode supports text web search only."
                        )
                    tool = {
                        key: value
                        for key, value in tool.items()
                        if key != "search_content_types"
                    }
            name = tool.get("name")
            if isinstance(name, str):
                if name in names:
                    raise InvalidRequestError("Tool names collide after conversion.")
                names.add(name)
            result.append(tool)
        return result

    @staticmethod
    def _input(item: JsonValue) -> JsonValue:
        if not isinstance(item, Mapping):
            return item
        if item.get("type") == "custom_tool_call":
            return {
                **{key: value for key, value in item.items() if key != "input"},
                "type": "function_call",
                "name": responses_tool_name_to_anthropic_name(
                    _name(item), namespace=_namespace(item)
                ),
                "arguments": json.dumps(
                    {"input": item.get("input", "")}, ensure_ascii=False
                ),
            }
        if item.get("type") == "custom_tool_call_output":
            return {**item, "type": "function_call_output"}
        return item

    def _choice(self, choice: JsonValue) -> JsonValue:
        if not isinstance(choice, Mapping):
            return choice
        if choice.get("type") == "custom":
            return {
                **choice,
                "type": "function",
                "name": responses_tool_name_to_anthropic_name(
                    _name(choice), namespace=_namespace(choice)
                ),
            }
        tools = choice.get("tools")
        if isinstance(tools, list):
            return {**choice, "tools": [self._choice(tool) for tool in tools]}
        return choice

    def presenter(self, public_model: str) -> OpenCodeResponsesPresenter:
        return OpenCodeResponsesPresenter(public_model, self.customs)


class OpenCodeResponsesPresenter:
    """Restore custom calls at completion, like the shared Chat conversion does."""

    def __init__(
        self, model: str, customs: Mapping[tuple[str | None, str], _CustomTool]
    ) -> None:
        self._native = NativeResponsesPresenter(public_model=model)
        self._customs = customs
        self._custom_items: set[str] = set()
        self._sequence = 0

    @property
    def completed(self) -> bool:
        return self._native.completed

    @property
    def terminal_failure_completes_wire(self) -> bool:
        return True

    def start(self) -> Iterable[str]:
        return self._native.start()

    def _item(self, value: JsonValue) -> JsonValue:
        if not isinstance(value, Mapping) or value.get("type") != "function_call":
            return value
        name = value.get("name")
        custom = (
            self._customs.get((_namespace(value), name))
            if isinstance(name, str)
            else None
        )
        if custom is None:
            return value
        arguments = value.get("arguments")
        result: JsonObject = {
            **{key: child for key, child in value.items() if key != "arguments"},
            "type": "custom_tool_call",
            "name": custom.name,
            "input": custom_tool_input_text_from_arguments(arguments)
            if isinstance(arguments, str) and arguments
            else "",
        }
        if custom.namespace is not None:
            result["namespace"] = custom.namespace
        return result

    def _tools(self, tools: JsonValue, namespace: str | None = None) -> JsonValue:
        if not isinstance(tools, list):
            return tools
        result: list[JsonValue] = []
        for tool in tools:
            if isinstance(tool, Mapping):
                name = tool.get("name")
                if (
                    tool.get("type") == "function"
                    and isinstance(name, str)
                    and (namespace, name) in self._customs
                ):
                    tool = self._customs[(namespace, name)].definition
                elif tool.get("type") == "namespace":
                    tool = {
                        **tool,
                        "tools": self._tools(tool.get("tools"), _name(tool)),
                    }
            result.append(tool)
        return result

    def feed(self, event_type: str, payload: JsonObject) -> Iterable[str]:
        data = deepcopy(payload)
        item = self._item(data.get("item"))
        if isinstance(item, dict) and item.get("type") == "custom_tool_call":
            data["item"] = item
            item_id = item.get("id")
            if isinstance(item_id, str):
                self._custom_items.add(item_id)
            if event_type == "response.output_item.done":
                coordinates = {
                    "item_id": item.get("id"),
                    "output_index": data.get("output_index"),
                }
                yield from self._emit(
                    "response.custom_tool_call_input.delta",
                    {**coordinates, "delta": item["input"]},
                )
                yield from self._emit(
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
            output = response.get("output")
            if isinstance(output, list):
                response["output"] = [self._item(value) for value in output]
            if "tools" in response:
                response["tools"] = self._tools(response["tools"])
        yield from self._emit(event_type, data)

    def _emit(self, event_type: str, payload: JsonObject) -> Iterable[str]:
        payload = {**payload, "type": event_type, "sequence_number": self._sequence}
        self._sequence += 1
        return self._native.feed(event_type, payload)

    def terminal_failure(
        self, raw_error: Exception, failure: ExecutionFailure
    ) -> Iterable[str]:
        if (
            isinstance(raw_error, ResponsesStreamFailure)
            and raw_error.event_type == "response.failed"
            and raw_error.payload is not None
        ):
            return self.feed(raw_error.event_type, raw_error.payload)
        return self._native.terminal_failure(raw_error, failure)


def _name(value: Mapping[str, JsonValue]) -> str:
    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise InvalidRequestError("Tool name must be a non-empty string.")
    return name


def _namespace(value: Mapping[str, JsonValue]) -> str | None:
    namespace = value.get("namespace")
    return namespace if isinstance(namespace, str) else None
