"""Correlate Copilot's rotating opaque Responses IDs within one attempt."""

from copy import deepcopy

from free_claude_code.core.json_types import JsonObject
from free_claude_code.providers.failure_policy import RetryableProviderProtocolError


class CopilotResponsesEvents:
    """Keep the first opaque ID for each output position, preserving replay data."""

    def __init__(self) -> None:
        self._response_id: str | None = None
        self._item_ids: dict[int, str] = {}
        self._tool_positions: set[int] = set()

    def __call__(self, event_type: str, payload: JsonObject) -> JsonObject:
        data = deepcopy(payload)
        response = data.get("response")
        if isinstance(response, dict):
            identifier = response.get("id")
            if isinstance(identifier, str) and identifier:
                if self._response_id is None:
                    self._response_id = identifier
                response["id"] = self._response_id
            output = response.get("output")
            if isinstance(output, list):
                for position, item in enumerate(output):
                    if isinstance(item, dict):
                        self._item(position, item)
        if "response_id" in data and self._response_id is not None:
            data["response_id"] = self._response_id

        position = data.get("output_index")
        valid_position = (
            isinstance(position, int)
            and not isinstance(position, bool)
            and position >= 0
        )
        if event_type == "response.function_call_arguments.delta" and (
            not valid_position or position not in self._tool_positions
        ):
            raise RetryableProviderProtocolError(
                "Copilot sent tool arguments before identifying their output item."
            )
        if valid_position:
            item = data.get("item")
            if isinstance(item, dict):
                self._item(position, item)
            if "item_id" in data and position in self._item_ids:
                data["item_id"] = self._item_ids[position]
        return data

    def _item(self, position: int, item: JsonObject) -> None:
        identifier = item.get("id")
        if isinstance(identifier, str) and identifier:
            item["id"] = self._item_ids.setdefault(position, identifier)
            if item.get("type") == "function_call":
                self._tool_positions.add(position)
