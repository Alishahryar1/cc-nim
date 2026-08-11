"""Provider model-list response parsing helpers."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from free_claude_code.application.model_metadata import (
    ProviderModelInfo as _ProviderModelInfo,
)


class ModelListResponseError(ValueError):
    """A provider model-list response cannot be parsed safely."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def model_infos_from_ids(
    model_ids: Iterable[str], *, supports_thinking: bool | None = None
) -> frozenset[_ProviderModelInfo]:
    """Build unknown-capability model metadata from plain provider model ids."""
    return frozenset(
        _ProviderModelInfo(model_id=model_id, supports_thinking=supports_thinking)
        for model_id in model_ids
        if model_id.strip()
    )


def extract_openai_model_infos(
    payload: Any,
    *,
    provider_name: str,
    collection_field: str | None = "data",
    id_field: str = "id",
    aliases_field: str | None = None,
    field_equals: tuple[str, str] | None = None,
    required_null_field: str | None = None,
    required_sequence_items: tuple[tuple[str, str], ...] = (),
    tags_field: str | None = None,
    thinking_tag: str = "reasoning",
    non_thinking_tag: str | None = None,
    thinking_boolean_path: tuple[str, ...] | None = None,
) -> frozenset[_ProviderModelInfo]:
    """Extract routable IDs from an OpenAI-compatible model-list response."""
    model_infos: set[_ProviderModelInfo] = set()
    item_location = collection_field or "root-array"
    for item in model_list_items(
        payload,
        provider_name=provider_name,
        collection_field=collection_field,
    ):
        model_id = _field(item, id_field)
        if not isinstance(model_id, str) or not model_id.strip():
            raise _malformed(
                provider_name,
                f"expected every {item_location} item to include {id_field}",
            )
        included = True
        if field_equals is not None:
            field_name, expected_value = field_equals
            field_value = _field(item, field_name)
            if not isinstance(field_value, str):
                raise _malformed(
                    provider_name,
                    f"expected every {item_location} item to include "
                    f"string {field_name}",
                )
            if field_value != expected_value:
                included = False

        if required_null_field is not None:
            if not _has_field(item, required_null_field):
                raise _malformed(
                    provider_name,
                    f"expected every {item_location} item to include "
                    f"{required_null_field}",
                )
            if _field(item, required_null_field) is not None:
                included = False

        for field_name, required_item in required_sequence_items:
            values = _field(item, field_name)
            if not _is_sequence(values) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise _malformed(
                    provider_name,
                    f"expected every {item_location} item to include "
                    f"{field_name} string array",
                )
            if required_item not in values:
                included = False

        supports_thinking: bool | None = None
        if tags_field is not None:
            tags_value = _field(item, tags_field)
            if not _is_sequence(tags_value) or any(
                not isinstance(tag, str) or not tag.strip() for tag in tags_value
            ):
                raise _malformed(
                    provider_name,
                    f"expected every {item_location} item to include "
                    f"{tags_field} string array",
                )
            tags = frozenset(tags_value)
            if thinking_tag in tags:
                supports_thinking = True
            elif non_thinking_tag is not None and non_thinking_tag in tags:
                supports_thinking = False

        if thinking_boolean_path is not None:
            capability = _path(item, thinking_boolean_path)
            if capability is not _MISSING:
                if not isinstance(capability, bool):
                    raise _malformed(
                        provider_name,
                        f"expected {'.'.join(thinking_boolean_path)} to be boolean",
                    )
                supports_thinking = capability

        if not included:
            continue

        model_infos.add(
            _ProviderModelInfo(
                model_id=model_id,
                supports_thinking=supports_thinking,
            )
        )
        if aliases_field is not None:
            aliases = _field(item, aliases_field)
            if not _is_sequence(aliases):
                raise _malformed(
                    provider_name,
                    f"expected every {item_location} item to include "
                    f"{aliases_field} array",
                )
            for alias in aliases:
                if not isinstance(alias, str) or not alias.strip():
                    raise _malformed(
                        provider_name,
                        f"expected every {aliases_field} item to be a model id",
                    )
                model_infos.add(
                    _ProviderModelInfo(
                        model_id=alias,
                        supports_thinking=supports_thinking,
                    )
                )

    if not model_infos:
        raise _malformed(provider_name, "response did not include any model ids")
    return frozenset(model_infos)


def extract_tool_capable_model_infos(
    payload: Any, *, provider_name: str
) -> frozenset[_ProviderModelInfo]:
    """Extract tool-capable models with ``supported_parameters`` metadata."""
    data = model_list_items(payload, provider_name=provider_name)

    model_infos: set[_ProviderModelInfo] = set()
    for item in data:
        model_id = _field(item, "id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise _malformed(provider_name, "expected every data item to include id")

        supported_parameters = _field(item, "supported_parameters")
        if not _is_sequence(supported_parameters):
            continue
        supported_parameter_names = {
            param for param in supported_parameters if isinstance(param, str)
        }
        if supported_parameter_names.isdisjoint({"tools", "tool_choice"}):
            continue
        model_infos.add(
            _ProviderModelInfo(
                model_id=model_id,
                supports_thinking="reasoning" in supported_parameter_names,
            )
        )

    return frozenset(model_infos)


def model_list_items(
    payload: Any,
    *,
    provider_name: str,
    collection_field: str | None = "data",
) -> tuple[Any, ...]:
    """Return a validated OpenAI-shaped model-list data array."""
    data = payload if collection_field is None else _field(payload, collection_field)
    if not _is_sequence(data):
        location = (
            "root array"
            if collection_field is None
            else (f"top-level {collection_field} array")
        )
        raise _malformed(
            provider_name,
            f"expected {location}",
        )
    return tuple(data)


def _field(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _has_field(item: Any, name: str) -> bool:
    if isinstance(item, Mapping):
        return name in item
    return hasattr(item, name)


_MISSING = object()


def _path(item: Any, path: tuple[str, ...]) -> Any:
    current = item
    for name in path:
        if not _has_field(current, name):
            return _MISSING
        current = _field(current, name)
    return current


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )


def _malformed(provider_name: str, reason: str) -> ModelListResponseError:
    return ModelListResponseError(
        f"{provider_name} model-list response is malformed: {reason}"
    )
