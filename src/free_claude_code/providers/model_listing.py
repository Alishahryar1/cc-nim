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
    payload: Any, *, provider_name: str
) -> frozenset[_ProviderModelInfo]:
    """Extract model metadata from an OpenAI-compatible ``/models`` response."""
    data = _field(payload, "data")
    if not _is_sequence(data):
        raise _malformed(provider_name, "expected top-level data array")

    model_ids: set[str] = set()
    for item in data:
        model_id = _field(item, "id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise _malformed(provider_name, "expected every data item to include id")
        model_ids.add(model_id)

    if not model_ids:
        raise _malformed(provider_name, "response did not include any model ids")
    return model_infos_from_ids(model_ids)


def extract_openrouter_tool_model_infos(
    payload: Any, *, provider_name: str
) -> frozenset[_ProviderModelInfo]:
    """Extract OpenRouter tool-capable model ids with thinking capability metadata."""
    data = _field(payload, "data")
    if not _is_sequence(data):
        raise _malformed(provider_name, "expected top-level data array")

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


def _field(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    )


def _malformed(provider_name: str, reason: str) -> ModelListResponseError:
    return ModelListResponseError(
        f"{provider_name} model-list response is malformed: {reason}"
    )
