"""Provider model-list response parsing helpers."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from providers.exceptions import ModelListResponseError


@dataclass(frozen=True, slots=True)
class ProviderModelInfo:
    """Internal provider model metadata used for gateway model-list shaping."""

    model_id: str
    supports_thinking: bool | None = None


def model_infos_from_ids(
    model_ids: Iterable[str], *, supports_thinking: bool | None = None
) -> frozenset[ProviderModelInfo]:
    """Build unknown-capability model metadata from plain provider model ids."""
    return frozenset(
        ProviderModelInfo(model_id=model_id, supports_thinking=supports_thinking)
        for model_id in model_ids
        if model_id.strip()
    )


def extract_openai_model_ids(payload: Any, *, provider_name: str) -> frozenset[str]:
    """Extract model ids from an OpenAI-compatible ``/models`` response."""
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
    return frozenset(model_ids)


def extract_openrouter_tool_model_ids(
    payload: Any, *, provider_name: str
) -> frozenset[str]:
    """Extract OpenRouter model ids that advertise tool-use support."""
    return frozenset(
        info.model_id
        for info in extract_openrouter_tool_model_infos(
            payload, provider_name=provider_name
        )
    )


def extract_openrouter_tool_model_infos(
    payload: Any, *, provider_name: str
) -> frozenset[ProviderModelInfo]:
    """Extract OpenRouter tool-capable model ids with thinking capability metadata."""
    data = _field(payload, "data")
    if not _is_sequence(data):
        raise _malformed(provider_name, "expected top-level data array")

    model_infos: set[ProviderModelInfo] = set()
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
            ProviderModelInfo(
                model_id=model_id,
                supports_thinking="reasoning" in supported_parameter_names,
            )
        )

    return frozenset(model_infos)


def extract_ollama_model_ids(payload: Any, *, provider_name: str) -> frozenset[str]:
    """Extract model ids from Ollama's native ``/api/tags`` response."""
    models = _field(payload, "models")
    if not _is_sequence(models):
        raise _malformed(provider_name, "expected top-level models array")

    model_ids: set[str] = set()
    for item in models:
        item_ids: list[str] = []
        for key in ("model", "name"):
            value = _field(item, key)
            if isinstance(value, str) and value.strip():
                item_ids.append(value)
        if not item_ids:
            raise _malformed(
                provider_name,
                "expected every models item to include model or name",
            )
        model_ids.update(item_ids)

    if not model_ids:
        raise _malformed(provider_name, "response did not include any model ids")
    return frozenset(model_ids)


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
