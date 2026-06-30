"""Provider-prefixed model reference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ConfiguredChatModelRef:
    """A unique configured chat model reference and the env keys that set it."""

    model_ref: str
    provider_id: str
    model_id: str
    sources: tuple[str, ...]


class ChatModelConfig(Protocol):
    model: str
    model_opus: str | None
    model_sonnet: str | None
    model_haiku: str | None


def parse_provider_type(model_ref: str) -> str:
    """Extract provider type from any 'provider/model' string."""

    return model_ref.split("/", 1)[0]


def parse_model_name(model_ref: str) -> str:
    """Extract model name from any 'provider/model' string."""

    return model_ref.split("/", 1)[1]


def configured_chat_model_refs(
    settings: ChatModelConfig,
) -> tuple[ConfiguredChatModelRef, ...]:
    """Return unique configured chat provider/model refs with source env keys."""

    candidates = (
        ("MODEL", settings.model),
        ("MODEL_OPUS", settings.model_opus),
        ("MODEL_SONNET", settings.model_sonnet),
        ("MODEL_HAIKU", settings.model_haiku),
    )
    sources_by_ref: dict[str, list[str]] = {}
    for source, model_ref in candidates:
        if model_ref is None:
            continue
        sources_by_ref.setdefault(model_ref, []).append(source)

    return tuple(
        ConfiguredChatModelRef(
            model_ref=model_ref,
            provider_id=parse_provider_type(model_ref),
            model_id=parse_model_name(model_ref),
            sources=tuple(sources),
        )
        for model_ref, sources in sources_by_ref.items()
    )
