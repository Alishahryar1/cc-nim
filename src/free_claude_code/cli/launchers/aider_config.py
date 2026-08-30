"""Process-local Aider configuration for FCC model routing."""

import re
from dataclasses import dataclass, field

from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.model_capabilities import ModelInputModality

from .model_catalog import ClientModel

AIDER_API_KEY_ENV_PREFIX = "FCC_AIDER_PROXY_AUTH_"
_AIDER_API_KEY_ENV_PATTERN = re.compile(rf"{AIDER_API_KEY_ENV_PREFIX}[A-Z0-9]+")


@dataclass(frozen=True, slots=True)
class AiderConfig:
    """Secret-free model settings and metadata for one Aider process."""

    settings: list[JsonObject] = field(repr=False)
    metadata: JsonObject = field(repr=False)


def build_aider_config(
    models: tuple[ClientModel, ...],
    *,
    messages_url: str,
    api_key_env: str,
) -> AiderConfig:
    """Project a non-empty FCC Messages catalog into Aider's file contracts."""

    if not models:
        raise ValueError("Aider requires at least one routable FCC model")
    if _AIDER_API_KEY_ENV_PATTERN.fullmatch(api_key_env) is None:
        raise ValueError("invalid Aider proxy-auth environment variable name")

    settings: list[JsonObject] = [
        {
            "name": "aider/extra_params",
            "extra_params": {
                "api_base": messages_url,
                "api_key": f"os.environ/{api_key_env}",
            },
        }
    ]
    settings.extend(
        {
            "name": f"anthropic/{model.wire_slug}",
            "accepts_settings": (
                ["reasoning_effort"] if model.supports_reasoning else []
            ),
        }
        for model in models
        if model.supports_reasoning is not None
    )
    metadata: JsonObject = {
        f"anthropic/{model.wire_slug}": _model_metadata(model) for model in models
    }
    return AiderConfig(settings=settings, metadata=metadata)


def _model_metadata(model: ClientModel) -> JsonObject:
    metadata: JsonObject = {
        "litellm_provider": "anthropic",
        "mode": "chat",
    }
    if model.input_modalities is not None:
        metadata["supports_vision"] = ModelInputModality.IMAGE in model.input_modalities
    if model.context_window_tokens is not None:
        metadata["max_input_tokens"] = model.context_window_tokens
    if model.max_output_tokens is not None:
        metadata["max_output_tokens"] = model.max_output_tokens
    return metadata
