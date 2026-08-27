"""Process-local Aider configuration for FCC model routing."""

import re
from dataclasses import dataclass, field

from free_claude_code.core.json_types import JsonObject

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
    metadata: JsonObject = {
        f"anthropic/{model.wire_slug}": {
            "litellm_provider": "anthropic",
            "mode": "chat",
        }
        for model in models
    }
    return AiderConfig(settings=settings, metadata=metadata)
