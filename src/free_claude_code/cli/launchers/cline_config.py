"""Process-local Cline CLI configuration for FCC model routing."""

from dataclasses import dataclass, field
from datetime import UTC, datetime

from free_claude_code.core.json_types import JsonObject

from .common import proxy_v1_url
from .model_catalog import ClientModel

# Cline's released session gateway only instantiates built-in providers. FCC
# replaces this process-local Responses provider's endpoint and catalog instead
# of creating a custom provider that the picker can see but inference cannot.
CLINE_PROVIDER_ID = "openai-native"


@dataclass(frozen=True, slots=True)
class ClineConfig:
    """Private provider settings and model registry for one Cline process."""

    providers: JsonObject = field(repr=False)
    models: JsonObject = field(repr=False)


def build_cline_config(
    models: tuple[ClientModel, ...],
    *,
    proxy_root_url: str,
    auth_token: str,
    now: datetime | None = None,
) -> ClineConfig:
    """Translate a non-empty FCC model snapshot into Cline's file contracts."""

    if not models:
        raise ValueError("Cline requires at least one routable FCC model")

    timestamp = (
        (now or datetime.now(UTC)).astimezone(UTC).isoformat().replace("+00:00", "Z")
    )
    default_model = models[0].wire_slug
    provider_settings: JsonObject = {
        "provider": CLINE_PROVIDER_ID,
        "apiKey": auth_token,
        "model": default_model,
        "protocol": "openai-responses",
        "baseUrl": proxy_v1_url(proxy_root_url),
        "capabilities": ["streaming", "tools"],
    }
    model_entries: JsonObject = {
        model.wire_slug: {
            "name": model.display_name,
            "capabilities": [
                "streaming",
                "tools",
                *(["reasoning"] if model.allows_reasoning else []),
            ],
            "supportsReasoning": model.allows_reasoning,
            "apiFormat": "openai-responses",
        }
        for model in models
    }

    return ClineConfig(
        providers={
            "version": 1,
            "modes": {},
            "lastUsedProvider": CLINE_PROVIDER_ID,
            "providers": {
                CLINE_PROVIDER_ID: {
                    "settings": provider_settings,
                    "updatedAt": timestamp,
                    "tokenSource": "manual",
                }
            },
        },
        models={
            "version": 1,
            "providers": {
                CLINE_PROVIDER_ID: {
                    "provider": {
                        "name": "Free Claude Code",
                        "baseUrl": proxy_v1_url(proxy_root_url),
                        "defaultModelId": default_model,
                        "protocol": "openai-responses",
                        "client": "openai",
                        "capabilities": ["streaming", "tools"],
                    },
                    "models": model_entries,
                }
            },
        },
    )
