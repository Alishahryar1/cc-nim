"""Shared FCC model-catalog projection for installed client launchers."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.request import Request

from free_claude_code.cli.local_http import open_local_request
from free_claude_code.config.provider_catalog import SUPPORTED_PROVIDER_IDS
from free_claude_code.core.gateway_model_ids import (
    GATEWAY_MODEL_ID_PREFIX,
    NO_THINKING_GATEWAY_MODEL_ID_PREFIX,
)
from free_claude_code.core.json_types import JsonObject, JsonValue

from .common import PROXY_PREFLIGHT_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class ClientModel:
    """One direct FCC model reference suitable for a client-side catalog."""

    wire_slug: str
    provider_model_ref: str
    display_name: str
    allows_reasoning: bool


def client_models_from_response(
    models_response: Mapping[str, JsonValue],
) -> tuple[ClientModel, ...]:
    """Project an FCC `/v1/models` response into direct client model records."""

    candidates = _catalog_candidates(models_response)
    normal_provider_refs = {
        candidate.provider_model_ref
        for candidate in candidates
        if candidate.allows_reasoning
    }
    models: list[ClientModel] = []
    seen_slugs: set[str] = set()

    for candidate in candidates:
        if (
            not candidate.allows_reasoning
            and candidate.provider_model_ref in normal_provider_refs
        ):
            continue
        if candidate.wire_slug in seen_slugs:
            continue
        seen_slugs.add(candidate.wire_slug)
        models.append(candidate)

    return tuple(models)


def fetch_proxy_models_response(proxy_root_url: str, auth_token: str) -> JsonObject:
    """Fetch the authenticated FCC-local `/v1/models` response directly."""

    url = f"{proxy_root_url.rstrip('/')}/v1/models"
    request = Request(
        url,
        headers={"Authorization": f"Bearer {auth_token}"},
        method="GET",
    )
    with open_local_request(
        request, timeout=PROXY_PREFLIGHT_TIMEOUT_SECONDS
    ) as response:
        payload: JsonValue = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("model list response was not a JSON object")
    return payload


def _catalog_candidates(
    models_response: Mapping[str, JsonValue],
) -> list[ClientModel]:
    data = models_response.get("data")
    if not isinstance(data, list):
        return []

    candidates: list[ClientModel] = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        model_id = _string_value(item.get("id"))
        if model_id is None:
            continue
        candidate = _candidate_from_model_id(
            model_id,
            display_name=_string_value(item.get("display_name")) or model_id,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _candidate_from_model_id(model_id: str, *, display_name: str) -> ClientModel | None:
    prefix, separator, remainder = model_id.partition("/")
    if not separator:
        return None

    if prefix == GATEWAY_MODEL_ID_PREFIX:
        if not _is_provider_model_ref(remainder):
            return None
        return ClientModel(
            wire_slug=remainder,
            provider_model_ref=remainder,
            display_name=display_name,
            allows_reasoning=True,
        )

    if prefix == NO_THINKING_GATEWAY_MODEL_ID_PREFIX:
        if not _is_provider_model_ref(remainder):
            return None
        return ClientModel(
            wire_slug=model_id,
            provider_model_ref=remainder,
            display_name=display_name,
            allows_reasoning=False,
        )

    if prefix in SUPPORTED_PROVIDER_IDS and remainder:
        return ClientModel(
            wire_slug=model_id,
            provider_model_ref=model_id,
            display_name=display_name,
            allows_reasoning=True,
        )

    return None


def _is_provider_model_ref(value: str) -> bool:
    provider_id, separator, provider_model = value.partition("/")
    return bool(separator and provider_model and provider_id in SUPPORTED_PROVIDER_IDS)


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) else None
