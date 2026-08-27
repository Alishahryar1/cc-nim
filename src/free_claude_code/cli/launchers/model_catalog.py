"""Shared FCC model-catalog projection for installed client launchers."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal
from urllib.request import Request

from free_claude_code.cli.local_http import open_local_request
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

    models: list[ClientModel] = []
    seen_slugs: set[str] = set()

    for candidate in _catalog_candidates(models_response):
        if candidate.wire_slug in seen_slugs:
            continue
        seen_slugs.add(candidate.wire_slug)
        models.append(candidate)

    return tuple(models)


def catalog_wire_slug_for_ref(
    models: Sequence[ClientModel],
    provider_model_ref: str | None,
) -> str | None:
    """Return the slug a client catalog advertises for one configured ref.

    A model the gateway reports as non-thinking is advertised under its
    no-thinking slug, not its bare provider ref, so a client told to select the
    bare ref would not find it in the catalog it was given.
    """

    if not provider_model_ref:
        return provider_model_ref

    for model in models:
        if model.provider_model_ref == provider_model_ref:
            return model.wire_slug
    return provider_model_ref


def fetch_proxy_models_response(
    proxy_root_url: str,
    auth_token: str,
    view: Literal["messages", "responses"] = "responses",
) -> JsonObject:
    """Fetch the authenticated FCC-local `/v1/models` response directly."""

    url = f"{proxy_root_url.rstrip('/')}/v1/models?view={view}"
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
        model_id = _nonempty_string(item.get("id"))
        if model_id is None:
            continue
        provider_model_ref = _provider_model_ref(item.get("provider_model_ref"))
        if provider_model_ref is None:
            continue
        candidates.append(
            ClientModel(
                wire_slug=model_id,
                provider_model_ref=provider_model_ref,
                display_name=_nonempty_string(item.get("display_name")) or model_id,
                allows_reasoning=model_id == provider_model_ref,
            )
        )
    return candidates


def _nonempty_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _provider_model_ref(value: object) -> str | None:
    ref = _nonempty_string(value)
    if ref is None:
        return None
    provider_id, separator, model_id = ref.partition("/")
    if not separator or not provider_id or not model_id:
        return None
    return ref
