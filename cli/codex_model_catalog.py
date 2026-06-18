"""Build Codex model catalogs from the FCC model-list route."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from api.gateway_model_ids import (
    GATEWAY_MODEL_ID_PREFIX,
    NO_THINKING_GATEWAY_MODEL_ID_PREFIX,
)
from config.provider_ids import SUPPORTED_PROVIDER_IDS

SUPPORTED_REASONING_LEVELS = [
    {"effort": "low", "description": "Fast responses with lighter reasoning"},
    {
        "effort": "medium",
        "description": "Balances speed and reasoning depth for everyday tasks",
    },
    {"effort": "high", "description": "Greater reasoning depth for complex problems"},
    {
        "effort": "xhigh",
        "description": "Extra high reasoning depth for complex problems",
    },
]

CODEX_BASE_INSTRUCTIONS = (
    "You are Codex, a coding agent. Help the user understand, modify, test, "
    "and review code in their workspace. Follow the user's instructions, use "
    "tools when needed, and communicate concise progress and verification."
)


@dataclass(frozen=True, slots=True)
class _CatalogCandidate:
    slug: str
    provider_model_ref: str
    display_name: str
    force_no_thinking: bool


def build_codex_model_catalog(models_response: Mapping[str, Any]) -> dict[str, Any]:
    """Convert FCC `/v1/models` data into Codex `model_catalog_json` payload."""

    candidates = list(_catalog_candidates(models_response))
    normal_provider_refs = {
        candidate.provider_model_ref
        for candidate in candidates
        if not candidate.force_no_thinking
    }
    models: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()

    for candidate in candidates:
        if (
            candidate.force_no_thinking
            and candidate.provider_model_ref in normal_provider_refs
        ):
            continue
        if candidate.slug in seen_slugs:
            continue
        seen_slugs.add(candidate.slug)
        models.append(_codex_catalog_entry(candidate, priority=len(models)))

    return {"models": models}


def write_codex_model_catalog(catalog_path: Path, catalog: Mapping[str, Any]) -> None:
    """Atomically write a Codex model catalog JSON file."""

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = catalog_path.with_name(f".{catalog_path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(
        json.dumps(catalog, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(catalog_path)


def _catalog_candidates(
    models_response: Mapping[str, Any],
) -> list[_CatalogCandidate]:
    data = models_response.get("data")
    if not isinstance(data, list):
        return []

    candidates: list[_CatalogCandidate] = []
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


def _candidate_from_model_id(
    model_id: str, *, display_name: str
) -> _CatalogCandidate | None:
    prefix, separator, remainder = model_id.partition("/")
    if not separator:
        return None

    if prefix == GATEWAY_MODEL_ID_PREFIX:
        if not _is_provider_model_ref(remainder):
            return None
        return _CatalogCandidate(
            slug=remainder,
            provider_model_ref=remainder,
            display_name=display_name,
            force_no_thinking=False,
        )

    if prefix == NO_THINKING_GATEWAY_MODEL_ID_PREFIX:
        if not _is_provider_model_ref(remainder):
            return None
        return _CatalogCandidate(
            slug=model_id,
            provider_model_ref=remainder,
            display_name=display_name,
            force_no_thinking=True,
        )

    if prefix in SUPPORTED_PROVIDER_IDS and remainder:
        return _CatalogCandidate(
            slug=model_id,
            provider_model_ref=model_id,
            display_name=display_name,
            force_no_thinking=False,
        )

    return None


def _codex_catalog_entry(
    candidate: _CatalogCandidate, *, priority: int
) -> dict[str, Any]:
    return {
        "slug": candidate.slug,
        "display_name": candidate.display_name,
        "description": "Free Claude Code provider model",
        "default_reasoning_level": "medium",
        "supported_reasoning_levels": SUPPORTED_REASONING_LEVELS,
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": priority,
        "additional_speed_tiers": [],
        "service_tiers": [],
        "base_instructions": CODEX_BASE_INSTRUCTIONS,
        "supports_reasoning_summaries": True,
        "default_reasoning_summary": "none",
        "support_verbosity": True,
        "default_verbosity": "low",
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text_and_image",
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "supports_parallel_tool_calls": True,
        "supports_image_detail_original": True,
        "context_window": 200000,
        "max_context_window": 200000,
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
        "supports_search_tool": True,
        "use_responses_lite": False,
    }


def _is_provider_model_ref(value: str) -> bool:
    provider_id, separator, provider_model = value.partition("/")
    return bool(separator and provider_model and provider_id in SUPPORTED_PROVIDER_IDS)


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) else None
