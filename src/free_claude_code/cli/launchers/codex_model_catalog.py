"""Build Codex model catalogs from the FCC model-list route."""

import json
import uuid
from collections.abc import Mapping
from pathlib import Path

from free_claude_code.core.json_types import JsonObject, JsonValue

from .model_catalog import ClientModel, client_models_from_response

SUPPORTED_REASONING_LEVELS: list[JsonObject] = [
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


def build_codex_model_catalog(models_response: Mapping[str, JsonValue]) -> JsonObject:
    """Convert FCC `/v1/models` data into Codex `model_catalog_json` payload."""

    return {
        "models": [
            _codex_catalog_entry(model, priority=priority)
            for priority, model in enumerate(
                client_models_from_response(models_response)
            )
        ]
    }


def write_codex_model_catalog(
    catalog_path: Path, catalog: Mapping[str, JsonValue]
) -> bool:
    """Atomically write changed Codex model catalog JSON."""

    content = (json.dumps(catalog, ensure_ascii=True, indent=2) + "\n").encode()
    try:
        if catalog_path.read_bytes() == content:
            return False
    except FileNotFoundError:
        pass

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = catalog_path.with_name(f".{catalog_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_bytes(content)
        temp_path.replace(catalog_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return True


def _codex_catalog_entry(candidate: ClientModel, *, priority: int) -> JsonObject:
    return {
        "slug": candidate.wire_slug,
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
