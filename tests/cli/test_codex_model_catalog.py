from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from cli.codex_model_catalog import (
    build_codex_model_catalog,
    write_codex_model_catalog,
)


def _models_payload(*model_ids: str) -> dict[str, Any]:
    return {
        "data": [
            {
                "id": model_id,
                "display_name": model_id.replace("anthropic/", ""),
            }
            for model_id in model_ids
        ]
    }


def _catalog_models(catalog: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    models = catalog["models"]
    assert isinstance(models, list)
    catalog_models: list[Mapping[str, Any]] = []
    for model in models:
        assert isinstance(model, Mapping)
        catalog_models.append(cast(Mapping[str, Any], model))
    return catalog_models


def _slugs(catalog: Mapping[str, Any]) -> list[str]:
    slugs: list[str] = []
    for model in _catalog_models(catalog):
        slug = model["slug"]
        assert isinstance(slug, str)
        slugs.append(slug)
    return slugs


def test_codex_catalog_converts_configured_and_cached_models_to_direct_slugs() -> None:
    catalog = build_codex_model_catalog(
        _models_payload(
            "anthropic/nvidia_nim/nvidia/nemotron-3-super",
            "claude-3-freecc-no-thinking/nvidia_nim/nvidia/nemotron-3-super",
            "anthropic/open_router/meta-llama/llama-3.3-70b",
            "claude-3-freecc-no-thinking/open_router/meta-llama/llama-3.3-70b",
        )
    )

    assert _slugs(catalog) == [
        "nvidia_nim/nvidia/nemotron-3-super",
        "open_router/meta-llama/llama-3.3-70b",
    ]
    model = _catalog_models(catalog)[0]
    assert {
        "slug",
        "display_name",
        "description",
        "default_reasoning_level",
        "supported_reasoning_levels",
        "shell_type",
        "visibility",
        "supported_in_api",
        "priority",
        "additional_speed_tiers",
        "service_tiers",
    } <= set(model)


def test_codex_catalog_excludes_claude_compatibility_model_ids() -> None:
    catalog = build_codex_model_catalog(
        _models_payload(
            "claude-opus-4-20250514",
            "claude-3-haiku-20240307",
            "anthropic/nvidia_nim/provider-model",
        )
    )

    assert _slugs(catalog) == ["nvidia_nim/provider-model"]


def test_codex_catalog_skips_no_thinking_duplicate_when_normal_slug_exists() -> None:
    catalog = build_codex_model_catalog(
        _models_payload(
            "claude-3-freecc-no-thinking/nvidia_nim/provider-model",
            "anthropic/nvidia_nim/provider-model",
        )
    )

    assert _slugs(catalog) == ["nvidia_nim/provider-model"]


def test_codex_catalog_preserves_no_thinking_only_entries_for_routing() -> None:
    catalog = build_codex_model_catalog(
        _models_payload("claude-3-freecc-no-thinking/open_router/plain-model")
    )

    assert _slugs(catalog) == ["claude-3-freecc-no-thinking/open_router/plain-model"]


def test_codex_catalog_ordering_and_priorities_are_deterministic() -> None:
    catalog = build_codex_model_catalog(
        _models_payload(
            "anthropic/gemini/models/gemini-test",
            "anthropic/nvidia_nim/nvidia/test",
            "anthropic/gemini/models/gemini-test",
            "anthropic/open_router/provider/test",
        )
    )

    models = _catalog_models(catalog)
    assert _slugs(catalog) == [
        "gemini/models/gemini-test",
        "nvidia_nim/nvidia/test",
        "open_router/provider/test",
    ]
    assert [model["priority"] for model in models] == [0, 1, 2]


def test_codex_catalog_accepts_future_direct_provider_slugs() -> None:
    catalog = build_codex_model_catalog(
        _models_payload(
            "nvidia_nim/provider-model",
            "anthropic/open_router/provider-model",
        )
    )

    assert _slugs(catalog) == [
        "nvidia_nim/provider-model",
        "open_router/provider-model",
    ]


def test_generated_catalog_schema_is_accepted_by_installed_codex(
    tmp_path: Path,
) -> None:
    codex_binary = shutil.which("codex")
    if codex_binary is None:
        pytest.skip("Codex CLI is not installed")

    catalog_path = tmp_path / "codex-model-catalog.json"
    write_codex_model_catalog(
        catalog_path,
        build_codex_model_catalog(_models_payload("anthropic/nvidia_nim/test-model")),
    )

    result = subprocess.run(
        [
            codex_binary,
            "debug",
            "models",
            "-c",
            f"model_catalog_json={json.dumps(str(catalog_path))}",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "nvidia_nim/test-model" in result.stdout
