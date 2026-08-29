import json
from pathlib import Path
from unittest.mock import patch

import pytest

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.ports import (
    RequestRuntimeLease,
    RequestRuntimePort,
)
from free_claude_code.config.settings import Settings
from free_claude_code.runtime.codex_catalog import CodexModelCatalogPublisher


class FakeRequestRuntime(RequestRuntimePort):
    def __init__(
        self,
        *,
        settings: Settings,
        cached_infos: tuple[ProviderModelInfo, ...] = (),
    ) -> None:
        self._settings = settings
        self._cached_infos = cached_infos

    async def acquire(self) -> RequestRuntimeLease:
        raise AssertionError("Catalog publication must not acquire a provider lease.")

    def current_settings(self) -> Settings:
        return self._settings

    def cached_model_info(
        self, provider_id: str, model_id: str
    ) -> ProviderModelInfo | None:
        del provider_id, model_id
        return None

    def cached_prefixed_model_infos(self) -> tuple[ProviderModelInfo, ...]:
        return self._cached_infos


def _runtime() -> FakeRequestRuntime:
    settings = Settings().model_copy(update={"model": "nvidia_nim/configured"})
    return FakeRequestRuntime(
        settings=settings,
        cached_infos=(ProviderModelInfo("open_router/discovered"),),
    )


def _catalog_slugs(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [model["slug"] for model in payload["models"]]


def test_publisher_projects_the_application_catalog_without_compatibility_ids(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "codex-model-catalog.json"
    publisher = CodexModelCatalogPublisher(catalog_path)

    publisher.publish(_runtime())

    assert _catalog_slugs(catalog_path) == [
        "nvidia_nim/configured",
        "open_router/discovered",
    ]


def test_startup_publication_creates_missing_catalog_and_preserves_existing(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "codex-model-catalog.json"
    publisher = CodexModelCatalogPublisher(catalog_path)

    publisher.ensure_exists(_runtime())
    assert _catalog_slugs(catalog_path) == [
        "nvidia_nim/configured",
        "open_router/discovered",
    ]

    catalog_path.write_text("complete prior catalog\n", encoding="utf-8")
    publisher.ensure_exists(_runtime())

    assert catalog_path.read_text(encoding="utf-8") == "complete prior catalog\n"


def test_empty_projection_preserves_existing_catalog(tmp_path: Path) -> None:
    catalog_path = tmp_path / "codex-model-catalog.json"
    catalog_path.write_text("last known good\n", encoding="utf-8")
    publisher = CodexModelCatalogPublisher(catalog_path)

    with (
        patch(
            "free_claude_code.runtime.codex_catalog.build_codex_model_catalog",
            return_value={"models": []},
        ),
        pytest.raises(ValueError, match="no routable models"),
    ):
        publisher.publish(_runtime())

    assert catalog_path.read_text(encoding="utf-8") == "last known good\n"
