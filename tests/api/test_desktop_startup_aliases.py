"""Production startup path: the desktop mount serves seeded picker aliases.

Regression guard for the Greptile P1 "picker aliases remain inactive"
finding: ``ApplicationRuntime.start()`` seeds the process-global alias
snapshot from the warmed model cache, and the desktop-prefixed
``/v1/models`` mount serves those aliases while the bare mount keeps raw
provider refs for every other FCC client.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from free_claude_code.api.app import create_app
from free_claude_code.api.ports import ApiServices
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.settings import Settings
from free_claude_code.core import gateway_model_ids as gmi
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager


@pytest.fixture(autouse=True)
def _isolate_alias_state():
    gmi.clear_picker_aliases()
    yield
    gmi.clear_picker_aliases()


def _settings() -> Settings:
    return Settings().model_copy(
        update={
            "model": "nvidia_nim/meta/llama-3.3-70b-instruct",
            "nvidia_nim_api_key": "test-key",
            "messaging_platform": "none",
        }
    )


@pytest.mark.asyncio
async def test_runtime_startup_seeds_aliases_served_by_desktop_mount() -> None:
    settings = _settings()
    manager = ProviderRuntimeManager(settings)
    runtime = ApplicationRuntime(manager, transcriber=None)
    manager.cache_model_infos(
        "nvidia_nim",
        [
            ProviderModelInfo(
                model_id="meta/llama-3.3-70b-instruct", supports_thinking=None
            )
        ],
    )
    # Drop the publication side effect so only ``start()`` can reseed.
    gmi.clear_picker_aliases()
    assert gmi.has_picker_aliases() is False

    with (
        patch.object(manager, "warm_referenced_model_cache", new=AsyncMock()),
        patch.object(manager, "start_model_list_refresh"),
        patch.object(manager, "close", new=AsyncMock()),
    ):
        await runtime.start()
        try:
            assert gmi.has_picker_aliases() is True

            app = create_app(
                ApiServices(requests=manager, admin=runtime, tasks=runtime)
            )
            client = TestClient(app)

            desktop = client.get("/claude-desktop/v1/models")
            assert desktop.status_code == 200
            desktop_ids = [m["id"] for m in desktop.json()["data"]]
            assert "claude-sonnet-nim-0001" in desktop_ids
            assert "anthropic/nvidia_nim/meta/llama-3.3-70b-instruct" not in desktop_ids

            bare = client.get("/v1/models")
            assert bare.status_code == 200
            bare_ids = [m["id"] for m in bare.json()["data"]]
            assert "claude-sonnet-nim-0001" not in bare_ids
            assert "anthropic/nvidia_nim/meta/llama-3.3-70b-instruct" in bare_ids
        finally:
            await runtime.close()
