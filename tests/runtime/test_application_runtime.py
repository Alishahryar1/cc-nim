from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.config.admin.persistence import PreparedAdminUpdate
from free_claude_code.config.settings import Settings
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager


class TrackingRuntime(ProviderRuntime):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.cleanup_calls = 0

    async def cleanup(self) -> None:
        self.cleanup_calls += 1
        await super().cleanup()


class TrackingFactory:
    def __init__(self) -> None:
        self.runtimes: list[TrackingRuntime] = []
        self.fail = False
        self.events: list[str] = []

    def __call__(self, settings: Settings) -> ProviderRuntime:
        self.events.append(f"construct:{settings.model}")
        if self.fail:
            raise RuntimeError("candidate failed")
        runtime = TrackingRuntime(settings)
        self.runtimes.append(runtime)
        return runtime


def _settings(model: str, *, port: int = 8082) -> Settings:
    return Settings().model_copy(update={"model": model, "port": port})


def _prepared(
    settings: Settings,
    tmp_path,
    *,
    pending_fields: tuple[str, ...] = (),
) -> PreparedAdminUpdate:
    return PreparedAdminUpdate(
        target_values={"MODEL": settings.model},
        settings=settings,
        errors=(),
        pending_fields=pending_fields,
        path=tmp_path / ".env",
    )


def _applied_response(pending_fields: tuple[str, ...] = ()) -> dict[str, object]:
    return {
        "applied": True,
        "valid": True,
        "errors": [],
        "env_preview": "MODEL=updated\n",
        "path": ".env",
        "pending_fields": list(pending_fields),
    }


@pytest.mark.asyncio
async def test_provider_apply_constructs_before_commit_then_publishes(tmp_path) -> None:
    factory = TrackingFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/old"),
        runtime_factory=factory,
    )
    runtime = ApplicationRuntime(manager)
    prepared = _prepared(_settings("nvidia_nim/new"), tmp_path)
    factory.events.clear()

    def commit(_prepared_update: PreparedAdminUpdate) -> dict[str, object]:
        factory.events.append("commit")
        assert manager.current_generation_id == 1
        return _applied_response()

    with (
        patch(
            "free_claude_code.runtime.application.prepare_admin_update",
            return_value=prepared,
        ),
        patch(
            "free_claude_code.runtime.application.commit_prepared_admin_update",
            side_effect=commit,
        ),
    ):
        result = await runtime.apply_admin_config({"MODEL": "nvidia_nim/new"})

    assert factory.events == ["construct:nvidia_nim/new", "commit"]
    assert manager.current_generation_id == 2
    assert manager.current_settings().model == "nvidia_nim/new"
    assert result["restart"] == {
        "required": False,
        "automatic": False,
        "admin_url": None,
        "fields": [],
    }
    await manager.close()


@pytest.mark.asyncio
async def test_candidate_failure_never_commits_and_preserves_current(tmp_path) -> None:
    factory = TrackingFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/old"),
        runtime_factory=factory,
    )
    runtime = ApplicationRuntime(manager)
    prepared = _prepared(_settings("nvidia_nim/new"), tmp_path)
    factory.fail = True

    with (
        patch(
            "free_claude_code.runtime.application.prepare_admin_update",
            return_value=prepared,
        ),
        patch(
            "free_claude_code.runtime.application.commit_prepared_admin_update"
        ) as commit,
        pytest.raises(RuntimeError, match="candidate failed"),
    ):
        await runtime.apply_admin_config({"MODEL": "nvidia_nim/new"})

    commit.assert_not_called()
    assert manager.current_generation_id == 1
    assert manager.current_settings().model == "nvidia_nim/old"
    await manager.close()


@pytest.mark.asyncio
async def test_persistence_failure_closes_candidate_and_preserves_current(
    tmp_path,
) -> None:
    factory = TrackingFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/old"),
        runtime_factory=factory,
    )
    runtime = ApplicationRuntime(manager)
    prepared = _prepared(_settings("nvidia_nim/new"), tmp_path)

    with (
        patch(
            "free_claude_code.runtime.application.prepare_admin_update",
            return_value=prepared,
        ),
        patch(
            "free_claude_code.runtime.application.commit_prepared_admin_update",
            side_effect=OSError("disk full"),
        ),
        pytest.raises(OSError, match="disk full"),
    ):
        await runtime.apply_admin_config({"MODEL": "nvidia_nim/new"})

    assert manager.current_generation_id == 1
    assert factory.runtimes[0].cleanup_calls == 0
    assert factory.runtimes[1].cleanup_calls == 1
    await manager.close()


@pytest.mark.asyncio
async def test_restart_required_apply_commits_without_hot_publication(tmp_path) -> None:
    factory = TrackingFactory()
    manager = ProviderRuntimeManager(
        _settings("nvidia_nim/old"),
        runtime_factory=factory,
    )
    restart = AsyncMock()
    runtime = ApplicationRuntime(manager, restart_callback=restart)
    prepared = _prepared(
        _settings("nvidia_nim/old", port=9090),
        tmp_path,
        pending_fields=("PORT",),
    )

    with (
        patch(
            "free_claude_code.runtime.application.prepare_admin_update",
            return_value=prepared,
        ),
        patch(
            "free_claude_code.runtime.application.commit_prepared_admin_update",
            return_value=_applied_response(("PORT",)),
        ) as commit,
    ):
        result = await runtime.apply_admin_config({"PORT": "9090"})

    commit.assert_called_once_with(prepared)
    assert manager.current_generation_id == 1
    assert len(factory.runtimes) == 1
    assert result["restart"] == {
        "required": True,
        "automatic": True,
        "admin_url": "http://127.0.0.1:9090/admin",
        "fields": ["PORT"],
    }
    restart.assert_not_awaited()
    await runtime.request_restart()
    restart.assert_awaited_once()
    await manager.close()
