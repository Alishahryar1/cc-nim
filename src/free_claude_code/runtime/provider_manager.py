"""Single-owner provider generations and application model catalog."""

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from loguru import logger

from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.application.model_metadata import (
    ProviderModelInfo,
    ProviderModelRefreshResult,
)
from free_claude_code.config.settings import Settings
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.base import BaseProvider
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.providers.runtime.discovery import (
    ProviderModelDiscovery,
    model_cache_provider_ids_for_settings,
)
from free_claude_code.providers.runtime.model_cache import ProviderModelCache
from free_claude_code.providers.runtime.validation import ConfiguredModelValidator

ProviderRuntimeFactory = Callable[[Settings], ProviderRuntime]
CommitConfig = Callable[[], None]


@dataclass(slots=True, eq=False)
class _ProviderGeneration:
    generation_id: int
    settings: Settings
    runtime: ProviderRuntime
    active_leases: int = 0
    retired: bool = False
    closed: bool = False
    drained: asyncio.Event = field(default_factory=asyncio.Event)
    cleanup_task: asyncio.Task[bool] | None = None

    def __post_init__(self) -> None:
        self.drained.set()


class ProviderGenerationLease:
    """Idempotent lease retaining one provider generation."""

    def __init__(
        self,
        manager: ProviderRuntimeManager,
        generation: _ProviderGeneration,
    ) -> None:
        self._manager = manager
        self._generation = generation
        self._released = False

    @property
    def generation_id(self) -> int:
        return self._generation.generation_id

    @property
    def settings(self) -> Settings:
        return self._generation.settings

    def is_provider_cached(self, provider_id: str) -> bool:
        return self._generation.runtime.is_cached(provider_id)

    def resolve_provider(self, provider_id: str) -> BaseProvider:
        return self._generation.runtime.resolve_provider(provider_id)

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._manager._release(self._generation)

    async def __aenter__(self) -> ProviderGenerationLease:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.release()


class ProviderRuntimeManager:
    """Own provider generations, leases, discovery, and model metadata."""

    def __init__(
        self,
        settings: Settings,
        *,
        runtime_factory: ProviderRuntimeFactory = ProviderRuntime,
    ) -> None:
        self._runtime_factory = runtime_factory
        self._replace_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._model_cache = ProviderModelCache(
            model_cache_provider_ids_for_settings(settings)
        )
        self._refresh_task: asyncio.Task[None] | None = None
        self._next_generation_id = 2
        self._retired: dict[int, _ProviderGeneration] = {}
        self._unpublished: set[ProviderRuntime] = set()
        self._closing = False
        self._closed = False
        self._current = _ProviderGeneration(
            generation_id=1,
            settings=settings,
            runtime=runtime_factory(settings),
        )
        self._trace_published(self._current, previous=None, reason="startup")

    @property
    def current_generation_id(self) -> int:
        return self._current.generation_id

    async def acquire(self) -> ProviderGenerationLease:
        if self._closing or self._closed:
            raise ApplicationUnavailableError("Provider runtime is shutting down.")
        generation = self._current
        generation.active_leases += 1
        generation.drained.clear()
        return ProviderGenerationLease(self, generation)

    def current_settings(self) -> Settings:
        return self._current.settings

    def cached_model_ids(self) -> dict[str, frozenset[str]]:
        return self._model_cache.cached_model_ids()

    def cached_model_supports_thinking(
        self, provider_id: str, model_id: str
    ) -> bool | None:
        return self._model_cache.cached_model_supports_thinking(provider_id, model_id)

    def cached_prefixed_model_infos(self) -> tuple[ProviderModelInfo, ...]:
        return self._model_cache.cached_prefixed_model_infos()

    def cache_model_infos(
        self,
        provider_id: str,
        model_infos: Iterable[ProviderModelInfo],
    ) -> None:
        self._model_cache.cache_model_infos(provider_id, model_infos)

    async def validate_configured_models(self) -> None:
        lease = await self.acquire()
        try:
            validator = ConfiguredModelValidator(
                lease.settings,
                lease.resolve_provider,
                self._model_cache,
            )
            await validator.validate_configured_models()
        finally:
            await lease.release()

    def start_model_list_refresh(self) -> None:
        """Start one non-blocking refresh for the current generation."""
        if self._closing or self._closed:
            return
        if self._refresh_task is not None and not self._refresh_task.done():
            return
        generation = self._current
        self._refresh_task = asyncio.create_task(
            self._refresh_generation_in_background(generation, only_missing=True)
        )

    async def refresh_model_list_cache(self) -> ProviderModelRefreshResult:
        """Run an explicit full refresh without racing replacement."""
        async with self._replace_lock:
            if self._closing or self._closed:
                raise ApplicationUnavailableError("Provider runtime is shutting down.")
            await self._cancel_refresh()
            return await self._refresh_generation(self._current, only_missing=False)

    async def replace(
        self,
        settings: Settings,
        *,
        commit: CommitConfig,
        reason: str = "admin_apply",
    ) -> int:
        """Prepare, commit, and atomically publish one replacement generation."""
        async with self._replace_lock:
            if self._closing or self._closed:
                raise ApplicationUnavailableError("Provider runtime is shutting down.")
            await self._cancel_refresh()
            await self._retry_unpublished_cleanup()
            candidate_id = self._next_generation_id
            candidate_runtime: ProviderRuntime | None = None
            try:
                candidate_runtime = self._runtime_factory(settings)
                commit()
            except Exception as exc:
                trace_event(
                    stage="runtime",
                    event="provider_generation.replace_failed",
                    source="runtime",
                    current_generation_id=self._current.generation_id,
                    candidate_generation_id=candidate_id,
                    reason=reason,
                    exc_type=type(exc).__name__,
                )
                if candidate_runtime is not None:
                    await self._cleanup_unpublished(candidate_runtime)
                raise

            self._next_generation_id += 1
            assert candidate_runtime is not None
            previous = self._current
            candidate = _ProviderGeneration(
                generation_id=candidate_id,
                settings=settings,
                runtime=candidate_runtime,
            )
            self._current = candidate
            self._model_cache.set_available_providers(
                model_cache_provider_ids_for_settings(settings)
            )
            previous.retired = True
            self._retired[previous.generation_id] = previous
            self._trace_published(candidate, previous=previous, reason=reason)
            self._trace_retired(previous, reason=reason)

            self._refresh_task = asyncio.create_task(
                self._refresh_generation_in_background(candidate, only_missing=False)
            )
            if previous.active_leases == 0:
                await self._close_generation(previous, forced=False)
            return candidate.generation_id

    async def close(self) -> None:
        """Reject new leases, drain existing work, and close every generation."""
        async with self._close_lock:
            if self._closed:
                return
            async with self._replace_lock:
                self._closing = True
                await self._cancel_refresh()
                current = self._current
                if not current.retired:
                    current.retired = True
                    self._retired[current.generation_id] = current
                    self._trace_retired(current, reason="shutdown")
                generations = tuple(self._retired.values())

            await asyncio.gather(
                *(generation.drained.wait() for generation in generations)
            )
            generation_results = await asyncio.gather(
                *(
                    self._close_generation(generation, forced=False)
                    for generation in generations
                )
            )
            unpublished_closed = await self._retry_unpublished_cleanup()
            if not all(generation_results) or not unpublished_closed:
                raise RuntimeError("One or more provider runtimes failed to close.")
            self._model_cache.clear()
            self._closed = True

    async def _release(self, generation: _ProviderGeneration) -> None:
        if generation.active_leases <= 0:
            return
        generation.active_leases -= 1
        if generation.active_leases != 0:
            return
        generation.drained.set()
        if generation.retired and not self._closing:
            await self._close_generation(generation, forced=False)

    async def _refresh_generation(
        self,
        generation: _ProviderGeneration,
        *,
        only_missing: bool,
    ) -> ProviderModelRefreshResult:
        if generation.closed:
            return ProviderModelRefreshResult()
        generation.active_leases += 1
        generation.drained.clear()
        try:
            discovery = ProviderModelDiscovery(
                generation.settings,
                generation.runtime.resolve_provider,
                self._model_cache,
            )
            return await discovery.refresh_model_list_cache(only_missing=only_missing)
        finally:
            await self._release(generation)

    async def _refresh_generation_in_background(
        self,
        generation: _ProviderGeneration,
        *,
        only_missing: bool,
    ) -> None:
        try:
            await self._refresh_generation(generation, only_missing=only_missing)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Provider model discovery task failed: exc_type={}",
                type(exc).__name__,
            )

    async def _cancel_refresh(self) -> None:
        task = self._refresh_task
        self._refresh_task = None
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _cleanup_unpublished(self, runtime: ProviderRuntime) -> bool:
        self._unpublished.add(runtime)
        try:
            await runtime.cleanup()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Unpublished provider generation cleanup failed: exc_type={}",
                type(exc).__name__,
            )
            return False
        self._unpublished.discard(runtime)
        return True

    async def _retry_unpublished_cleanup(self) -> bool:
        all_closed = True
        for runtime in tuple(self._unpublished):
            if not await self._cleanup_unpublished(runtime):
                all_closed = False
        return all_closed

    async def _close_generation(
        self,
        generation: _ProviderGeneration,
        *,
        forced: bool,
    ) -> bool:
        if generation.closed:
            return True
        if generation.active_leases != 0:
            return False
        task = generation.cleanup_task
        if task is None:
            task = asyncio.create_task(
                self._run_generation_cleanup(generation, forced=forced),
                name=f"provider-generation-cleanup-{generation.generation_id}",
            )
            generation.cleanup_task = task
        return await asyncio.shield(task)

    async def _run_generation_cleanup(
        self,
        generation: _ProviderGeneration,
        *,
        forced: bool,
    ) -> bool:
        task = asyncio.current_task()
        try:
            try:
                await generation.runtime.cleanup()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Provider generation cleanup failed: generation_id={} exc_type={}",
                    generation.generation_id,
                    type(exc).__name__,
                )
                return False

            generation.closed = True
            self._retired.pop(generation.generation_id, None)
            trace_event(
                stage="runtime",
                event="provider_generation.closed",
                source="runtime",
                generation_id=generation.generation_id,
                active_leases=generation.active_leases,
                forced=forced,
                outcome="ok",
            )
            return True
        finally:
            if not generation.closed and generation.cleanup_task is task:
                generation.cleanup_task = None

    @staticmethod
    def _trace_published(
        generation: _ProviderGeneration,
        *,
        previous: _ProviderGeneration | None,
        reason: str,
    ) -> None:
        trace_event(
            stage="runtime",
            event="provider_generation.published",
            source="runtime",
            generation_id=generation.generation_id,
            previous_generation_id=(
                previous.generation_id if previous is not None else None
            ),
            reason=reason,
        )

    @staticmethod
    def _trace_retired(generation: _ProviderGeneration, *, reason: str) -> None:
        trace_event(
            stage="runtime",
            event="provider_generation.retired",
            source="runtime",
            generation_id=generation.generation_id,
            active_leases=generation.active_leases,
            reason=reason,
        )
