"""Isolated browser-test composition for the local Admin UI."""

import asyncio
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import uvicorn

from free_claude_code.api.app import create_app
from free_claude_code.api.ports import ApiServices
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config import env_migrations, paths
from free_claude_code.config.env_migrations import recognized_env_keys
from free_claude_code.config.loader import clear_settings_cache, get_settings
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.inference import InferenceEvent, InferenceStreamLedger
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager


class _ModelListingProvider(BaseProvider):
    def __init__(
        self,
        model_infos: frozenset[ProviderModelInfo] = frozenset(),
        *,
        error: Exception | None = None,
    ) -> None:
        super().__init__(
            ProviderConfig(
                api_key="browser-test",
                base_url="https://provider.invalid/v1",
                rate_limit=1_000,
                rate_window=1,
                max_concurrency=100,
                http_read_timeout=1.0,
                http_write_timeout=1.0,
                http_connect_timeout=1.0,
                proxy=None,
                log_raw_sse_events=False,
                log_api_error_tracebacks=False,
            )
        )
        self._model_infos = model_infos
        self._error = error

    def preflight_stream(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        return None

    async def cleanup(self) -> None:
        return None

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        if self._error is not None:
            raise self._error
        return self._model_infos

    async def stream_response(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[InferenceEvent]:
        if False:
            yield InferenceStreamLedger("unused", "unused").start_response()


def _close_manager(manager: ProviderRuntimeManager) -> None:
    """Close async runtime ownership outside Playwright's active sync loop."""

    errors: list[BaseException] = []

    def close() -> None:
        try:
            asyncio.run(manager.close())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(
        target=close,
        name="fcc-admin-playwright-cleanup",
        daemon=True,
    )
    thread.start()
    thread.join(timeout=5.0)
    if thread.is_alive():
        pytest.fail("Admin browser-test runtime did not close")
    if errors:
        raise errors[0]


@pytest.fixture
def admin_base_url(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[str]:
    """Serve one fully isolated Admin application on an OS-assigned port."""

    config_dir = tmp_path / ".fcc"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for key in recognized_env_keys():
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("FCC_ENV_FILE", raising=False)
    monkeypatch.setenv("MODEL", "open_router/e2e-default")
    monkeypatch.setenv("OPENROUTER_API_KEY", "e2e-openrouter-key")
    monkeypatch.setenv("GROQ_API_KEY", "e2e-groq-key")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "e2e-cloudflare-token")
    monkeypatch.setenv("MESSAGING_PLATFORM", "none")
    monkeypatch.setenv("VOICE_NOTE_ENABLED", "false")
    monkeypatch.setenv("FCC_OPEN_BROWSER", "false")
    monkeypatch.setenv("PROXY_AUTH_ENABLED", "false")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "e2e-proxy-token")
    for key, value in getattr(request, "param", {}).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(paths, "config_dir_path", lambda: config_dir)
    monkeypatch.setattr(env_migrations, "legacy_env_paths", lambda: ())
    monkeypatch.setattr(env_migrations, "verified_checkout_env_path", lambda: None)
    clear_settings_cache()

    provider_secret = "CREDENTIAL[unrecognized-format-987654321]"
    providers: dict[str, BaseProvider] = {
        "open_router": _ModelListingProvider(
            frozenset(
                {
                    ProviderModelInfo("vendor/model-a"),
                    ProviderModelInfo("vendor/model-b"),
                }
            )
        ),
        "groq": _ModelListingProvider(
            error=RuntimeError(f"Provider rejected credential {provider_secret}")
        ),
    }
    manager = ProviderRuntimeManager(
        get_settings(),
        runtime_factory=lambda snapshot: ProviderRuntime(snapshot, dict(providers)),
    )
    runtime = ApplicationRuntime(manager, transcriber=None)
    app = create_app(
        ApiServices(
            requests=manager,
            admin=runtime,
            tasks=runtime,
        )
    )

    async def local_provider_result(
        provider_id: str,
        base_url: str,
        path: str,
    ) -> dict[str, object]:
        return {
            "provider_id": provider_id,
            "status": "reachable",
            "label": "Reachable",
            "base_url": base_url,
        }

    monkeypatch.setattr(
        "free_claude_code.api.admin_routes._check_local_provider",
        local_provider_result,
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            log_level="error",
            access_log=False,
            lifespan="off",
        )
    )
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        name="fcc-admin-playwright",
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + 5.0
    try:
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("Admin browser-test server exited during startup")
            if time.monotonic() >= deadline:
                raise TimeoutError("Admin browser-test server did not start")
            time.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
        listener.close()
        _close_manager(manager)
        clear_settings_cache()
        if thread.is_alive():
            pytest.fail("Admin browser-test server did not stop")
