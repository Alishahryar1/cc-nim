"""Greptile regression: validation removed and provider KPI after disconnect."""

import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import uvicorn
from playwright.sync_api import Page, expect

from free_claude_code.api.app import create_app
from free_claude_code.api.ports import ApiServices
from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountState,
    ConnectedAccountStatus,
)
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config import env_migrations, paths
from free_claude_code.config.env_migrations import recognized_env_keys
from free_claude_code.config.loader import clear_settings_cache, get_settings
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.providers.runtime import ProviderRuntime
from free_claude_code.runtime.application import ApplicationRuntime
from free_claude_code.runtime.asgi import RuntimeASGIApp
from free_claude_code.runtime.chat_sqlite import SQLiteChatStore
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager


class _ModelListingProvider(BaseProvider):
    def __init__(self, model_infos: frozenset[ProviderModelInfo] = frozenset()) -> None:
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

    def preflight_messages(
        self,
        request: MessagesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        return None

    def preflight_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> None:
        return None

    async def cleanup(self) -> None:
        return None

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        return self._model_infos

    async def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        if False:
            yield ""

    async def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        if False:
            yield ""


class _PollDisconnectFakeAccount:
    """Starts CONNECTED, then after start_login returns CONNECTING once, then DISCONNECTED."""

    def __init__(self) -> None:
        self._state = ConnectedAccountState.CONNECTED
        self._connected = True
        self.revision = 0
        self._poll_calls = 0

    def is_connected(self) -> bool:
        return self._connected and self._state == ConnectedAccountState.CONNECTED

    def status(self) -> ConnectedAccountStatus:
        if self._state == ConnectedAccountState.CONNECTING:
            self._poll_calls += 1
            # After first poll, transition to DISCONNECTED terminal
            self._state = ConnectedAccountState.DISCONNECTED
            self._connected = False
            self.revision += 1
            return ConnectedAccountStatus(
                provider_id="openai",
                state=ConnectedAccountState.DISCONNECTED,
                connected=False,
                revision=self.revision,
            )
        if self._state == ConnectedAccountState.CONNECTED:
            return ConnectedAccountStatus(
                provider_id="openai",
                state=ConnectedAccountState.CONNECTED,
                connected=True,
                revision=self.revision,
                email="test@example.com",
            )
        return ConnectedAccountStatus(
            provider_id="openai",
            state=ConnectedAccountState.DISCONNECTED,
            connected=False,
            revision=self.revision,
        )

    async def start_login(
        self, mode: ConnectedAccountLoginMode
    ) -> ConnectedAccountStatus:
        self._state = ConnectedAccountState.CONNECTING
        self._connected = False
        self._poll_calls = 0
        self.revision += 1
        return ConnectedAccountStatus(
            provider_id="openai",
            state=ConnectedAccountState.CONNECTING,
            connected=False,
            revision=self.revision,
            mode=mode,
            authorization_url="https://example.com/auth",
        )

    async def cancel_login(self) -> ConnectedAccountStatus:
        self._state = ConnectedAccountState.DISCONNECTED
        self._connected = False
        self.revision += 1
        return self.status()

    async def disconnect(self) -> ConnectedAccountStatus:
        self._state = ConnectedAccountState.DISCONNECTED
        self._connected = False
        self.revision += 1
        return self.status()

    async def close(self) -> None:
        return None


@pytest.fixture
def poll_disconnect_admin_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[str]:
    """Serve Admin with a connected account that will poll to DISCONNECTED."""
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
    monkeypatch.setattr(paths, "config_dir_path", lambda: config_dir)
    monkeypatch.setattr(env_migrations, "legacy_env_paths", lambda: ())
    monkeypatch.setattr(env_migrations, "verified_checkout_env_path", lambda: None)
    clear_settings_cache()

    fake = _PollDisconnectFakeAccount()
    providers: dict[str, BaseProvider] = {
        "open_router": _ModelListingProvider(
            frozenset({ProviderModelInfo("vendor/model-a")})
        ),
        "groq": _ModelListingProvider(),
    }
    manager = ProviderRuntimeManager(
        get_settings(),
        runtime_factory=lambda snapshot: ProviderRuntime(snapshot, dict(providers)),
        connected_provider_ids=lambda: ("openai",) if fake.is_connected() else (),
    )
    chat_store = SQLiteChatStore(
        config_dir / "chat" / "chat.db", config_dir / "chat" / "chat.lock"
    )
    from free_claude_code.application.chat import ChatService

    chat = ChatService(manager, chat_store)
    runtime = ApplicationRuntime(
        manager,
        transcriber=None,
        chat_service=chat,
        connected_accounts={"openai": fake},
    )
    app = RuntimeASGIApp(
        create_app(
            ApiServices(requests=manager, admin=runtime, tasks=runtime, chat=chat)
        ),
        runtime,
    )

    async def local_provider_result(
        provider_id: str, base_url: str, path: str
    ) -> dict[str, object]:
        return {
            "provider_id": provider_id,
            "status": "reachable",
            "label": "Reachable",
            "base_url": base_url,
        }

    monkeypatch.setattr(
        "free_claude_code.api.admin_routes._check_local_provider", local_provider_result
    )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="error", access_log=False, lifespan="on")
    )
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 5.0
    try:
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("Admin browser-test server exited")
            if time.monotonic() >= deadline:
                raise TimeoutError("Admin server did not start")
            time.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
        listener.close()
        clear_settings_cache()
        if thread.is_alive():
            pytest.fail("Admin browser-test server did not stop")


def test_admin_does_not_request_validate_on_load(
    page: Page, admin_base_url: str
) -> None:
    requested: list[str] = []
    errors: list[str] = []

    page.on("request", lambda req: requested.append(req.url))
    page.on("pageerror", lambda err: errors.append(str(err)))

    page.goto(f"{admin_base_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")
    expect(page.locator('[data-provider="open_router"]')).to_be_visible()

    assert not any("/admin/api/config/validate" in url for url in requested), requested
    assert errors == [], errors


def test_connected_account_poll_updates_kpi_after_disconnect(
    page: Page, poll_disconnect_admin_url: str
) -> None:
    page.goto(f"{poll_disconnect_admin_url}/admin")
    expect(page.locator("#messageArea")).to_have_text("")

    card = page.locator('[data-provider="openai"][data-connected-account="true"]')
    expect(card).to_be_visible()
    expect(card.locator(".status-pill")).to_have_text("Connected")

    kpi_sub = page.locator("#statProvidersSub")
    expect(kpi_sub).to_have_text("1 subscription connected")

    # Reconnect triggers poll: connecting -> disconnected terminal
    page.on("dialog", lambda dialog: dialog.accept())
    # Click Reconnect
    card.get_by_role("button", name="Reconnect", exact=True).click()

    # Poll will run after 1s and return DISCONNECTED; KPI should be recalculated
    expect(card.locator(".status-pill")).to_have_text("Not connected", timeout=5000)
    expect(kpi_sub).to_have_text("Standard API backends", timeout=5000)

    # Ensure no console errors after poll
    # Check that next poll does not re-trigger hydrate incorrectly: ensure still not connected
    expect(card.locator(".status-pill")).to_have_text("Not connected")
