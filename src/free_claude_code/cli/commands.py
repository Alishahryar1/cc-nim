"""Implementations for installed Free Claude Code commands."""

import threading
import time
import webbrowser
from enum import StrEnum

import uvicorn
from loguru import logger

from free_claude_code.cli.claude_desktop import configure_claude_desktop_config
from free_claude_code.cli.launchers.common import preflight_proxy
from free_claude_code.cli.process_registry import kill_all_best_effort
from free_claude_code.cli.tls_proxy import (
    CaddyTlsProxy,
    desktop_gateway_base_url,
    load_or_create_front_identity,
    probe_fcc_front,
    tls_root_url,
)
from free_claude_code.config.loader import (
    clear_settings_cache,
    get_settings,
    repair_invalid_managed_provider_proxies,
)
from free_claude_code.config.paths import managed_env_path
from free_claude_code.config.server_urls import local_admin_url, local_proxy_root_url
from free_claude_code.config.settings import Settings
from free_claude_code.runtime.bootstrap import build_asgi_app

SERVER_GRACEFUL_SHUTDOWN_SECONDS = 5
GATEWAY_HEALTH_UPGRADE_SECONDS = 10.0


def serve() -> None:
    """Start and supervise the FastAPI server."""
    ServerSupervisor().run()


class ServerStatus(StrEnum):
    """Observable state of the server owned by a supervisor."""

    STARTING = "Starting"
    RUNNING = "Running"
    STOPPING = "Stopping"
    STOPPED = "Stopped"


class ServerSupervisor:
    """Own one FCC server lifecycle, including config-driven restarts."""

    def __init__(self, *, console_logging: bool = True) -> None:
        self._console_logging = console_logging
        self._lock = threading.Lock()
        self._server: uvicorn.Server | None = None
        self._desktop_gateway_url: str | None = None
        self._run_scheduled = False
        self._running = False
        self._stop_requested = False
        self._restart_generation = 0

    @property
    def status(self) -> ServerStatus:
        with self._lock:
            if self._run_scheduled:
                return ServerStatus.STARTING
            if not self._running:
                return ServerStatus.STOPPED
            if self._server is None:
                return ServerStatus.STARTING
            if self._server.should_exit:
                return ServerStatus.STOPPING
            if self._server.started:
                return ServerStatus.RUNNING
            return ServerStatus.STARTING

    def desktop_gateway_url(self) -> str | None:
        """The desktop-scoped gateway URL the active generation is serving.

        Resolved once the generation's HTTPS front has been started (or the
        plain-HTTP fallback chosen), so Claude Desktop integrations read the
        live endpoint rather than a stale or not-yet-ready value. ``None``
        before the first generation publishes one.
        """

        with self._lock:
            return self._desktop_gateway_url

    def schedule_run(self) -> bool:
        """Reserve a worker run before its thread starts."""

        with self._lock:
            if self._stop_requested or self._run_scheduled or self._running:
                return False
            self._run_scheduled = True
            return True

    def run(self, *, open_admin_browser: bool | None = None) -> None:
        """Block until stopped, applying only fully closed Admin restarts."""

        with self._lock:
            self._run_scheduled = False
            if self._running:
                raise RuntimeError("The FCC server supervisor is already running.")
            if self._stop_requested:
                return
            self._running = True

        opened_admin_browser = False
        try:
            try:
                while not self._is_stop_requested():
                    with self._lock:
                        restart_generation = self._restart_generation
                    settings = load_server_settings()
                    should_open_admin = (
                        settings.open_admin_browser
                        if open_admin_browser is None
                        else open_admin_browser
                    ) and not opened_admin_browser
                    if not self._run_once(
                        settings,
                        open_admin_browser=should_open_admin,
                        restart_generation=restart_generation,
                    ):
                        return
                    opened_admin_browser = opened_admin_browser or should_open_admin
                    clear_settings_cache()
            except KeyboardInterrupt:
                return
        finally:
            with self._lock:
                self._server = None
                self._desktop_gateway_url = None
                self._running = False
            kill_all_best_effort()

    def request_restart(self) -> bool:
        """Reload an active generation or coalesce into a scheduled fresh run."""

        with self._lock:
            if self._stop_requested:
                return False
            if self._run_scheduled:
                self._restart_generation += 1
                return True
            if not self._running:
                return False
            self._restart_generation += 1
            if self._server is not None:
                self._server.should_exit = True
            return True

    def request_stop(self) -> None:
        """Permanently stop this supervisor after graceful runtime cleanup."""

        with self._lock:
            self._stop_requested = True
            self._run_scheduled = False
            if self._server is not None:
                self._server.should_exit = True

    def _is_stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested

    def _run_once(
        self,
        settings: Settings,
        *,
        open_admin_browser: bool,
        restart_generation: int,
    ) -> bool:
        asgi_app = build_asgi_app(
            settings,
            restart_callback=self._request_runtime_restart,
        )
        config = uvicorn.Config(
            asgi_app,
            host=settings.host,
            port=settings.port,
            log_level="debug",
            log_config=(
                uvicorn.config.LOGGING_CONFIG if self._console_logging else None
            ),
            timeout_graceful_shutdown=SERVER_GRACEFUL_SHUTDOWN_SECONDS,
        )
        server = uvicorn.Server(config)
        with self._lock:
            self._server = server
            if self._stop_requested or self._restart_generation != restart_generation:
                server.should_exit = True

        # Own the managed HTTPS front for exactly this generation: started
        # before the server accepts traffic so Claude Desktop's config-merge
        # probe finds a live TLS endpoint, stopped when the generation ends.
        # ``start()`` never raises and falls back to plain HTTP on failure.
        tls_front = CaddyTlsProxy(settings)
        readiness: threading.Thread | None = None
        try:
            tls_front.start()
            # Publish the desktop-scoped gateway URL this generation serves.
            # The plain-HTTP fallback is correct until the front verifies,
            # and verification needs FCC's /health marker THROUGH the front,
            # which only answers while Uvicorn serves — so a concurrent
            # readiness task probes during ``server.run()`` and swaps in the
            # TLS-prefixed URL the moment the front verifies, keeping the
            # HTTPS URL published for the whole live serving window.
            self._publish_desktop_gateway_url(settings)
            # Run the readiness whenever a front MAY own the TLS port, not
            # only when managed startup succeeded: an external reverse proxy
            # already bound there fails the pre-Uvicorn probe and makes the
            # managed caddy exit on the occupied port (``start()`` -> False),
            # yet it still serves FCC's /health once Uvicorn is up. Gating on
            # ``start()`` would strand such a front on the HTTP fallback.
            if settings.tls_proxy_enabled:
                readiness = self._start_gateway_https_readiness(settings)
            if open_admin_browser:
                schedule_open_admin_browser(settings)
            server.run()
        finally:
            # Join the readiness task before tearing down the front so the
            # published HTTPS URL stays live until the generation actually
            # stops and no probe outlives the generation's front.
            if readiness is not None:
                readiness.join(timeout=GATEWAY_HEALTH_UPGRADE_SECONDS + 2.0)
            tls_front.stop()

        with self._lock:
            if self._server is server:
                self._server = None
            restart_requested = self._restart_generation != restart_generation
            stop_requested = self._stop_requested
        return restart_requested and not stop_requested and asgi_app.runtime.is_closed

    def _publish_desktop_gateway_url(self, settings: Settings) -> None:
        """Publish the resolved desktop gateway URL for this generation."""

        gateway_url = desktop_gateway_base_url(settings)
        with self._lock:
            self._desktop_gateway_url = gateway_url
        logger.info("Claude Desktop gateway: {}", gateway_url)

    def _start_gateway_https_readiness(self, settings: Settings) -> threading.Thread:
        """Spawn the readiness task that upgrades the published URL to HTTPS.

        The front can only pass the adoption probe while Uvicorn serves, so
        the probe runs concurrently with ``server.run()`` and the
        TLS-prefixed URL is published during the live serving window — not
        after it ends. The task is joined before the generation's front is
        stopped, so the HTTPS URL stays published until the generation
        actually stops.
        """

        readiness = threading.Thread(
            target=self._await_gateway_https_readiness,
            args=(settings,),
            name="fcc-gateway-https-readiness",
            daemon=True,
        )
        readiness.start()
        return readiness

    def _await_gateway_https_readiness(self, settings: Settings) -> None:
        """Publish the TLS-prefixed gateway URL once the front verifies.

        The upgrade is retryable for the whole readiness window: the
        persisted-config rewrite and the in-memory publication commit
        together, so a transient rewrite failure keeps both surfaces on
        the plain-HTTP fallback and the next probe reattempts the pair
        rather than leaving them split across two URLs.
        """

        root = tls_root_url(settings)
        identity = load_or_create_front_identity()
        deadline = time.monotonic() + GATEWAY_HEALTH_UPGRADE_SECONDS
        while time.monotonic() < deadline:
            verified = probe_fcc_front(root, identity)
            if verified and self._publish_verified_https_gateway_url(settings, root):
                return
            # A probe that verifies the front but fails the persisted
            # config rewrite keeps both surfaces on the consistent
            # plain-HTTP fallback; fall through to the sleep and retry the
            # whole upgrade on the next probe — the readiness window is
            # the only time the write and the publication can be made
            # atomic together.
            time.sleep(0.25)

    def _publish_verified_https_gateway_url(
        self, settings: Settings, root: str
    ) -> bool:
        """Publish the TLS-prefixed URL for a front that just verified.

        Publishes the verified root directly — re-resolving would probe the
        front a second time for the same answer. The persisted Claude
        Desktop config is re-merged with the verified URL BEFORE the
        in-memory value swaps: the pre-lifecycle merge recorded the
        plain-HTTP fallback while the front was still starting, Claude
        Desktop reads the config file (not this process's memory), so an
        in-memory upgrade the file never received would advertise an HTTPS
        endpoint Claude Desktop cannot use. Persist-first also makes the
        write failure the caller's signal to keep probing: a transient
        ``OSError`` (disk full, EBUSY rename) leaves both surfaces on the
        consistent plain-HTTP fallback instead of splitting them, and the
        readiness loop retries the whole upgrade on its next probe.

        Returns whether the HTTPS URL was published — ``False`` means the
        persisted config could not be rewritten and the caller should
        retry.
        """

        gateway_url = desktop_gateway_base_url(settings, base_url=root)
        if not self._repersist_verified_gateway_url(settings, gateway_url):
            return False
        with self._lock:
            self._desktop_gateway_url = gateway_url
        logger.info("Claude Desktop gateway: {}", gateway_url)
        return True

    def _repersist_verified_gateway_url(
        self, settings: Settings, gateway_url: str
    ) -> bool:
        """Rewrite the Claude Desktop routing block onto the verified front.

        Runs once per verified readiness upgrade attempt, inside the live
        serving window. The persisted URL is the desktop-scoped one — the
        same value published in memory — because Claude Desktop discovers
        models against the prefix mount that serves picker aliases, not
        the bare root. Failures downgrade to a warning and return
        ``False``: the in-memory publication is deferred until the write
        lands, so a config-merge failure must not take the serving
        generation down and must not strand the two surfaces on different
        URLs either.
        """

        try:
            merged = configure_claude_desktop_config(
                settings=settings,
                gateway_base_url=gateway_url,
            )
        except OSError as exc:
            logger.warning("Could not re-merge the Claude Desktop config: {}", exc)
            return False
        logger.info(
            "Claude Desktop routing re-merged onto the verified HTTPS front{}.",
            "" if merged else " (no change needed)",
        )
        return True

    def _request_runtime_restart(self) -> None:
        self.request_restart()


def load_server_settings() -> Settings:
    """Return canonical settings after repairing invalid managed proxies."""

    settings = get_settings()
    removed = repair_invalid_managed_provider_proxies()
    if not removed:
        return settings

    logger.warning(
        "Removed invalid managed provider proxy settings from {}: {}. "
        "Configure valid proxy URLs in Admin if needed.",
        managed_env_path(),
        ", ".join(removed),
    )
    clear_settings_cache()
    return get_settings()


def open_admin_when_ready(settings: Settings) -> bool:
    """Wait briefly for /health, then open the current Admin UI."""

    admin_url = local_admin_url(settings)
    proxy_root_url = local_proxy_root_url(settings)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if preflight_proxy(proxy_root_url) is None:
            return webbrowser.open(admin_url)
        time.sleep(0.15)
    return False


def schedule_open_admin_browser(settings: Settings) -> None:
    """Open Admin after health succeeds without blocking the caller."""

    threading.Thread(
        target=open_admin_when_ready,
        args=(settings,),
        name="fcc-open-admin-browser",
        daemon=True,
    ).start()
