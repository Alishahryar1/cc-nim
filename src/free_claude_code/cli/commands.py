"""Implementations for installed Free Claude Code commands."""

import os
import shutil
import sys
import threading
import time
import webbrowser
from enum import StrEnum
from pathlib import Path

import uvicorn

from free_claude_code.cli.launchers.common import preflight_proxy
from free_claude_code.cli.process_registry import kill_all_best_effort
from free_claude_code.config.env_migrations import (
    explicit_env_file_migration_warning,
    migrate_owned_env_files,
)
from free_claude_code.config.paths import (
    legacy_env_paths,
    managed_env_path,
)
from free_claude_code.config.server_urls import local_admin_url, local_proxy_root_url
from free_claude_code.config.settings import Settings, get_settings
from free_claude_code.runtime.bootstrap import build_asgi_app

SERVER_GRACEFUL_SHUTDOWN_SECONDS = 5


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
                    get_settings.cache_clear()
            except KeyboardInterrupt:
                return
        finally:
            with self._lock:
                self._server = None
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

        if open_admin_browser:
            schedule_open_admin_browser(settings)
        server.run()

        with self._lock:
            if self._server is server:
                self._server = None
            restart_requested = self._restart_generation != restart_generation
            stop_requested = self._stop_requested
        return restart_requested and not stop_requested and asgi_app.runtime.is_closed

    def _request_runtime_restart(self) -> None:
        self.request_restart()


def load_server_settings() -> Settings:
    """Apply owned config migrations before returning the cached settings."""

    _migrate_legacy_env_if_missing()
    _migrate_config_env_keys()
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


def _migrate_legacy_env_if_missing() -> Path | None:
    """Copy a legacy user env into the managed config path when absent."""

    env_file = managed_env_path()
    if env_file.exists():
        return None

    # TODO: Remove after the ~/.fcc/.env migration has had a release cycle.
    for legacy_env in legacy_env_paths():
        if not legacy_env.is_file():
            continue
        env_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(legacy_env, env_file)
        return legacy_env

    return None


def _migrate_config_env_keys() -> tuple[Path, ...]:
    """Apply dotenv key migrations before Settings loads config."""

    migrated = migrate_owned_env_files()
    if warning := explicit_env_file_migration_warning(os.environ):
        print(warning, file=sys.stderr)
    return migrated
