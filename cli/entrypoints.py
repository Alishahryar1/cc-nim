"""CLI entry points for the installed package."""

from __future__ import annotations

import os
import shutil
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from api.admin_urls import local_admin_url, local_proxy_root_url
from api.app import GracefulLifespanApp, create_app
from cli.launchers.common import preflight_proxy
from cli.process_registry import (
    kill_all_best_effort,
)
from config.paths import (
    config_dir_path,
    legacy_env_paths,
    managed_env_path,
)
from config.settings import Settings, get_settings

SERVER_GRACEFUL_SHUTDOWN_SECONDS = 5


def _load_env_template() -> str:
    """Load the canonical root env template from package resources or source."""
    import importlib.resources

    packaged = importlib.resources.files("cli").joinpath("env.example")
    if packaged.is_file():
        return packaged.read_text("utf-8")

    source_template = Path(__file__).resolve().parents[1] / ".env.example"
    if source_template.is_file():
        return source_template.read_text(encoding="utf-8")

    raise FileNotFoundError("Could not find bundled or source .env.example template.")


def serve() -> None:
    """Start the FastAPI server (registered as `fcc-server` script)."""
    opened_admin_browser = False
    try:
        try:
            while True:
                _migrate_legacy_env_if_missing()
                settings = get_settings()
                if not _run_supervised_server(
                    settings, open_admin_browser=not opened_admin_browser
                ):
                    return
                opened_admin_browser = True
                get_settings.cache_clear()
        except KeyboardInterrupt:
            return
    finally:
        kill_all_best_effort()


def _admin_browser_open_enabled() -> bool:
    """Whether to open /admin when the server becomes reachable (FCC_OPEN_BROWSER)."""

    raw = os.environ.get("FCC_OPEN_BROWSER", "true").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _schedule_open_admin_browser(settings: Settings) -> None:
    """After /health succeeds, open the admin UI in the default browser (daemon thread)."""

    if not _admin_browser_open_enabled():
        return

    admin_url = local_admin_url(settings)
    proxy_root_url = local_proxy_root_url(settings)

    def open_when_ready() -> None:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if preflight_proxy(proxy_root_url) is None:
                webbrowser.open(admin_url)
                return
            time.sleep(0.15)

    threading.Thread(
        target=open_when_ready, name="fcc-open-admin-browser", daemon=True
    ).start()


def _run_supervised_server(settings: Settings, *, open_admin_browser: bool) -> bool:
    """Run one uvicorn server instance; return whether admin requested restart."""

    restart_requested = False
    server_holder: dict[str, uvicorn.Server] = {}

    def request_restart() -> None:
        nonlocal restart_requested
        restart_requested = True
        if server := server_holder.get("server"):
            server.should_exit = True

    app = create_app(lifespan_enabled=False)
    app.state.admin_restart_callback = request_restart
    asgi_app = GracefulLifespanApp(app)
    config = uvicorn.Config(
        asgi_app,
        host=settings.host,
        port=settings.port,
        log_level="debug",
        timeout_graceful_shutdown=SERVER_GRACEFUL_SHUTDOWN_SECONDS,
    )
    server = uvicorn.Server(config)
    server_holder["server"] = server
    if open_admin_browser:
        _schedule_open_admin_browser(settings)
    server.run()
    return restart_requested


def init() -> None:
    """Scaffold config at ~/.fcc/.env (registered as `fcc-init`)."""
    config_dir = config_dir_path()
    env_file = managed_env_path()

    migrated_from = _migrate_legacy_env_if_missing()
    if migrated_from is not None:
        print(f"Config migrated from {migrated_from} to {env_file}")
        print(
            "Edit it to set your API keys and model preferences, then run: fcc-server"
        )
        return

    if env_file.exists():
        print(f"Config already exists at {env_file}")
        print("Delete it first if you want to reset to defaults.")
        return

    config_dir.mkdir(parents=True, exist_ok=True)
    template = _load_env_template()
    env_file.write_text(template, encoding="utf-8")
    print(f"Config created at {env_file}")
    print("Edit it to set your API keys and model preferences, then run: fcc-server")


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
