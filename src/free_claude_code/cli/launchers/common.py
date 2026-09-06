"""Shared process helpers for installed client CLI launchers."""

import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request

from free_claude_code.cli.local_http import open_local_request
from free_claude_code.cli.process_registry import (
    kill_pid_tree_best_effort,
    register_pid,
    unregister_pid,
)
from free_claude_code.config.paths import config_dir_path, server_log_path
from free_claude_code.core.interprocess_lock import InterprocessFileLock

PROXY_PREFLIGHT_PATH = "/health"
PROXY_PREFLIGHT_TIMEOUT_SECONDS = 1.5
# Maximum time to wait for server startup (seconds)
_SERVER_START_TIMEOUT_SECONDS = 30.0
# Interval between health checks during server startup (seconds)
_SERVER_START_POLL_INTERVAL_SECONDS = 0.5


def _startup_lock_path_for(proxy_root_url: str) -> Path:
    """Return the interprocess startup lock path scoped to one server endpoint.

    Two launchers targeting different local proxies (for example different
    ``settings.port`` values) do not block each other's startup.
    """
    normalized = proxy_root_url.rstrip("/").lower()
    safe = normalized.replace(":", "_").replace("/", "_").replace("..", "_")
    return config_dir_path() / f"{safe}.server.startup.lock"


def proxy_v1_url(proxy_root_url: str) -> str:
    """Return the canonical local proxy API root for client launchers."""

    stripped = proxy_root_url.rstrip("/")
    return stripped if stripped.endswith("/v1") else f"{stripped}/v1"


def preflight_proxy(proxy_root_url: str) -> str | None:
    """Return an error message when the local proxy health check is unreachable."""

    url = f"{proxy_root_url.rstrip('/')}{PROXY_PREFLIGHT_PATH}"
    request = Request(url, method="GET")
    try:
        with open_local_request(
            request, timeout=PROXY_PREFLIGHT_TIMEOUT_SECONDS
        ) as response:
            status_code = response.status
    except HTTPError as exc:
        return f"returned HTTP {exc.code}"
    except URLError as exc:
        return str(exc.reason)
    except OSError as exc:
        return str(exc)

    if not 200 <= status_code < 300:
        return f"returned HTTP {status_code}"
    return None


def resolve_client_binary(
    *,
    binary_name: str,
    display_name: str,
    install_hint: str,
) -> str:
    """Resolve an installed client binary or exit with a user-facing hint."""

    client_command = shutil.which(binary_name)
    if client_command is None:
        print(
            f"Could not find {display_name} command: {binary_name}",
            file=sys.stderr,
        )
        print(install_hint, file=sys.stderr)
        raise SystemExit(127)
    return client_command


def run_client_process(
    *,
    command: list[str],
    env: Mapping[str, str],
    binary_name: str,
    display_name: str,
    install_hint: str,
) -> None:
    """Run a client CLI command and mirror its exit code."""

    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(command, env=dict(env))
        if process.pid:
            register_pid(process.pid)
        return_code = process.wait()
    except FileNotFoundError:
        print(
            f"Could not find {display_name} command: {binary_name}",
            file=sys.stderr,
        )
        print(install_hint, file=sys.stderr)
        raise SystemExit(127) from None
    except KeyboardInterrupt:
        if process is not None and process.pid:
            kill_pid_tree_best_effort(process.pid)
            process.wait()
        raise
    finally:
        if process is not None and process.pid:
            unregister_pid(process.pid)

    raise SystemExit(return_code)


def _ensure_fcc_server_running(proxy_root_url: str) -> bool:
    """Return True if the FCC server is reachable (starting it if needed)."""
    # Fast path: if already reachable, do nothing.
    if preflight_proxy(proxy_root_url) is None:
        return True

    lock_path = _startup_lock_path_for(proxy_root_url)
    lock = InterprocessFileLock(lock_path)
    deadline = time.monotonic() + _SERVER_START_TIMEOUT_SECONDS

    # Become the designated starter: hold the lock until the server is ready
    # (or startup definitively fails) so peers for this endpoint never spawn a
    # second server.
    if lock.acquire(wait=False):
        try:
            return _start_server_and_wait_until_ready(proxy_root_url, deadline)
        finally:
            lock.release()

    # Another launcher owns startup for this endpoint. Wait for its server to
    # become ready and, if that starter gives up or dies, take over the startup
    # role ourselves.
    while True:
        if preflight_proxy(proxy_root_url) is None:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_SERVER_START_POLL_INTERVAL_SECONDS)
        if lock.acquire(wait=False):
            try:
                if time.monotonic() >= deadline:
                    return False
                return _start_server_and_wait_until_ready(proxy_root_url, deadline)
            finally:
                lock.release()


def _start_server_and_wait_until_ready(proxy_root_url: str, deadline: float) -> bool:
    """Start the server if needed and return True once it is reachable."""

    # Re-check because another process may have started the server while we
    # waited for the lock.
    if preflight_proxy(proxy_root_url) is None:
        return True
    if not _spawn_server_process():
        return False
    return _wait_for_server_ready(proxy_root_url, deadline)


def _spawn_server_process() -> bool:
    """Spawn one detached fcc-server subprocess.

    Return False when the subprocess cannot be created (for example a
    read-only or full configuration directory) so callers surface the normal
    proxy-unreachable message instead of a traceback.
    """
    try:
        log_path = server_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", buffering=1) as log_file:
            subprocess.Popen(
                [sys.executable, "-m", "free_claude_code.cli.entrypoints", "serve"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError:
        return False
    return True


def _wait_for_server_ready(proxy_root_url: str, deadline: float) -> bool:
    """Poll the health endpoint until the server is ready or the deadline passes."""
    while time.monotonic() < deadline:
        if preflight_proxy(proxy_root_url) is None:
            return True
        time.sleep(_SERVER_START_POLL_INTERVAL_SECONDS)
    return False
