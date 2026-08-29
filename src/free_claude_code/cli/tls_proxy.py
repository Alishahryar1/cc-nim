"""HTTPS front proxy for clients that refuse plain HTTP (e.g. Claude Desktop).

Two cooperating pieces:

1. ``probe_fcc_front`` / ``resolve_gateway_base_url`` — detect whether an
   HTTPS endpoint that fronts *this* FCC server already answers on the
   configured TLS port (verified via FCC's unauthenticated ``/health``
   marker, since an adopted front also receives clients' API keys). Any
   working external front is reused as-is; FCC never touches it.
2. ``CaddyTlsProxy`` — when nothing answers and a ``caddy`` binary exists,
   spawn an FCC-managed instance: generated Caddyfile under ``~/.fcc/caddy/``
   with ``tls internal``, admin API disabled, and all of Caddy's state
   sandboxed inside that directory. The proxy forwards to the local FCC
   server and is stopped alongside it.
"""

import http.client
import json
import os
import shutil
import ssl
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from loguru import logger

from free_claude_code.config import paths
from free_claude_code.config.server_urls import (
    _browser_host_for_local_urls,
    local_proxy_root_url,
)
from free_claude_code.config.settings import Settings

CADDY_BINARY = "caddy"
CADDY_HOME_DIRNAME = "caddy"
READY_TIMEOUT_SECONDS = 10.0
FCC_HEALTH_MARKER = {"status": "healthy"}


def _unverified_context() -> ssl.SSLContext:
    """TLS context accepting self-signed local certificates."""

    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def probe_https(url: str, timeout_seconds: float = 1.0) -> bool:
    """Whether ``url`` answers over TLS with any HTTP status.

    Certificate errors are expected (self-signed local CA) and ignored;
    only transport success counts.
    """

    try:
        with urllib.request.urlopen(
            url, timeout=timeout_seconds, context=_unverified_context()
        ):
            return True
    except urllib.error.HTTPError as exc:
        # Any HTTP status (404/401/502 ...) proves the TLS front is alive;
        # only transport-level failures mean it is not ready.
        exc.close()  # close the error body so no ResourceWarning fires at GC
        return True
    except urllib.error.URLError, OSError, ValueError:
        return False


def probe_fcc_front(base_url: str, timeout_seconds: float = 1.0) -> bool:
    """Whether ``base_url`` fronts *this* FCC server, not just any TLS listener.

    Adopting an external front is a trust decision: whatever answers also
    receives clients' API keys and inference traffic. A listener therefore only
    counts as an existing front when it serves FCC's unauthenticated
    ``/health`` marker through the TLS layer — which every reverse proxy in
    front of FCC passes through untouched.
    """

    health_url = f"{base_url.rstrip('/')}/health"
    request = urllib.request.Request(health_url)
    try:
        with urllib.request.urlopen(
            request, timeout=timeout_seconds, context=_unverified_context()
        ) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        exc.close()
        return False
    except (
        urllib.error.URLError,
        OSError,
        ValueError,
        http.client.HTTPException,
    ):
        return False
    try:
        payload = json.loads(body)
    except json.JSONDecodeError, UnicodeDecodeError:
        return False
    return (
        isinstance(payload, dict)
        and payload.get("status") == FCC_HEALTH_MARKER["status"]
    )


def tls_root_url(settings: Settings) -> str:
    """HTTPS URL Claude Desktop should use instead of the plain HTTP root.

    Always ``localhost``: the TLS front is local by definition, and the
    managed Caddy instance requests its internal certificate for that name.
    """

    return f"https://localhost:{settings.tls_proxy_port}"


def resolve_gateway_base_url(settings: Settings) -> str:
    """Best gateway base URL: HTTPS when something fronts the TLS port, else HTTP."""

    if settings.tls_proxy_enabled and probe_fcc_front(tls_root_url(settings)):
        return tls_root_url(settings)
    return local_proxy_root_url(settings)


def desktop_gateway_base_url(settings: Settings, base_url: str | None = None) -> str:
    """Gateway URL scoped to Claude Desktop: root plus the desktop path prefix.

    The prefix routes Claude Desktop onto the alias-emitting API mount while
    every other FCC client keeps hitting the bare paths, which always serve
    raw provider refs. ``base_url`` overrides resolution for callers that
    already know which root serves (e.g. a verified HTTPS front).
    """

    base = resolve_gateway_base_url(settings) if base_url is None else base_url
    return f"{base.rstrip('/')}/{settings.desktop_gateway_prefix}"


def _default_home_dir() -> Path:
    """Sandbox directory for the managed front's state and certificates.

    Routed through ``paths.config_dir_path()`` so the test-suite config-dir
    monkeypatch is honored; ``FCC_CONFIG_HOME`` stays a hard override.
    """

    override = os.environ.get("FCC_CONFIG_HOME")
    base = Path(override) if override else paths.config_dir_path()
    return base / CADDY_HOME_DIRNAME


class CaddyTlsProxy:
    """Lifecycle owner of one FCC-managed ``caddy run`` child process."""

    def __init__(self, settings: Settings, home_dir: Path | None = None) -> None:
        self._settings = settings
        self._home_dir = home_dir or _default_home_dir()
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def caddyfile_path(self) -> Path:
        return self._home_dir / "Caddyfile"

    def render_caddyfile(self) -> str:
        """Caddyfile text fronting the local FCC server with an internal cert."""

        upstream_host = _browser_host_for_local_urls(self._settings)
        if ":" in upstream_host:
            upstream_host = f"[{upstream_host}]"
        upstream = f"{upstream_host}:{self._settings.port}"
        return "\n".join(
            [
                "{",
                "    admin off",
                # The automatic HTTP->HTTPS redirect would try to bind :80,
                # which unprivileged users cannot do. Disabling just the
                # redirect keeps automatic certificate management alive.
                "    auto_https disable_redirects",
                "}",
                f"localhost:{self._settings.tls_proxy_port} {{",
                "    tls internal",
                f"    reverse_proxy {upstream} {{",
                "        flush_interval -1",
                "    }",
                "}",
                "",
            ]
        )

    def start(self) -> bool:
        """Start the managed proxy; returns whether HTTPS is now reachable.

        Reuses an existing external front — only after it proves it fronts
        this FCC server — without spawning anything. Never raises: failures
        are logged and reported as ``False`` so callers can fall back to
        plain HTTP.
        """

        if not self._settings.tls_proxy_enabled:
            return False
        target = tls_root_url(self._settings)
        if probe_fcc_front(target):
            logger.info("Reusing existing FCC HTTPS front at {}", target)
            return True

        binary = shutil.which(CADDY_BINARY)
        if binary is None:
            logger.warning(
                "No HTTPS front answered on {} and no caddy binary found; "
                "Claude Desktop routing stays on plain HTTP.",
                target,
            )
            return False

        try:
            self._home_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.caddyfile_path.write_text(self.render_caddyfile(), encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "Could not prepare the managed caddy state directory: {}; "
                "Claude Desktop routing stays on plain HTTP.",
                exc,
            )
            return False
        env = dict(os.environ)
        env["XDG_DATA_HOME"] = str(self._home_dir / "data")
        env["XDG_CONFIG_HOME"] = str(self._home_dir / "config")
        try:
            self._process = subprocess.Popen(
                [
                    binary,
                    "run",
                    "--config",
                    str(self.caddyfile_path),
                    "--adapter",
                    "caddyfile",
                ],
                cwd=self._home_dir,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            logger.warning("Could not start managed caddy: {}", exc)
            self._process = None
            return False

        deadline_ready = READY_TIMEOUT_SECONDS
        step = 0.25
        waited = 0.0
        while waited < deadline_ready:
            if self._process.poll() is not None:
                logger.warning(
                    "Managed caddy exited early with code {}; "
                    "Claude Desktop routing stays on plain HTTP.",
                    self._process.returncode,
                )
                self._process = None
                return False
            if probe_https(target, timeout_seconds=0.5):
                logger.info("Managed HTTPS front ready at {}", target)
                return True
            waited += step

        logger.warning(
            "Managed caddy did not become ready on {} in {}s; "
            "Claude Desktop routing stays on plain HTTP.",
            target,
            READY_TIMEOUT_SECONDS,
        )
        self.stop()
        return False

    def stop(self) -> None:
        """Terminate the managed child process, if we own one."""

        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
