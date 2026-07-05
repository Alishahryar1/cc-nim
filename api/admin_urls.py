"""Helpers for presenting local admin URLs."""

from config.settings import Settings


def _browser_host_for_local_urls(settings: Settings) -> str:
    """Host fragment for URLs shown to humans on the same machine as the server."""

    host = settings.host.strip() if settings.host else "127.0.0.1"
    if host in {"0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return host


def local_proxy_root_url(settings: Settings) -> str:
    """Return the proxy root URL (no path) for clients on the same machine."""

    return f"http://{_browser_host_for_local_urls(settings)}:{settings.port}"


def local_admin_url(settings: Settings) -> str:
    """Return a browser-friendly URL for the localhost-only admin UI."""

    return f"{local_proxy_root_url(settings)}/admin"


def admin_launch_message(settings: Settings) -> str:
    """Return the startup message shown by supported launch commands."""

    return f"Admin UI: {local_admin_url(settings)} (local-only)"
