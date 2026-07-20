"""Direct HTTP and child-environment policy for FCC-local traffic."""

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

_DIRECT_OPENER = build_opener(ProxyHandler({}))
_LOOPBACK_BYPASS_HOSTS = ("127.0.0.1", "localhost", "::1")
_NO_PROXY_KEYS = ("NO_PROXY", "no_proxy")


def open_local_request(request: Request, *, timeout: float) -> Any:
    """Open an FCC-local request without consulting machine proxy settings."""

    return _DIRECT_OPENER.open(request, timeout=timeout)


def with_local_proxy_bypass(
    base_env: Mapping[str, str],
    *,
    proxy_root_url: str,
) -> dict[str, str]:
    """Copy an environment and keep its FCC-local destination off proxies."""

    host = urlsplit(proxy_root_url).hostname
    if host is None:
        raise ValueError("Local proxy root URL must include a host.")

    env = dict(base_env)
    entries: list[str] = []
    seen: set[str] = set()
    for key in _NO_PROXY_KEYS:
        for raw_entry in base_env.get(key, "").split(","):
            _append_unique(entries, seen, raw_entry)
    for local_host in (*_LOOPBACK_BYPASS_HOSTS, host):
        _append_unique(entries, seen, local_host)

    value = ",".join(entries)
    for key in _NO_PROXY_KEYS:
        env[key] = value
    return env


def _append_unique(entries: list[str], seen: set[str], raw_entry: str) -> None:
    entry = raw_entry.strip()
    normalized = entry.casefold()
    if not entry or normalized in seen:
        return
    entries.append(entry)
    seen.add(normalized)
