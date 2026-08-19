"""Egress policy for user-controlled web_fetch URLs (SSRF guard)."""

import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

_DOMAIN_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")


@dataclass(frozen=True, slots=True)
class WebDomainRule:
    """One normalized Anthropic domain-filter entry."""

    host: str
    path: str = ""

    def matches(self, url: str) -> bool:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        if host != self.host and not host.endswith(f".{self.host}"):
            return False
        rule_path = self.path.rstrip("/") or "/"
        if rule_path == "/":
            return True
        path = parsed.path or "/"
        return path == rule_path or path.startswith(f"{rule_path}/")


@dataclass(frozen=True, slots=True)
class WebDomainPolicy:
    """Normalized allow/block policy for one local web server tool."""

    allowed: tuple[WebDomainRule, ...] = ()
    blocked: tuple[WebDomainRule, ...] = ()

    def allows(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return False
        if self.allowed and not any(rule.matches(url) for rule in self.allowed):
            return False
        return not any(rule.matches(url) for rule in self.blocked)

    def search_query(self, query: str) -> str:
        """Add recall hints while post-filtering remains authoritative."""
        allowed = " ".join(f"site:{rule.host}{rule.path}" for rule in self.allowed)
        blocked = " ".join(f"-site:{rule.host}{rule.path}" for rule in self.blocked)
        return " ".join(part for part in (query, allowed, blocked) if part)


def parse_web_domain_rule(raw: str, *, allow_path: bool) -> WebDomainRule:
    """Parse one documented domain-filter entry into a normalized value."""
    value = raw.strip()
    if not value:
        raise ValueError("domain entries must not be empty")
    if not value.isascii():
        raise ValueError("domain entries must contain ASCII characters only")
    if any(marker in value for marker in ("://", "@", ":", "?", "#", "*", "\\")):
        raise ValueError(
            "domain entries must not contain a scheme, user info, port, query, "
            "fragment, or wildcard"
        )

    host, separator, raw_path = value.partition("/")
    host = host.lower().rstrip(".")
    if (
        not host
        or len(host) > 253
        or any(not _DOMAIN_LABEL.fullmatch(label) for label in host.split("."))
    ):
        raise ValueError(f"invalid domain hostname: {raw!r}")
    if separator and not allow_path:
        raise ValueError("web_fetch domain filters must not contain a path")
    path = f"/{raw_path}" if separator else ""
    return WebDomainRule(host=host, path=path)


@dataclass(frozen=True, slots=True)
class WebFetchEgressPolicy:
    """Egress rules for user-influenced web_fetch URLs."""

    allow_private_network_targets: bool
    allowed_schemes: frozenset[str]
    domains: WebDomainPolicy = WebDomainPolicy()


class WebFetchEgressViolation(ValueError):
    """Raised when a web_fetch URL is rejected by egress policy (SSRF guard)."""


class WebFetchAccessError(WebFetchEgressViolation):
    """Raised when an allowed web_fetch target cannot be reached safely."""


def web_fetch_allowed_scheme_set(raw_schemes: str) -> frozenset[str]:
    """Return normalized schemes allowed for web_fetch."""

    return frozenset(
        part.strip().lower() for part in raw_schemes.split(",") if part.strip()
    )


def _port_for_url(parsed) -> int:
    try:
        explicit_port = parsed.port
    except ValueError as exc:
        raise WebFetchEgressViolation("web_fetch URL contains an invalid port") from exc
    if explicit_port is not None:
        return explicit_port
    return 443 if (parsed.scheme or "").lower() == "https" else 80


def _stream_getaddrinfo_or_raise(host: str, port: int) -> list[tuple]:
    try:
        return socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
        )
    except OSError as exc:
        raise WebFetchAccessError(f"Could not resolve host {host!r}: {exc}") from exc


def get_validated_stream_addrinfos_for_egress(
    url: str, policy: WebFetchEgressPolicy
) -> list[tuple]:
    """Resolve and validate a URL for web_fetch, returning getaddrinfo rows for pinning.

    Each HTTP connect pins to only these `getaddrinfo` results so a malicious DNS
    server cannot rebind to a disallowed address between resolution and the TCP
    connect (used by :func:`api.web_tools.outbound._run_web_fetch`).
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in policy.allowed_schemes:
        raise WebFetchEgressViolation(
            f"URL scheme {scheme!r} is not allowed for web_fetch"
        )
    if parsed.username is not None or parsed.password is not None:
        raise WebFetchEgressViolation(
            "URLs containing user information are not allowed for web_fetch"
        )

    if not policy.domains.allows(url):
        raise WebFetchEgressViolation(
            "URL is not allowed by the web_fetch domain policy"
        )

    host = parsed.hostname
    if host is None or host == "":
        raise WebFetchEgressViolation("web_fetch URL must include a host")

    port = _port_for_url(parsed)

    if policy.allow_private_network_targets:
        return _stream_getaddrinfo_or_raise(host, port)

    host_lower = host.lower()
    if host_lower == "localhost" or host_lower.endswith(".localhost"):
        raise WebFetchEgressViolation("localhost targets are not allowed for web_fetch")
    if host_lower.endswith(".local"):
        raise WebFetchEgressViolation(".local hostnames are not allowed for web_fetch")

    try:
        parsed_ip = ipaddress.ip_address(host)
    except ValueError:
        parsed_ip = None

    if parsed_ip is not None:
        if not parsed_ip.is_global:
            raise WebFetchEgressViolation(
                f"Non-public IP host {host!r} is not allowed for web_fetch"
            )
        return _stream_getaddrinfo_or_raise(host, port)

    infos = _stream_getaddrinfo_or_raise(host, port)
    for *_, sockaddr in infos:
        addr = sockaddr[0]
        try:
            resolved = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if not resolved.is_global:
            raise WebFetchEgressViolation(
                f"Host {host!r} resolves to a non-public address ({resolved})"
            )
    return infos


def enforce_web_fetch_egress(url: str, policy: WebFetchEgressPolicy) -> None:
    """Validate ``url`` (scheme, host, and resolved addresses) for web_fetch."""
    get_validated_stream_addrinfos_for_egress(url, policy)
