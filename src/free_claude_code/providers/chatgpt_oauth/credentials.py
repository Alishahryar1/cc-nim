"""ChatGPT/Codex OAuth credential loading and refresh.

This mirrors the token sources used by OpenAI's Codex CLI and the Hermes
auth file. Refreshed tokens are written back to the source auth file so
rotated refresh tokens survive restarts (matching OpenCode's behaviour of
persisting refreshed credentials), and concurrent refreshes are deduplicated
with a process-wide lock.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"


class ChatGPTOAuthError(Exception):
    """Raised when ChatGPT OAuth credential handling fails."""


@dataclasses.dataclass(frozen=True)
class ChatGPTOAuthCredentials:
    """Resolved OAuth credentials for one request."""

    access_token: str
    account_id: str
    refresh_token: str | None = None
    expires_at: int | None = None
    source_name: str = ""


@dataclasses.dataclass(frozen=True)
class _TokenSource:
    name: str
    path: Path
    access_token: str | None
    refresh_token: str | None
    id_token: str | None = None

    @property
    def has_access_token(self) -> bool:
        return isinstance(self.access_token, str) and self.access_token.strip() != ""

    @property
    def has_refresh_token(self) -> bool:
        return isinstance(self.refresh_token, str) and self.refresh_token.strip() != ""


def _home() -> Path:
    return Path.home()


def _codex_home() -> Path:
    codex_home = Path(os.environ.get("CODEX_HOME", "")).expanduser()
    if not str(codex_home).strip() or str(codex_home) == ".":
        codex_home = _home() / ".codex"
    return codex_home


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ChatGPTOAuthError(f"Could not parse {path}: {exc}") from exc


def _load_codex_cli_source() -> _TokenSource:
    path = _codex_home() / "auth.json"
    payload = _load_json(path)
    tokens = payload.get("tokens") or {}
    return _TokenSource(
        name="codex-cli",
        path=path,
        access_token=tokens.get("access_token"),
        refresh_token=tokens.get("refresh_token"),
        id_token=tokens.get("id_token"),
    )


def _load_hermes_source() -> _TokenSource:
    path = _home() / ".hermes" / "auth.json"
    payload = _load_json(path)
    provider = ((payload.get("providers") or {}).get("openai-codex") or {})
    tokens = provider.get("tokens") or {}
    return _TokenSource(
        name="hermes-openai-codex",
        path=path,
        access_token=tokens.get("access_token"),
        refresh_token=tokens.get("refresh_token"),
        id_token=tokens.get("id_token"),
    )


def _reload_source(source: _TokenSource) -> _TokenSource:
    """Re-read one token source from disk (e.g. after another thread refreshed)."""
    if source.name == "codex-cli":
        return _load_codex_cli_source()
    if source.name == "hermes-openai-codex":
        return _load_hermes_source()
    return source


def _load_sources() -> list[_TokenSource]:
    return [_load_hermes_source(), _load_codex_cli_source()]


def _decode_jwt_claims(token: str | None) -> dict[str, Any]:
    if not token or token.count(".") < 2:
        return {}
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload.encode("utf-8")))
    except Exception:
        return {}


def _extract_account_id_from_claims(claims: dict[str, Any]) -> str:
    """Extract the ChatGPT account id from decoded JWT claims.

    Mirrors OpenCode's extraction order: top-level ``chatgpt_account_id``,
    then the namespaced auth claim, then a generic ``account_id``, then the
    first organization id.
    """
    account_id = claims.get("chatgpt_account_id")
    if isinstance(account_id, str) and account_id:
        return account_id
    auth_claim = claims.get("https://api.openai.com/auth") or {}
    account_id = auth_claim.get("chatgpt_account_id")
    if isinstance(account_id, str) and account_id:
        return account_id
    account_id = claims.get("account_id")
    if isinstance(account_id, str) and account_id:
        return account_id
    organizations = claims.get("organizations")
    if isinstance(organizations, list) and organizations:
        first = organizations[0]
        if isinstance(first, dict):
            org_id = first.get("id")
            if isinstance(org_id, str) and org_id:
                return org_id
    return ""


def _extract_account_id(access_token: str) -> str:
    return _extract_account_id_from_claims(_decode_jwt_claims(access_token))


def extract_account_id_from_tokens(
    access_token: str | None = None,
    id_token: str | None = None,
) -> str:
    """Extract the account id, preferring the id token like OpenCode does."""
    if id_token:
        account_id = _extract_account_id_from_claims(_decode_jwt_claims(id_token))
        if account_id:
            return account_id
    if access_token:
        return _extract_account_id_from_claims(_decode_jwt_claims(access_token))
    return ""


def _access_token_seconds_remaining(access_token: str) -> int | None:
    claims = _decode_jwt_claims(access_token)
    exp = claims.get("exp")
    if not isinstance(exp, (int, float)):
        return None
    return int(exp - time.time())


def _refresh_access_token(refresh_token: str) -> tuple[str, str | None, int | None]:
    """Refresh an OAuth access token and return the new credential set."""
    response = httpx.post(
        CODEX_OAUTH_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CODEX_OAUTH_CLIENT_ID,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=httpx.Timeout(30.0),
    )
    if response.status_code != 200:
        raise ChatGPTOAuthError(
            f"OAuth refresh failed with HTTP {response.status_code}"
        )
    payload = response.json()
    new_access = payload.get("access_token")
    new_refresh = payload.get("refresh_token") or refresh_token
    expires_in = payload.get("expires_in")
    if not isinstance(new_access, str) or not new_access:
        raise ChatGPTOAuthError("OAuth refresh response did not contain an access token.")
    expires_at = None
    if isinstance(expires_in, (int, float)):
        expires_at = int(time.time() + expires_in)
    new_id_token = payload.get("id_token")
    return new_access, new_refresh, expires_at, new_id_token


def _persist_refreshed_tokens(
    source: _TokenSource,
    *,
    access_token: str,
    refresh_token: str | None,
    id_token: str | None,
    expires_at: int | None,
) -> None:
    """Write refreshed tokens back to the source auth file.

    OpenCode persists refreshed credentials so rotated refresh tokens keep
    working across restarts; we do the same. Failures are logged but never
    fatal — the in-memory refreshed token still serves this process.
    """
    try:
        payload = _load_json(source.path)
        if source.name == "hermes-openai-codex":
            tokens = (
                payload.setdefault("providers", {})
                .setdefault("openai-codex", {})
                .setdefault("tokens", {})
            )
        else:
            tokens = payload.setdefault("tokens", {})
        tokens["access_token"] = access_token
        if refresh_token:
            tokens["refresh_token"] = refresh_token
        if id_token:
            tokens["id_token"] = id_token
        if expires_at is not None:
            tokens["expires_at"] = expires_at
        source.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = source.path.with_suffix(source.path.suffix + ".tmp")
        try:
            temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temp_path, source.path)
        finally:
            temp_path.unlink(missing_ok=True)
        try:
            os.chmod(source.path, 0o600)
        except OSError:
            pass
    except Exception as exc:  # noqa: BLE001 - persistence must not break refresh
        logger.warning(
            "ChatGPT OAuth: could not persist refreshed tokens to {}: {}",
            source.path,
            exc,
        )


_REFRESH_LOCK = threading.Lock()


def _ensure_fresh_source(source: _TokenSource) -> _TokenSource:
    remaining = (
        _access_token_seconds_remaining(source.access_token)
        if source.access_token
        else None
    )
    if remaining is None or remaining > 300:
        return source
    if not source.has_refresh_token or source.refresh_token is None:
        # Token is expiring and we cannot refresh; return as-is and let the
        # upstream request fail with a clear 401 if expired.
        return source

    with _REFRESH_LOCK:
        # Another thread may have refreshed while we waited on the lock.
        current = _reload_source(source)
        if current.has_access_token:
            remaining = _access_token_seconds_remaining(current.access_token)
            if remaining is not None and remaining > 300:
                return current
        if not current.has_refresh_token or current.refresh_token is None:
            return source

        new_access, new_refresh, expires_at, new_id_token = _refresh_access_token(
            current.refresh_token
        )
        _persist_refreshed_tokens(
            current,
            access_token=new_access,
            refresh_token=new_refresh,
            id_token=new_id_token,
            expires_at=expires_at,
        )
        return dataclasses.replace(
            current,
            access_token=new_access,
            refresh_token=new_refresh,
            id_token=new_id_token or current.id_token,
        )


def _choose_runtime_source(sources: list[_TokenSource]) -> _TokenSource:
    refresh_errors: list[str] = []
    for item in sources:
        if item.name == "hermes-openai-codex" and item.has_access_token:
            try:
                return _ensure_fresh_source(item)
            except ChatGPTOAuthError as exc:
                refresh_errors.append(f"{item.name}: {exc}")
    for item in sources:
        if item.has_access_token:
            try:
                return _ensure_fresh_source(item)
            except ChatGPTOAuthError as exc:
                refresh_errors.append(f"{item.name}: {exc}")
    suffix = f" Refresh failures: {'; '.join(refresh_errors)}" if refresh_errors else ""
    raise ChatGPTOAuthError(
        f"No usable Codex/ChatGPT OAuth access token found.{suffix}"
    )


def load_chatgpt_oauth_credentials(
    *,
    access_token: str | None = None,
    account_id: str | None = None,
) -> ChatGPTOAuthCredentials:
    """Resolve OAuth credentials from explicit values or auth files.

    Priority:
      1. Explicit access_token / account_id.
      2. Token files (~/.hermes/auth.json, ~/.codex/auth.json).
    """
    if access_token and access_token.strip():
        resolved_account_id = (account_id or "").strip() or _extract_account_id(access_token)
        return ChatGPTOAuthCredentials(
            access_token=access_token.strip(),
            account_id=resolved_account_id,
        )

    source = _choose_runtime_source(_load_sources())
    resolved_account_id = (account_id or "").strip() or extract_account_id_from_tokens(
        access_token=source.access_token,
        id_token=source.id_token,
    )
    return ChatGPTOAuthCredentials(
        access_token=source.access_token or "",
        account_id=resolved_account_id,
        refresh_token=source.refresh_token,
        source_name=source.name,
    )


def import_codex_cli_tokens() -> ChatGPTOAuthCredentials:
    """Load ChatGPT/Codex OAuth tokens from an existing Codex CLI installation.

    Raises ChatGPTOAuthError when the auth file is missing, malformed, or does
    not contain a usable access token.
    """
    source = _load_codex_cli_source()
    if not source.has_access_token:
        path = source.path
        raise ChatGPTOAuthError(
            f"No Codex CLI access token found at {path}. "
            "Run 'codex login' first or use the ChatGPT OAuth Login button."
        )
    return ChatGPTOAuthCredentials(
        access_token=source.access_token or "",
        account_id=extract_account_id_from_tokens(
            access_token=source.access_token,
            id_token=source.id_token,
        ),
        refresh_token=source.refresh_token,
        source_name=source.name,
    )
