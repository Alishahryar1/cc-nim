"""ChatGPT/Codex OAuth credential loading and refresh.

This mirrors the token sources used by OpenAI's Codex CLI and the Hermes
auth file. Tokens are read from disk (not written) and refreshed in memory
when they are close to expiry.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

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
    )


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


def _extract_account_id(access_token: str) -> str:
    claims = _decode_jwt_claims(access_token)
    auth_claim = claims.get("https://api.openai.com/auth") or {}
    account_id = auth_claim.get("chatgpt_account_id")
    if isinstance(account_id, str) and account_id:
        return account_id
    # Fallback: some tokens carry the account id in the top-level claim.
    account_id = claims.get("chatgpt_account_id") or claims.get("account_id")
    if isinstance(account_id, str) and account_id:
        return account_id
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
    return new_access, new_refresh, expires_at


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
    new_access, new_refresh, _ = _refresh_access_token(source.refresh_token)
    return dataclasses.replace(
        source,
        access_token=new_access,
        refresh_token=new_refresh,
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
    resolved_account_id = (account_id or "").strip() or _extract_account_id(
        source.access_token or ""
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
        account_id=_extract_account_id(source.access_token or ""),
        refresh_token=source.refresh_token,
        source_name=source.name,
    )
