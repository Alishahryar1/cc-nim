"""Headless ChatGPT/Codex OAuth device-flow login.

This mirrors the device-auth path used by OpenCode so Free Claude Code can
obtain ChatGPT/Codex OAuth tokens without requiring the official ``codex`` CLI.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from .credentials import (
    CODEX_OAUTH_CLIENT_ID,
    CODEX_OAUTH_TOKEN_URL,
    _codex_home,
    _decode_jwt_claims,
    extract_account_id_from_tokens,
)

CHATGPT_OAUTH_ISSUER = "https://auth.openai.com"
CHATGPT_OAUTH_DEVICE_URL = f"{CHATGPT_OAUTH_ISSUER}/api/accounts/deviceauth/usercode"
CHATGPT_OAUTH_DEVICE_TOKEN_URL = (
    f"{CHATGPT_OAUTH_ISSUER}/api/accounts/deviceauth/token"
)
CHATGPT_OAUTH_DEVICE_VERIFICATION_URL = f"{CHATGPT_OAUTH_ISSUER}/codex/device"
CHATGPT_OAUTH_DEVICE_CALLBACK = f"{CHATGPT_OAUTH_ISSUER}/deviceauth/callback"
CHATGPT_OAUTH_POLL_SAFETY_MS = 3000


class _PendingChatGPTOAuthLogin(Exception):
    """Internal signal that the user has not yet completed device authorization."""


class ChatGPTOAuthLoginError(Exception):
    """Raised when the ChatGPT OAuth login flow fails."""


class ChatGPTOAuthLoginTimeoutError(ChatGPTOAuthLoginError):
    """Raised when the user does not complete login before the deadline."""


def _user_agent() -> str:
    """Return a User-Agent for OAuth endpoints."""
    from free_claude_code.core.version import package_version

    return f"free-claude-code/{package_version()}"


def _parse_device_response(response: httpx.Response) -> dict[str, Any]:
    if response.status_code != 200:
        raise ChatGPTOAuthLoginError(
            f"Device auth initiation failed: HTTP {response.status_code}"
        )
    data = response.json()
    if not isinstance(data, dict):
        raise ChatGPTOAuthLoginError("Device auth response was not a JSON object")
    return data


def _initiate_device_auth(
    http_client: httpx.Client | None = None,
) -> tuple[str, str, int]:
    """Start a device-auth flow and return (device_auth_id, user_code, interval_ms)."""
    client = http_client or httpx.Client()
    try:
        response = client.post(
            CHATGPT_OAUTH_DEVICE_URL,
            headers={
                "Content-Type": "application/json",
                "User-Agent": _user_agent(),
            },
            json={"client_id": CODEX_OAUTH_CLIENT_ID},
            timeout=httpx.Timeout(30.0),
        )
    finally:
        if http_client is None:
            client.close()

    data = _parse_device_response(response)
    device_auth_id = data.get("device_auth_id")
    user_code = data.get("user_code")
    interval = data.get("interval")
    if not isinstance(device_auth_id, str) or not device_auth_id:
        raise ChatGPTOAuthLoginError("Device auth response missing device_auth_id")
    if not isinstance(user_code, str) or not user_code:
        raise ChatGPTOAuthLoginError("Device auth response missing user_code")
    try:
        interval_ms = max(int(interval or 5), 1) * 1000
    except (TypeError, ValueError) as exc:
        raise ChatGPTOAuthLoginError(
            f"Device auth response contained invalid interval: {interval}"
        ) from exc
    return device_auth_id, user_code, interval_ms


def _poll_device_auth(
    device_auth_id: str,
    user_code: str,
    *,
    deadline: float,
    interval_ms: int = 5000,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Poll the device-auth token endpoint until success or timeout."""
    client = http_client or httpx.Client()
    try:
        while True:
            response = client.post(
                CHATGPT_OAUTH_DEVICE_TOKEN_URL,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": _user_agent(),
                },
                json={
                    "device_auth_id": device_auth_id,
                    "user_code": user_code,
                },
                timeout=httpx.Timeout(30.0),
            )

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "authorization_code" in data:
                    return data
                raise ChatGPTOAuthLoginError(
                    "Device auth token response missing authorization_code"
                )

            if response.status_code not in {403, 404}:
                raise ChatGPTOAuthLoginError(
                    f"Device auth token polling failed: HTTP {response.status_code}"
                )

            if time.monotonic() >= deadline:
                raise ChatGPTOAuthLoginTimeoutError(
                    "Timed out waiting for ChatGPT OAuth login completion."
                )

            time.sleep((interval_ms + CHATGPT_OAUTH_POLL_SAFETY_MS) / 1000.0)
    finally:
        if http_client is None:
            client.close()


def _exchange_authorization_code(
    authorization_code: str,
    code_verifier: str,
    *,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Exchange an authorization code for OAuth tokens."""
    client = http_client or httpx.Client()
    try:
        response = client.post(
            CODEX_OAUTH_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": CHATGPT_OAUTH_DEVICE_CALLBACK,
                "client_id": CODEX_OAUTH_CLIENT_ID,
                "code_verifier": code_verifier,
            },
            timeout=httpx.Timeout(30.0),
        )
    finally:
        if http_client is None:
            client.close()

    if response.status_code != 200:
        raise ChatGPTOAuthLoginError(
            f"Token exchange failed: HTTP {response.status_code}"
        )
    data = response.json()
    if not isinstance(data, dict):
        raise ChatGPTOAuthLoginError("Token exchange response was not a JSON object")
    return data


def _extract_expires_at(tokens: dict[str, Any]) -> int | None:
    """Return a Unix timestamp for token expiry, if known."""
    # Prefer the explicit expires_in from the token response.
    expires_in = tokens.get("expires_in")
    if isinstance(expires_in, (int, float)):
        return int(time.time() + expires_in)
    # Fall back to the exp claim in the access token.
    access_token = tokens.get("access_token")
    if isinstance(access_token, str):
        claims = _decode_jwt_claims(access_token)
        exp = claims.get("exp")
        if isinstance(exp, (int, float)):
            return int(exp)
    return None


def _write_codex_auth_file(
    tokens: dict[str, Any],
    *,
    auth_path: Path | None = None,
) -> Path:
    """Persist tokens to the Codex CLI auth file used by the provider loader."""
    path = auth_path or (_codex_home() / "auth.json")
    path.parent.mkdir(parents=True, exist_ok=True)

    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    id_token = tokens.get("id_token")
    expires_at = _extract_expires_at(tokens)

    payload_tokens: dict[str, Any] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
    }
    if isinstance(id_token, str) and id_token:
        payload_tokens["id_token"] = id_token
    if expires_at is not None:
        payload_tokens["expires_at"] = expires_at

    payload = {"tokens": payload_tokens}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def perform_chatgpt_oauth_login(
    *,
    timeout_seconds: float = 600.0,
    auth_path: Path | None = None,
    http_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Run the full ChatGPT/Codex device-auth flow and persist the tokens.

    Returns the token payload. Raises ChatGPTOAuthLoginError or
    ChatGPTOAuthLoginTimeoutError on failure.
    """
    device_auth_id, user_code, interval_ms = _initiate_device_auth(http_client)

    print(
        "\n".join(
            [
                "",
                "ChatGPT OAuth login",
                "===================",
                f"1. Open: {CHATGPT_OAUTH_DEVICE_VERIFICATION_URL}",
                f"2. Enter code: {user_code}",
                f"3. Waiting up to {int(timeout_seconds)} seconds for authorization...",
            ]
        ),
        flush=True,
    )

    deadline = time.monotonic() + timeout_seconds
    device_data = _poll_device_auth(
        device_auth_id,
        user_code,
        deadline=deadline,
        interval_ms=interval_ms,
        http_client=http_client,
    )

    authorization_code = device_data.get("authorization_code")
    code_verifier = device_data.get("code_verifier")
    if not isinstance(authorization_code, str) or not authorization_code:
        raise ChatGPTOAuthLoginError(
            "Device auth response missing authorization_code"
        )
    if not isinstance(code_verifier, str) or not code_verifier:
        raise ChatGPTOAuthLoginError("Device auth response missing code_verifier")

    tokens = _exchange_authorization_code(
        authorization_code,
        code_verifier,
        http_client=http_client,
    )

    access_token = tokens.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise ChatGPTOAuthLoginError("Token exchange did not return an access_token")

    account_id = extract_account_id_from_tokens(
        access_token=access_token,
        id_token=tokens.get("id_token"),
    )
    tokens["account_id"] = account_id

    path = _write_codex_auth_file(tokens, auth_path=auth_path)
    print(f"Tokens saved to: {path}", flush=True)
    if account_id:
        print(f"Account ID: {account_id}", flush=True)

    return tokens


def exchange_device_auth_for_tokens(
    device_auth_id: str,
    user_code: str,
    *,
    timeout_seconds: float = 5.0,
    auth_path: Path | None = None,
    http_client: httpx.Client | None = None,
) -> dict[str, Any] | None:
    """Poll for device-auth completion and exchange tokens.

    Returns the token payload on success, or ``None`` if the user has not yet
    completed authorization (the caller should retry). Raises
    ChatGPTOAuthLoginError for terminal failures.
    """
    try:
        device_data = _poll_device_auth(
            device_auth_id,
            user_code,
            deadline=time.monotonic() + timeout_seconds,
            http_client=http_client,
        )
    except ChatGPTOAuthLoginTimeoutError:
        return None

    authorization_code = device_data.get("authorization_code")
    code_verifier = device_data.get("code_verifier")
    if not isinstance(authorization_code, str) or not authorization_code:
        raise ChatGPTOAuthLoginError(
            "Device auth response missing authorization_code"
        )
    if not isinstance(code_verifier, str) or not code_verifier:
        raise ChatGPTOAuthLoginError("Device auth response missing code_verifier")

    tokens = _exchange_authorization_code(
        authorization_code,
        code_verifier,
        http_client=http_client,
    )

    access_token = tokens.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise ChatGPTOAuthLoginError("Token exchange did not return an access_token")

    account_id = extract_account_id_from_tokens(
        access_token=access_token,
        id_token=tokens.get("id_token"),
    )
    tokens["account_id"] = account_id
    _write_codex_auth_file(tokens, auth_path=auth_path)
    return tokens


def chatgpt_oauth_login_command() -> None:
    """CLI entry point for ``fcc-chatgpt-oauth-login``."""
    try:
        perform_chatgpt_oauth_login()
    except ChatGPTOAuthLoginTimeoutError as exc:
        print(f"Timeout: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    except ChatGPTOAuthLoginError as exc:
        print(f"Login failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
