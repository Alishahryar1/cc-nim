"""Browser-based ChatGPT/Codex OAuth login with PKCE and a local callback.

This mirrors OpenCode's primary "ChatGPT Pro/Plus (browser)" method: a local
HTTP server on ``127.0.0.1:1455`` receives the ``/auth/callback`` redirect,
and the authorization code is exchanged for tokens using PKCE (S256). The
user never copies a URL or code — the browser opens automatically and the
callback completes the flow.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import string
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from .credentials import (
    CODEX_OAUTH_CLIENT_ID,
    CODEX_OAUTH_TOKEN_URL,
    extract_account_id_from_tokens,
)
from .oauth_login import (
    ChatGPTOAuthLoginError,
    ChatGPTOAuthLoginTimeoutError,
    _write_codex_auth_file,
)

CHATGPT_OAUTH_ISSUER = "https://auth.openai.com"
CHATGPT_OAUTH_AUTHORIZE_URL = f"{CHATGPT_OAUTH_ISSUER}/oauth/authorize"
OAUTH_CALLBACK_HOST = "127.0.0.1"
OAUTH_CALLBACK_PORT = 1455
OAUTH_CALLBACK_PATH = "/auth/callback"
OAUTH_CANCEL_PATH = "/cancel"
OAUTH_SCOPE = "openid profile email offline_access"
OAUTH_ORIGINATOR = "free-claude-code"
BROWSER_LOGIN_TIMEOUT_SECONDS = 300.0


class ChatGPTOAuthBrowserLoginError(ChatGPTOAuthLoginError):
    """Raised when the browser-based OAuth login fails."""


class ChatGPTOAuthBrowserUnavailableError(ChatGPTOAuthBrowserLoginError):
    """Raised when the browser flow cannot start (no browser or busy port)."""


def _generate_pkce() -> tuple[str, str]:
    """Return a (verifier, S256 challenge) pair shaped like OpenCode's."""
    alphabet = string.ascii_letters + string.digits + "-._~"
    verifier = "".join(secrets.choice(alphabet) for _ in range(43))
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def build_authorize_url(
    redirect_uri: str,
    code_challenge: str,
    state: str,
) -> str:
    """Build the ChatGPT OAuth authorize URL (Codex simplified flow)."""
    params = {
        "response_type": "code",
        "client_id": CODEX_OAUTH_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": OAUTH_SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "state": state,
        "originator": OAUTH_ORIGINATOR,
    }
    return f"{CHATGPT_OAUTH_AUTHORIZE_URL}?{urlencode(params)}"


def _callback_page(title: str, message: str, *, success: bool) -> bytes:
    color = "#10a37f" if success else "#d93025"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; display: flex; min-height: 100vh;
         align-items: center; justify-content: center; margin: 0;
         background: #0d0d0d; color: #ececec; }}
  .card {{ text-align: center; max-width: 420px; padding: 32px; }}
  h1 {{ color: {color}; font-size: 1.4rem; margin-bottom: 12px; }}
  p {{ color: #b4b4b4; line-height: 1.5; }}
</style>
</head>
<body>
  <div class="card">
    <h1>{title}</h1>
    <p>{message}</p>
    <p>You can close this tab now.</p>
  </div>
  <script>setTimeout(function () {{ window.close(); }}, 3000);</script>
</body>
</html>""".encode("utf-8")


def _exchange_code_for_tokens(
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> dict[str, Any]:
    """Exchange an authorization code for OAuth tokens (PKCE)."""
    response = httpx.post(
        CODEX_OAUTH_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": CODEX_OAUTH_CLIENT_ID,
            "code_verifier": code_verifier,
        },
        timeout=httpx.Timeout(30.0),
    )
    if response.status_code != 200:
        raise ChatGPTOAuthBrowserLoginError(
            f"Token exchange failed: HTTP {response.status_code}"
        )
    data = response.json()
    if not isinstance(data, dict):
        raise ChatGPTOAuthBrowserLoginError(
            "Token exchange response was not a JSON object"
        )
    return data


class _BrowserLoginFlow:
    """State for one in-flight browser login."""

    def __init__(self, port: int = OAUTH_CALLBACK_PORT) -> None:
        self.port = port
        self.verifier, self.challenge = _generate_pkce()
        self.state = secrets.token_urlsafe(32)
        self.done = threading.Event()
        self.deadline = time.monotonic() + BROWSER_LOGIN_TIMEOUT_SECONDS
        self.tokens: dict[str, Any] | None = None
        self.error: str | None = None

    @property
    def redirect_uri(self) -> str:
        return f"http://localhost:{self.port}{OAUTH_CALLBACK_PATH}"

    def finish_tokens(self, tokens: dict[str, Any]) -> None:
        self.tokens = tokens
        self.done.set()

    def finish_error(self, message: str) -> None:
        self.error = message
        self.done.set()

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self.deadline


class _CallbackHandler(BaseHTTPRequestHandler):
    """Handle the OAuth redirect from the user's browser."""

    server: "_CallbackHTTPServer"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send_page(self, status: int, title: str, message: str, *, success: bool) -> None:
        body = _callback_page(title, message, success=success)
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        flow = self.server.active_flow

        if parsed.path == OAUTH_CANCEL_PATH:
            if flow is not None:
                flow.finish_error("Login cancelled")
            self._send_page(200, "Login cancelled", "The login was cancelled.", success=False)
            return

        if parsed.path != OAUTH_CALLBACK_PATH:
            self.send_error(404)
            return

        params = parse_qs(parsed.query)
        error = (params.get("error") or [None])[0]
        if error is not None:
            description = (params.get("error_description") or [error])[0]
            if flow is not None:
                flow.finish_error(description)
            self._send_page(200, "Login failed", description, success=False)
            return

        code = (params.get("code") or [None])[0]
        state = (params.get("state") or [None])[0]
        if not code:
            message = "Missing authorization code"
            if flow is not None:
                flow.finish_error(message)
            self._send_page(400, "Login failed", message, success=False)
            return

        if flow is None or state != flow.state:
            message = "Invalid state - potential CSRF attack"
            if flow is not None:
                flow.finish_error(message)
            self._send_page(400, "Login failed", message, success=False)
            return

        try:
            tokens = _exchange_code_for_tokens(code, flow.redirect_uri, flow.verifier)
        except ChatGPTOAuthLoginError as exc:
            flow.finish_error(str(exc))
            self._send_page(200, "Login failed", str(exc), success=False)
            return

        flow.finish_tokens(tokens)
        self._send_page(
            200,
            "Login successful",
            "ChatGPT OAuth login completed. Returning to Free Claude Code...",
            success=True,
        )


class _CallbackHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port: int = OAUTH_CALLBACK_PORT) -> None:
        self.active_flow: _BrowserLoginFlow | None = None
        self.flow_lock = threading.Lock()
        super().__init__((OAUTH_CALLBACK_HOST, port), _CallbackHandler)

    def begin_flow(self, flow: _BrowserLoginFlow) -> None:
        with self.flow_lock:
            flow.port = self.server_address[1]
            self.active_flow = flow


_SERVER_LOCK = threading.Lock()
_SERVER: _CallbackHTTPServer | None = None
_SERVER_THREAD: threading.Thread | None = None


def _ensure_callback_server() -> _CallbackHTTPServer:
    """Start (or reuse) the local OAuth callback server."""
    global _SERVER, _SERVER_THREAD
    with _SERVER_LOCK:
        if _SERVER is not None:
            return _SERVER
        try:
            server = _CallbackHTTPServer()
        except OSError as exc:
            raise ChatGPTOAuthBrowserUnavailableError(
                f"Could not start the local OAuth callback server on "
                f"{OAUTH_CALLBACK_HOST}:{OAUTH_CALLBACK_PORT} ({exc}). "
                "Close whatever is using that port or use the device login flow."
            ) from exc
        thread = threading.Thread(
            target=server.serve_forever,
            name="chatgpt-oauth-callback",
            daemon=True,
        )
        thread.start()
        _SERVER = server
        _SERVER_THREAD = thread
        return server


def start_browser_login() -> dict[str, str]:
    """Begin a browser login and return the authorize URL to open.

    Replaces any older pending flow. The caller opens the returned URL in the
    user's browser and then polls :func:`browser_login_status`.
    """
    server = _ensure_callback_server()
    flow = _BrowserLoginFlow()
    server.begin_flow(flow)
    return {
        "authorize_url": build_authorize_url(
            flow.redirect_uri, flow.challenge, flow.state
        ),
        "expires_in": str(int(BROWSER_LOGIN_TIMEOUT_SECONDS)),
    }


def browser_login_status(
    *,
    auth_path: Any = None,
) -> dict[str, Any]:
    """Return the status of the in-flight browser login.

    Status values: ``idle`` (no flow started), ``pending``, ``complete``
    (tokens persisted and included), or ``error``.
    """
    server = _SERVER
    flow = server.active_flow if server is not None else None
    if flow is None:
        return {"status": "idle"}

    if not flow.done.is_set():
        if flow.expired:
            flow.finish_error("OAuth callback timeout - authorization took too long")
        else:
            return {"status": "pending"}

    if flow.tokens is not None:
        tokens = flow.tokens
        access_token = tokens.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            return {"status": "error", "message": "Token exchange returned no access token"}
        account_id = extract_account_id_from_tokens(
            access_token=access_token,
            id_token=tokens.get("id_token"),
        )
        tokens["account_id"] = account_id
        _write_codex_auth_file(tokens, auth_path=auth_path)
        return {
            "status": "complete",
            "access_token": access_token,
            "account_id": account_id,
            "message": "Login successful. Tokens saved to ~/.codex/auth.json.",
        }

    return {
        "status": "error",
        "message": flow.error or "Browser login failed",
    }


def perform_browser_login(
    *,
    timeout_seconds: float = BROWSER_LOGIN_TIMEOUT_SECONDS,
    auth_path: Any = None,
) -> dict[str, Any]:
    """Run the full browser login: start server, open browser, wait, persist.

    Raises ChatGPTOAuthBrowserUnavailableError when the flow cannot start,
    ChatGPTOAuthLoginTimeoutError on timeout, or ChatGPTOAuthBrowserLoginError
    on failure. Returns the token payload on success.
    """
    server = _ensure_callback_server()
    flow = _BrowserLoginFlow()
    server.begin_flow(flow)
    authorize_url = build_authorize_url(flow.redirect_uri, flow.challenge, flow.state)

    print("Opening ChatGPT OAuth login in your browser...", flush=True)
    print(f"If it does not open, visit: {authorize_url}", flush=True)
    if not webbrowser.open(authorize_url):
        raise ChatGPTOAuthBrowserUnavailableError(
            "Could not open a browser automatically. "
            "Use --device for the headless device-code login instead."
        )

    if not flow.done.wait(timeout=timeout_seconds):
        raise ChatGPTOAuthLoginTimeoutError(
            "Timed out waiting for the OAuth callback."
        )

    if flow.tokens is None:
        raise ChatGPTOAuthBrowserLoginError(flow.error or "Browser login failed")

    tokens = flow.tokens
    access_token = tokens.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise ChatGPTOAuthBrowserLoginError(
            "Token exchange did not return an access_token"
        )

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
