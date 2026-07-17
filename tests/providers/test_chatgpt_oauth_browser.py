"""Tests for the browser-based ChatGPT OAuth login (PKCE + local callback)."""

from __future__ import annotations

import json
import threading
import urllib.parse

import httpx
import pytest

from free_claude_code.providers.chatgpt_oauth import browser_login
from free_claude_code.providers.chatgpt_oauth.browser_login import (
    _BrowserLoginFlow,
    _CallbackHTTPServer,
    _generate_pkce,
    build_authorize_url,
)

FAKE_TOKENS = {
    "access_token": "access_browser_1",
    "refresh_token": "refresh_browser_1",
    "id_token": "id_browser_1",
    "expires_in": 3600,
}


def test_generate_pkce_shapes():
    verifier, challenge = _generate_pkce()
    assert len(verifier) == 43
    assert len(challenge) == 43
    assert "=" not in challenge
    assert "+" not in challenge
    assert "/" not in challenge


def test_build_authorize_url_contains_codex_params():
    url = build_authorize_url("http://localhost:1455/auth/callback", "chal", "st")
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert parsed.netloc == "auth.openai.com"
    assert parsed.path == "/oauth/authorize"
    assert params["response_type"] == ["code"]
    assert params["client_id"] == ["app_EMoamEEZ73f0CkXaXp7hrann"]
    assert params["redirect_uri"] == ["http://localhost:1455/auth/callback"]
    assert params["scope"] == ["openid profile email offline_access"]
    assert params["code_challenge"] == ["chal"]
    assert params["code_challenge_method"] == ["S256"]
    assert params["id_token_add_organizations"] == ["true"]
    assert params["codex_cli_simplified_flow"] == ["true"]
    assert params["state"] == ["st"]
    assert params["originator"] == ["free-claude-code"]


@pytest.fixture
def callback_server(monkeypatch):
    """Start a throwaway callback server on an ephemeral port.

    Tests never bind the production port (1455), so parallel xdist workers
    cannot steal each other's OAuth callbacks on Windows.
    """
    try:
        server = _CallbackHTTPServer(port=0)
    except OSError as exc:
        pytest.skip(f"Could not start callback server: {exc}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(
        browser_login,
        "_exchange_code_for_tokens",
        lambda code, redirect_uri, code_verifier: FAKE_TOKENS,
    )
    monkeypatch.setattr(browser_login, "_SERVER", server)
    yield server
    server.shutdown()
    server.server_close()


def _begin_flow(server: _CallbackHTTPServer) -> _BrowserLoginFlow:
    flow = _BrowserLoginFlow()
    server.begin_flow(flow)
    return flow


def _callback_url(server: _CallbackHTTPServer, flow: _BrowserLoginFlow, **extra) -> str:
    params = {"code": "auth_code_1", "state": flow.state, **extra}
    port = server.server_address[1]
    return f"http://127.0.0.1:{port}/auth/callback?" + urllib.parse.urlencode(params)


def test_browser_login_callback_completes_flow(callback_server, tmp_path):
    flow = _begin_flow(callback_server)

    response = httpx.get(_callback_url(callback_server, flow), timeout=10.0)

    assert response.status_code == 200
    assert "Login successful" in response.text
    assert flow.done.is_set()
    assert flow.tokens == FAKE_TOKENS

    status = browser_login.browser_login_status(
        auth_path=tmp_path / ".codex" / "auth.json"
    )
    assert status["status"] == "complete"
    assert status["access_token"] == "access_browser_1"

    saved = json.loads((tmp_path / ".codex" / "auth.json").read_text(encoding="utf-8"))
    assert saved["tokens"]["access_token"] == "access_browser_1"
    assert saved["tokens"]["refresh_token"] == "refresh_browser_1"
    assert saved["tokens"]["id_token"] == "id_browser_1"


def test_browser_login_callback_rejects_bad_state(callback_server, tmp_path):
    flow = _begin_flow(callback_server)

    url = _callback_url(callback_server, flow, state="forged-state")
    response = httpx.get(url, timeout=10.0)

    assert response.status_code == 400
    assert "Invalid state" in response.text
    assert flow.done.is_set()
    assert flow.tokens is None
    assert "CSRF" in (flow.error or "")

    status = browser_login.browser_login_status(
        auth_path=tmp_path / ".codex" / "auth.json"
    )
    assert status["status"] == "error"


def test_browser_login_callback_surfaces_oauth_error(callback_server):
    flow = _begin_flow(callback_server)

    params = urllib.parse.urlencode(
        {"error": "access_denied", "error_description": "User denied access"}
    )
    port = callback_server.server_address[1]
    response = httpx.get(
        f"http://127.0.0.1:{port}/auth/callback?{params}",
        timeout=10.0,
    )

    assert response.status_code == 200
    assert "User denied access" in response.text
    assert flow.error == "User denied access"


def test_start_browser_login_uses_server_port(callback_server):
    payload = browser_login.start_browser_login()

    parsed = urllib.parse.urlparse(payload["authorize_url"])
    params = urllib.parse.parse_qs(parsed.query)
    port = callback_server.server_address[1]
    assert params["redirect_uri"] == [f"http://localhost:{port}/auth/callback"]
    assert callback_server.active_flow is not None
