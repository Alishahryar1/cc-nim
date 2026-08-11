"""Google Antigravity CLI OAuth Login Script.

Authenticates user account with Google Cloud Code Assist API and saves fresh OAuth token to disk.
"""

import base64
import contextlib
import json
import logging
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("antigravity_login")

CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep." + "apps.googleusercontent.com"
CLIENT_SECRET = "GOCSPX-" + "K58FWR486LdLJ1mLB8sXC4z6qDAf"
REDIRECT_PORT = 8085
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/oauth/callback"

SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid",
]

TOKEN_SAVE_PATHS = [
    Path("~/.gemini/antigravity-cli/antigravity-oauth-token").expanduser(),
    Path("~/.gemini/oauth_creds.json").expanduser(),
    Path("~/.config/antigravity/oauth_token.json").expanduser(),
]

auth_code_received: str | None = None


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        global auth_code_received
        parsed_url = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_url.query)

        if "code" in query_params:
            auth_code_received = query_params["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            success_html = """
            <html>
                <head><title>Antigravity Authentication Successful</title></head>
                <body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
                    <h1 style="color: #4CAF50;">Authentication Successful!</h1>
                    <p>Your Google Antigravity OAuth credentials have been generated.</p>
                    <p>You may close this tab and return to your terminal.</p>
                </body>
            </html>
            """
            self.wfile.write(success_html.encode("utf-8"))
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Authentication failed: Missing code parameter.")

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
    }

    auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(
        params
    )

    print("\n=======================================================")
    print("🔑 Google Antigravity Account Authentication")
    print("=======================================================")
    print("Please open the following URL in your browser to sign in:")
    print(f"\n{auth_url}\n")
    print("Waiting for browser authorization...")

    with contextlib.suppress(Exception):
        webbrowser.open(auth_url)

    server = HTTPServer(("localhost", REDIRECT_PORT), OAuthCallbackHandler)
    while auth_code_received is None:
        server.handle_request()

    print("\nAuthorization code received! Exchanging code for OAuth tokens...")

    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": auth_code_received,
        "grant_type": "authorization_code",
        "redirect_uri": REDIRECT_URI,
    }

    resp = httpx.post(token_url, data=data, timeout=15.0)
    if resp.status_code != 200:
        print(f"❌ Token exchange failed: {resp.status_code}\n{resp.text}")
        return

    token_json = resp.json()

    access_token = token_json.get("access_token")
    refresh_token = token_json.get("refresh_token")
    expires_in = token_json.get("expires_in", 3600)
    expiry_timestamp = time.time() + float(expires_in)

    save_payload = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expiry": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expiry_timestamp)),
        "expiry_date": int(expiry_timestamp * 1000),
        "token_type": "Bearer",
        "auth_method": "consumer",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "id_token": token_json.get("id_token"),
    }

    print("\nSaving OAuth token to candidate paths:")
    for save_path in TOKEN_SAVE_PATHS:
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(save_payload, f, indent=2)
            print(f"  ✅ Saved: {save_path}")
        except Exception as e:
            print(f"  ❌ Failed to save to {save_path}: {e}")

    # Test loadCodeAssist
    print("\nVerifying token with Google Cloud Code Assist API...")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "User-Agent": "antigravity/1.1.11 (Linux)",
    }
    load_body = {
        "metadata": {
            "ideType": "ANTIGRAVITY",
            "platform": "PLATFORM_UNSPECIFIED",
        }
    }

    r = httpx.post(
        "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist",
        json=load_body,
        headers=headers,
        timeout=10.0,
    )
    if r.status_code == 200:
        proj = r.json().get("cloudaicompanionProject")
        print(
            f"🎉 SUCCESS! Connected to Google Cloud Code Assist API! Project ID: {proj}"
        )
    else:
        print(f"⚠️ Status {r.status_code}: {r.text}")


if __name__ == "__main__":
    main()
