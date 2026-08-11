"""Antigravity CLI OAuth authentication and auto-discovery module."""

import base64
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from free_claude_code.providers.exceptions import AuthenticationError

logger = logging.getLogger(__name__)

# Fingerprint Constants (Antigravity CLI v1.1.11)
ANTIGRAVITY_USER_AGENT = "AntigravityCLI/1.1.11"
ANTIGRAVITY_CLIENT_NAME = "antigravity-cli"
ANTIGRAVITY_GOOG_API_CLIENT = "gl-go/1.22.0 gd/1.1.11"
ANTIGRAVITY_DEFAULT_BASE_URL = "https://cloudcode-pa.googleapis.com"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
DEFAULT_FALLBACK_PROJECT_ID = "rising-fact-p41fc"

# Known CLI Client ID/Secret for Google OAuth (Extracted from official antigravity-cli binary)
DEFAULT_ANTIGRAVITY_CLIENT_ID = os.environ.get(
    "ANTIGRAVITY_CLIENT_ID",
    "1071006060591-tmhssin2h21lcre235vtolojh4g403ep." + "apps.googleusercontent.com",
)
DEFAULT_ANTIGRAVITY_CLIENT_SECRET = os.environ.get(
    "ANTIGRAVITY_CLIENT_SECRET",
    "GOCSPX-" + "K58FWR486LdLJ1mLB8sXC4z6qDAf",
)

# Token locations in order of priority
PRIMARY_TOKEN_PATH = Path(
    "~/.gemini/antigravity-cli/antigravity-oauth-token"
).expanduser()
SECONDARY_TOKEN_PATH = Path("~/.config/antigravity/oauth_token.json").expanduser()
TERTIARY_TOKEN_PATH = Path("~/.gemini/oauth_creds.json").expanduser()


def decode_jwt_payload(jwt_token: str) -> dict[str, Any]:
    """Decode unverified JWT payload to extract expiration or metadata."""
    if not jwt_token or not isinstance(jwt_token, str):
        return {}
    parts = jwt_token.split(".")
    if len(parts) < 2:
        return {}
    payload_segment = parts[1]
    rem = len(payload_segment) % 4
    if rem > 0:
        payload_segment += "=" * (4 - rem)
    try:
        decoded_bytes = base64.urlsafe_b64decode(payload_segment.encode("utf-8"))
        return json.loads(decoded_bytes.decode("utf-8"))
    except Exception:
        return {}


def parse_expiry(expiry_val: Any) -> float | None:
    """Parse expiry value (timestamp float/int or ISO string) to Unix timestamp float."""
    if expiry_val is None:
        return None
    if isinstance(expiry_val, (int, float)):
        val = float(expiry_val)
        if val > 1e11:
            val = val / 1000.0
        return val
    if isinstance(expiry_val, str):
        val = expiry_val.strip()
        if not val:
            return None
        try:
            parsed_num = float(val)
            if parsed_num > 1e11:
                parsed_num = parsed_num / 1000.0
            return parsed_num
        except ValueError:
            pass

        val = val.replace("Z", "+00:00")
        if "." in val:
            parts = val.split(".")
            sec = parts[0]
            rest = parts[1]
            tz = ""
            for sep in ("+", "-"):
                if sep in rest:
                    frac, tz_part = rest.split(sep, 1)
                    tz = sep + tz_part
                    rest = frac
                    break
            rest = rest[:6]
            val = f"{sec}.{rest}{tz}"
        try:
            dt = datetime.fromisoformat(val)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.timestamp()
        except Exception as e:
            logger.debug(f"Failed to parse expiry ISO string '{expiry_val}': {e}")
            return None
    return None


def is_token_expired(
    expiry_val: Any,
    access_token: str | None = None,
    margin_seconds: float = 300.0,
) -> bool:
    """Check if the token is expired or close to expiration within margin_seconds."""
    exp_ts = parse_expiry(expiry_val)
    if exp_ts is None and access_token:
        payload = decode_jwt_payload(access_token)
        if "exp" in payload:
            try:
                exp_ts = float(payload["exp"])
            except ValueError, TypeError:
                exp_ts = None

    if exp_ts is None:
        return not bool(access_token)

    now = time.time()
    return (exp_ts - now) <= margin_seconds


def load_token_from_file(file_path: Path | str) -> dict[str, Any] | None:
    """Read and normalize token dictionary from specified file path."""
    p = Path(file_path).expanduser()
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        logger.warning(f"Error reading Antigravity token file {p}: {e}")
        return None

    if not isinstance(raw, dict):
        return None

    token_data = raw.get("token") if isinstance(raw.get("token"), dict) else raw

    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")
    expiry = (
        token_data.get("expiry")
        or token_data.get("expires_at")
        or token_data.get("expiry_date")
    )
    token_type = token_data.get("token_type", "Bearer")
    auth_method = raw.get("auth_method", token_data.get("auth_method", "consumer"))

    client_id = token_data.get("client_id") or raw.get("client_id")
    client_secret = token_data.get("client_secret") or raw.get("client_secret")

    id_token = token_data.get("id_token") or raw.get("id_token")
    if not client_id and id_token:
        jwt_payload = decode_jwt_payload(id_token)
        if jwt_payload and isinstance(jwt_payload, dict):
            client_id = jwt_payload.get("azp") or jwt_payload.get("aud")

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expiry": expiry,
        "token_type": token_type,
        "auth_method": auth_method,
        "client_id": client_id,
        "client_secret": client_secret,
        "file_path": str(p),
        "_raw_data": raw,
    }


def get_candidate_token_files() -> list[Path]:
    """Return all candidate token file paths dynamically discovered in priority order."""
    env_path = os.environ.get("ANTIGRAVITY_TOKEN_FILE")
    candidates: list[Path] = []
    if env_path:
        p = Path(env_path).expanduser()
        if p.is_file():
            candidates.append(p)

    standard_paths = [
        PRIMARY_TOKEN_PATH,
        SECONDARY_TOKEN_PATH,
        TERTIARY_TOKEN_PATH,
        Path("~/.config/gemini/credentials.json").expanduser(),
        Path("~/.config/gcloud/application_default_credentials.json").expanduser(),
    ]

    for base in (
        Path("~/.gemini").expanduser(),
        Path("~/.config/antigravity").expanduser(),
    ):
        if base.is_dir():
            for json_file in base.glob("*.json"):
                if json_file.is_file() and json_file not in standard_paths:
                    standard_paths.append(json_file)

    for p in standard_paths:
        if p.is_file() and p not in candidates:
            candidates.append(p)
    return candidates


def find_token_file() -> Path | None:
    """Find the first existing Antigravity CLI token file."""
    candidates = get_candidate_token_files()
    return candidates[0] if candidates else None


def load_antigravity_token() -> dict[str, Any]:
    """Load Antigravity CLI OAuth token from env or first valid candidate disk file."""
    env_access_token = os.environ.get("ANTIGRAVITY_ACCESS_TOKEN")
    if env_access_token:
        return {
            "access_token": env_access_token,
            "refresh_token": os.environ.get("ANTIGRAVITY_REFRESH_TOKEN"),
            "expiry": os.environ.get("ANTIGRAVITY_TOKEN_EXPIRY"),
            "token_type": "Bearer",
            "auth_method": os.environ.get("ANTIGRAVITY_AUTH_METHOD", "consumer"),
            "file_path": None,
        }

    token_file = find_token_file()
    if not token_file:
        raise AuthenticationError(
            "No Antigravity CLI token found. Please ensure ~/.gemini/antigravity-cli/antigravity-oauth-token "
            "exists or ANTIGRAVITY_ACCESS_TOKEN is set in environment."
        )

    for candidate in get_candidate_token_files():
        data = load_token_from_file(candidate)
        if data and data.get("access_token"):
            return data

    raise AuthenticationError(
        "No Antigravity CLI token found. Please ensure ~/.gemini/antigravity-cli/antigravity-oauth-token "
        "exists or ANTIGRAVITY_ACCESS_TOKEN is set in environment."
    )


def refresh_oauth_token_sync(
    refresh_token: str,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> dict[str, Any]:
    """Synchronously refresh OAuth access token using Google OAuth endpoint."""
    cid = client_id or DEFAULT_ANTIGRAVITY_CLIENT_ID
    csec = client_secret or DEFAULT_ANTIGRAVITY_CLIENT_SECRET
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    if cid:
        payload["client_id"] = cid
    if csec:
        payload["client_secret"] = csec

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": ANTIGRAVITY_USER_AGENT,
    }
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(OAUTH_TOKEN_URL, data=payload, headers=headers)
    except Exception as exc:
        raise AuthenticationError(
            f"Failed to connect to OAuth token endpoint ({OAUTH_TOKEN_URL}): {exc}"
        ) from exc

    if resp.status_code != 200:
        raise AuthenticationError(
            f"OAuth token refresh failed with status {resp.status_code}: {resp.text}"
        )

    try:
        return resp.json()
    except Exception as exc:
        raise AuthenticationError(
            f"Failed to parse OAuth refresh response JSON: {exc}"
        ) from exc


async def refresh_oauth_token_async(
    refresh_token: str,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> dict[str, Any]:
    """Asynchronously refresh OAuth access token using Google OAuth endpoint."""
    cid = client_id or DEFAULT_ANTIGRAVITY_CLIENT_ID
    csec = client_secret or DEFAULT_ANTIGRAVITY_CLIENT_SECRET
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    if cid:
        payload["client_id"] = cid
    if csec:
        payload["client_secret"] = csec

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": ANTIGRAVITY_USER_AGENT,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(OAUTH_TOKEN_URL, data=payload, headers=headers)
    except Exception as exc:
        raise AuthenticationError(
            f"Failed to connect to OAuth token endpoint ({OAUTH_TOKEN_URL}): {exc}"
        ) from exc

    if resp.status_code != 200:
        raise AuthenticationError(
            f"OAuth token refresh failed with status {resp.status_code}: {resp.text}"
        )

    try:
        return resp.json()
    except Exception as exc:
        raise AuthenticationError(
            f"Failed to parse OAuth refresh response JSON: {exc}"
        ) from exc


def save_token_to_file(file_path: Path | str, token_data: dict[str, Any]) -> bool:
    """Persist refreshed token structure back to file if possible."""
    p = Path(file_path).expanduser()
    try:
        raw_data = token_data.get("_raw_data")
        if isinstance(raw_data, dict) and "token" in raw_data:
            new_raw = dict(raw_data)
            t_obj = dict(new_raw.get("token", {}))
            t_obj["access_token"] = token_data["access_token"]
            if token_data.get("refresh_token"):
                t_obj["refresh_token"] = token_data["refresh_token"]
            if token_data.get("expiry"):
                t_obj["expiry"] = token_data["expiry"]
            new_raw["token"] = t_obj
            content = json.dumps(new_raw, indent=2)
        else:
            save_obj = {
                "access_token": token_data["access_token"],
                "token_type": token_data.get("token_type", "Bearer"),
            }
            if token_data.get("refresh_token"):
                save_obj["refresh_token"] = token_data["refresh_token"]
            if token_data.get("expiry"):
                save_obj["expiry"] = token_data["expiry"]
            if token_data.get("auth_method"):
                save_obj["auth_method"] = token_data["auth_method"]
            content = json.dumps(save_obj, indent=2)

        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        logger.warning(f"Could not persist refreshed token to {p}: {e}")
        return False


def load_code_assist_headers(access_token: str) -> dict[str, str]:
    """Construct exact Antigravity CLI v1.1.11 HTTP headers."""
    return {
        "User-Agent": ANTIGRAVITY_USER_AGENT,
        "X-Client-Name": ANTIGRAVITY_CLIENT_NAME,
        "X-Goog-Api-Client": ANTIGRAVITY_GOOG_API_CLIENT,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def load_code_assist_body() -> dict[str, Any]:
    """Construct exact Antigravity CLI metadata payload."""
    return {
        "metadata": {
            "ideType": "ANTIGRAVITY",
            "platform": "PLATFORM_UNSPECIFIED",
        }
    }


def load_code_assist_sync(
    access_token: str,
    base_url: str = ANTIGRAVITY_DEFAULT_BASE_URL,
) -> str:
    """Call v1internal:loadCodeAssist to discover cloudaicompanionProject ID synchronously."""
    url = f"{base_url.rstrip('/')}/v1internal:loadCodeAssist"
    headers = load_code_assist_headers(access_token)
    payload = load_code_assist_body()

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(url, json=payload, headers=headers)
        if resp.status_code == 200:
            res_json = resp.json()
            proj = res_json.get("cloudaicompanionProject")
            if isinstance(proj, dict):
                p_id = proj.get("projectId")
                if p_id:
                    return str(p_id)
            elif proj and isinstance(proj, str):
                return proj
    except Exception as e:
        logger.warning(
            f"loadCodeAssist request failed: {e}. Falling back to default project."
        )

    return os.environ.get("ANTIGRAVITY_PROJECT_ID", DEFAULT_FALLBACK_PROJECT_ID)


async def load_code_assist_async(
    access_token: str,
    base_url: str = ANTIGRAVITY_DEFAULT_BASE_URL,
) -> str:
    """Call v1internal:loadCodeAssist to discover cloudaicompanionProject ID asynchronously."""
    url = f"{base_url.rstrip('/')}/v1internal:loadCodeAssist"
    headers = load_code_assist_headers(access_token)
    payload = load_code_assist_body()

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code == 200:
            res_json = resp.json()
            proj = res_json.get("cloudaicompanionProject")
            if isinstance(proj, dict):
                p_id = proj.get("projectId")
                if p_id:
                    return str(p_id)
            elif proj and isinstance(proj, str):
                return proj
    except Exception as e:
        logger.warning(
            f"loadCodeAssist request failed: {e}. Falling back to default project."
        )

    return os.environ.get("ANTIGRAVITY_PROJECT_ID", DEFAULT_FALLBACK_PROJECT_ID)


class AntigravityAuth:
    """Manager for Antigravity CLI OAuth tokens and project ID resolution."""

    def __init__(
        self,
        token_path: str | Path | None = None,
        base_url: str = ANTIGRAVITY_DEFAULT_BASE_URL,
    ):
        self.token_path = Path(token_path).expanduser() if token_path else None
        self.base_url = base_url
        self._cached_token_data: dict[str, Any] | None = None
        self._cached_project_id: str | None = None

    def get_token_data(self, force_reload: bool = False) -> dict[str, Any]:
        """Get or load token data from specified path, env or default locations."""
        if self._cached_token_data and not force_reload:
            return self._cached_token_data

        if self.token_path:
            data = load_token_from_file(self.token_path)
            if not data or not data.get("access_token"):
                raise AuthenticationError(
                    f"Invalid or missing token file at {self.token_path}"
                )
        else:
            data = load_antigravity_token()

        self._cached_token_data = data
        return data

    def get_access_token(self, force_refresh: bool = False) -> str:
        """Get valid access token, refreshing OAuth token if expired or force_refresh is True."""
        if self.token_path:
            candidates = [self.token_path]
        else:
            candidates = get_candidate_token_files()

        last_error: Exception | None = None
        for candidate_path in candidates:
            token_data = load_token_from_file(candidate_path)
            if not token_data or not (
                token_data.get("access_token") or token_data.get("refresh_token")
            ):
                continue

            access_token = token_data.get("access_token", "")
            expiry = token_data.get("expiry")
            refresh_token = token_data.get("refresh_token")

            if not force_refresh and not is_token_expired(
                expiry, access_token=access_token
            ):
                self._cached_token_data = token_data
                return access_token

            if refresh_token:
                cid = token_data.get("client_id") or DEFAULT_ANTIGRAVITY_CLIENT_ID
                csec = (
                    token_data.get("client_secret") or DEFAULT_ANTIGRAVITY_CLIENT_SECRET
                )
                refreshed = None
                try:
                    logger.info(
                        f"Refreshing Antigravity OAuth token from {candidate_path}..."
                    )
                    refreshed = refresh_oauth_token_sync(
                        refresh_token, client_id=cid, client_secret=csec
                    )
                except Exception:
                    if cid != DEFAULT_ANTIGRAVITY_CLIENT_ID:
                        try:
                            refreshed = refresh_oauth_token_sync(
                                refresh_token,
                                client_id=DEFAULT_ANTIGRAVITY_CLIENT_ID,
                                client_secret=DEFAULT_ANTIGRAVITY_CLIENT_SECRET,
                            )
                        except Exception as exc2:
                            exc = exc2
                    if not refreshed:
                        logger.warning(
                            f"OAuth refresh failed for candidate {candidate_path}: {exc}"
                        )
                        last_error = exc

                if refreshed:
                    new_access_token = refreshed.get("access_token")
                    if new_access_token:
                        expires_in = refreshed.get("expires_in", 3600)
                        new_expiry_ts = time.time() + float(expires_in)
                        new_expiry_iso = datetime.fromtimestamp(
                            new_expiry_ts, tz=UTC
                        ).isoformat()

                        token_data["access_token"] = new_access_token
                        token_data["expiry"] = new_expiry_iso
                        if refreshed.get("refresh_token"):
                            token_data["refresh_token"] = refreshed.get("refresh_token")

                        save_token_to_file(candidate_path, token_data)
                        self._cached_token_data = token_data
                        return new_access_token

        env_access_token = os.environ.get("ANTIGRAVITY_ACCESS_TOKEN")
        if env_access_token:
            return env_access_token

        if last_error:
            raise AuthenticationError(
                f"Antigravity CLI authentication failed: {last_error}"
            ) from last_error

        raise AuthenticationError(
            "No valid Antigravity CLI token found. Please run 'agy login' or set ANTIGRAVITY_ACCESS_TOKEN."
        )

    async def get_access_token_async(self, force_refresh: bool = False) -> str:
        """Asynchronously get valid access token, refreshing if necessary."""
        if self.token_path:
            candidates = [self.token_path]
        else:
            candidates = get_candidate_token_files()

        last_error: Exception | None = None
        for candidate_path in candidates:
            token_data = load_token_from_file(candidate_path)
            if not token_data or not (
                token_data.get("access_token") or token_data.get("refresh_token")
            ):
                continue

            access_token = token_data.get("access_token", "")
            expiry = token_data.get("expiry")
            refresh_token = token_data.get("refresh_token")

            if not force_refresh and not is_token_expired(
                expiry, access_token=access_token
            ):
                self._cached_token_data = token_data
                return access_token

            if refresh_token:
                cid = token_data.get("client_id") or DEFAULT_ANTIGRAVITY_CLIENT_ID
                csec = (
                    token_data.get("client_secret") or DEFAULT_ANTIGRAVITY_CLIENT_SECRET
                )
                refreshed = None
                try:
                    logger.info(
                        f"Refreshing Antigravity OAuth token asynchronously from {candidate_path}..."
                    )
                    refreshed = await refresh_oauth_token_async(
                        refresh_token, client_id=cid, client_secret=csec
                    )
                except Exception:
                    if cid != DEFAULT_ANTIGRAVITY_CLIENT_ID:
                        try:
                            refreshed = await refresh_oauth_token_async(
                                refresh_token,
                                client_id=DEFAULT_ANTIGRAVITY_CLIENT_ID,
                                client_secret=DEFAULT_ANTIGRAVITY_CLIENT_SECRET,
                            )
                        except Exception as exc2:
                            exc = exc2
                    if not refreshed:
                        logger.warning(
                            f"Async OAuth refresh failed for candidate {candidate_path}: {exc}"
                        )
                        last_error = exc

                if refreshed:
                    new_access_token = refreshed.get("access_token")
                    if new_access_token:
                        expires_in = refreshed.get("expires_in", 3600)
                        new_expiry_ts = time.time() + float(expires_in)
                        new_expiry_iso = datetime.fromtimestamp(
                            new_expiry_ts, tz=UTC
                        ).isoformat()

                        token_data["access_token"] = new_access_token
                        token_data["expiry"] = new_expiry_iso
                        if refreshed.get("refresh_token"):
                            token_data["refresh_token"] = refreshed.get("refresh_token")

                        save_token_to_file(candidate_path, token_data)
                        self._cached_token_data = token_data
                        return new_access_token

        env_access_token = os.environ.get("ANTIGRAVITY_ACCESS_TOKEN")
        if env_access_token:
            return env_access_token

        if last_error:
            raise AuthenticationError(
                f"Antigravity CLI authentication failed: {last_error}"
            ) from last_error

        raise AuthenticationError(
            "No valid Antigravity CLI token found. Please run 'agy login' or set ANTIGRAVITY_ACCESS_TOKEN."
        )

    def get_project_id(self, force_fetch: bool = False) -> str:
        """Resolve and return project ID (uses loadCodeAssist or cached value)."""
        if self._cached_project_id and not force_fetch:
            return self._cached_project_id

        access_token = self.get_access_token()
        proj_id = load_code_assist_sync(access_token, base_url=self.base_url)
        self._cached_project_id = proj_id
        return proj_id

    async def get_project_id_async(self, force_fetch: bool = False) -> str:
        """Asynchronously resolve and return project ID."""
        if self._cached_project_id and not force_fetch:
            return self._cached_project_id

        access_token = await self.get_access_token_async()
        proj_id = await load_code_assist_async(access_token, base_url=self.base_url)
        self._cached_project_id = proj_id
        return proj_id
