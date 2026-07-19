"""Local admin UI routes and APIs."""

import ipaddress
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from free_claude_code.application.model_metadata import ProviderModelRefreshResult
from free_claude_code.config.admin.manifest import FIELD_BY_KEY
from free_claude_code.config.admin.persistence import validate_updates
from free_claude_code.config.admin.sources import is_locked_source
from free_claude_code.config.admin.values import load_config_response, load_value_state
from free_claude_code.config.model_refs import configured_chat_model_refs
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.providers.runtime.config import parse_credential_keys
from free_claude_code.providers.runtime.rotating import RotatingProvider

from .dependencies import get_services
from .ports import ApiServices

router = APIRouter()

STATIC_DIR = Path(__file__).resolve().parent / "admin_static"
LOCAL_PROVIDER_PATHS = {
    "lmstudio": "/models",
    "llamacpp": "/models",
    "ollama": "/api/tags",
}


class AdminConfigPayload(BaseModel):
    """Partial config update submitted by the admin UI."""

    values: dict[str, Any] = Field(default_factory=dict)


def _is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    normalized = host.strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _origin_is_local(origin: str | None) -> bool:
    if not origin:
        return True
    parsed = urlsplit(origin)
    return _is_loopback_host(parsed.hostname)


def require_loopback_admin(request: Request) -> None:
    """Allow admin access only from the local machine."""

    client_host = request.client.host if request.client else None
    if not _is_loopback_host(client_host):
        raise HTTPException(status_code=403, detail="Admin UI is local-only")

    origin = request.headers.get("origin")
    if not _origin_is_local(origin):
        raise HTTPException(status_code=403, detail="Admin UI is local-only")


def _asset_response(filename: str) -> FileResponse:
    path = STATIC_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Admin asset not found")
    return FileResponse(path)


@router.get("/admin", include_in_schema=False)
async def admin_page(request: Request):
    require_loopback_admin(request)
    return _asset_response("index.html")


@router.get("/admin/assets/{filename}", include_in_schema=False)
async def admin_asset(filename: str, request: Request):
    require_loopback_admin(request)
    if filename not in {"admin.css", "admin.js"}:
        raise HTTPException(status_code=404, detail="Admin asset not found")
    return _asset_response(filename)


@router.get("/admin/api/config")
async def get_admin_config(request: Request):
    require_loopback_admin(request)
    return load_config_response()


@router.post("/admin/api/config/validate")
async def validate_admin_config(payload: AdminConfigPayload, request: Request):
    require_loopback_admin(request)
    return validate_updates(_filtered_values(payload.values))


@router.post("/admin/api/config/apply")
async def apply_admin_config(
    payload: AdminConfigPayload,
    request: Request,
    background_tasks: BackgroundTasks,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    result = await services.admin.apply_admin_config(_filtered_values(payload.values))
    restart = result.get("restart")
    if isinstance(restart, dict) and restart.get("automatic"):
        background_tasks.add_task(services.admin.request_restart)
    return result


@router.get("/admin/api/status")
async def admin_status(
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    return services.admin.admin_status()


@router.get("/admin/api/providers/local-status")
async def local_provider_status(request: Request):
    require_loopback_admin(request)
    config = load_config_response()
    values = {field["key"]: field["value"] for field in config["fields"]}
    checks = []
    for provider_id, path in LOCAL_PROVIDER_PATHS.items():
        base_url = _local_provider_url(provider_id, values)
        checks.append(await _check_local_provider(provider_id, base_url, path))
    return {"providers": checks}


@router.post("/admin/api/providers/{provider_id}/test")
async def test_provider(
    provider_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    return await services.admin.test_provider(provider_id)


_CREDENTIAL_ENV_KEYS = frozenset(
    descriptor.credential_env
    for descriptor in PROVIDER_CATALOG.values()
    if descriptor.credential_env is not None
)


class _CredentialKeyAddRequest(BaseModel):
    key: str


def _mask_credential_key(key: str) -> str:
    """Return a display-safe rendering of one credential key."""
    if len(key) <= 4:
        return "****"
    if len(key) <= 10:
        return f"{key[:2]}…{key[-2:]}"
    return f"{key[:6]}…{key[-4:]}"


def _credential_entry_or_404(env_key: str) -> dict[str, Any]:
    if env_key not in _CREDENTIAL_ENV_KEYS:
        raise HTTPException(status_code=404, detail="Unknown credential env key")
    return load_value_state().get(env_key, {"value": "", "source": "default"})


def _require_unlocked_credential(entry: dict[str, Any]) -> None:
    if is_locked_source(entry["source"]):
        raise HTTPException(
            status_code=409,
            detail=(
                "This credential is set via the process environment and cannot "
                "be edited from the dashboard."
            ),
        )


@router.get("/admin/api/credentials/{env_key}/keys")
async def list_credential_keys(
    env_key: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    """List the configured keys for one provider credential (masked)."""
    require_loopback_admin(request)
    entry = _credential_entry_or_404(env_key)
    keys = parse_credential_keys(str(entry["value"]))

    # Best-effort live key health from the rotating provider, index-aligned
    # with the configured key list.
    health: list[dict[str, Any] | None] = [None] * len(keys)
    provider_id = next(
        (
            descriptor.provider_id
            for descriptor in PROVIDER_CATALOG.values()
            if descriptor.credential_env == env_key
        ),
        None,
    )
    if provider_id is not None:
        try:
            async with services.requests.acquire() as lease:
                if lease.is_provider_cached(provider_id):
                    provider = lease.resolve_provider(provider_id)
                    if isinstance(provider, RotatingProvider):
                        snapshots = provider.key_health()
                        for i in range(min(len(keys), len(snapshots))):
                            health[i] = snapshots[i]
        except Exception:
            pass  # Health is informational only; never fail the listing.

    return {
        "env_key": env_key,
        "source": entry["source"],
        "locked": is_locked_source(entry["source"]),
        "count": len(keys),
        "keys": [_mask_credential_key(key) for key in keys],
        "health": health,
    }


@router.post("/admin/api/credentials/{env_key}/keys")
async def add_credential_key(
    env_key: str,
    payload: _CredentialKeyAddRequest,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    """Append one key to a provider credential and apply immediately."""
    require_loopback_admin(request)
    entry = _credential_entry_or_404(env_key)
    _require_unlocked_credential(entry)

    new_key = payload.key.strip()
    if not new_key:
        raise HTTPException(status_code=400, detail="Key is empty")
    if "," in new_key:
        raise HTTPException(
            status_code=400, detail="Paste a single key without commas"
        )

    keys = list(parse_credential_keys(str(entry["value"])))
    if new_key in keys:
        raise HTTPException(status_code=409, detail="Key is already configured")
    keys.append(new_key)

    result = await services.admin.apply_admin_config({env_key: ",".join(keys)})
    if not result.get("applied"):
        raise HTTPException(
            status_code=400,
            detail="; ".join(result.get("errors", [])) or "Update rejected",
        )
    return {
        "applied": True,
        "env_key": env_key,
        "count": len(keys),
        "added": _mask_credential_key(new_key),
        "restart": result.get("restart"),
    }


@router.delete("/admin/api/credentials/{env_key}/keys/{index}")
async def delete_credential_key(
    env_key: str,
    index: int,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    """Remove one key from a provider credential and apply immediately."""
    require_loopback_admin(request)
    entry = _credential_entry_or_404(env_key)
    _require_unlocked_credential(entry)

    keys = list(parse_credential_keys(str(entry["value"])))
    if index < 0 or index >= len(keys):
        raise HTTPException(status_code=404, detail="Key index out of range")
    removed = keys.pop(index)

    result = await services.admin.apply_admin_config({env_key: ",".join(keys)})
    if not result.get("applied"):
        raise HTTPException(
            status_code=400,
            detail="; ".join(result.get("errors", [])) or "Update rejected",
        )
    return {
        "applied": True,
        "env_key": env_key,
        "count": len(keys),
        "removed": _mask_credential_key(removed),
        "restart": result.get("restart"),
    }


@router.get("/admin/api/models")
async def models(
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    return _model_options(services)


@router.post("/admin/api/models/refresh")
async def refresh_models(
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    result = await services.admin.refresh_models()
    return _model_options(services, refresh_result=result)


def _model_options(
    services: ApiServices,
    *,
    refresh_result: ProviderModelRefreshResult | None = None,
) -> dict[str, list[str]]:
    configured = {
        ref.model_ref
        for ref in configured_chat_model_refs(services.requests.current_settings())
    }
    discovered = {
        info.model_id for info in services.requests.cached_prefixed_model_infos()
    }
    failed_provider_ids = (
        refresh_result.failed_provider_ids if refresh_result is not None else ()
    )
    return {
        "models": sorted(configured | discovered, key=str.casefold),
        "failed_providers": list(failed_provider_ids),
    }


def _filtered_values(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if key in FIELD_BY_KEY}


def _local_provider_url(provider_id: str, values: dict[str, str]) -> str:
    if provider_id == "lmstudio":
        return values.get("LM_STUDIO_BASE_URL", "")
    if provider_id == "llamacpp":
        return values.get("LLAMACPP_BASE_URL", "")
    if provider_id == "ollama":
        return values.get("OLLAMA_BASE_URL", "")
    return ""


async def _check_local_provider(
    provider_id: str, base_url: str, path: str
) -> dict[str, Any]:
    clean_url = base_url.strip().rstrip("/")
    if not clean_url:
        return {
            "provider_id": provider_id,
            "status": "missing_url",
            "label": "Missing URL",
            "base_url": base_url,
        }

    url = f"{clean_url}{path}"
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            response = await client.get(url)
        ok = 200 <= response.status_code < 300
        return {
            "provider_id": provider_id,
            "status": "reachable" if ok else "offline",
            "label": "Reachable" if ok else "Offline",
            "base_url": base_url,
            "status_code": response.status_code,
        }
    except Exception as exc:
        return {
            "provider_id": provider_id,
            "status": "offline",
            "label": "Offline",
            "base_url": base_url,
            "error_type": type(exc).__name__,
        }
