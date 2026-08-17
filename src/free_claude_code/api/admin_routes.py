"""Local admin UI routes and APIs."""

import ipaddress
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
)
from free_claude_code.application.model_metadata import ProviderModelRefreshResult
from free_claude_code.config.admin.manifest import FIELD_BY_KEY
from free_claude_code.config.admin.persistence import validate_updates
from free_claude_code.config.admin.values import load_config_response, load_value_state
from free_claude_code.config.model_refs import configured_chat_model_refs
from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    ProviderAuthKind,
)
from free_claude_code.core.json_types import JsonObject, JsonValue

from .dependencies import get_services
from .ports import ApiServices

router = APIRouter()

STATIC_DIR = Path(__file__).resolve().parent / "admin_static"
LOCAL_PROVIDER_PATHS = {
    "lmstudio": "/models",
    "llamacpp": "/models",
    "ollama": "/api/tags",
}
_LOCAL_PROVIDER_CHECK_FAILURE_MESSAGE = (
    "Could not connect. Verify the URL and that the local provider is running."
)


class AdminConfigPayload(BaseModel):
    """Partial config update submitted by the admin UI."""

    values: JsonObject = Field(default_factory=dict)


class ConnectedAccountLoginPayload(BaseModel):
    """Interactive connected-account login selection."""

    mode: ConnectedAccountLoginMode = ConnectedAccountLoginMode.BROWSER


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
    values = {key: entry.value or "" for key, entry in load_value_state().items()}
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


@router.get("/admin/api/providers/{provider_id}/auth")
async def connected_account_status(
    provider_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    _require_connected_account_provider(provider_id)
    status = await services.admin.connected_account_status(provider_id)
    return _no_store(status.as_dict())


@router.post("/admin/api/providers/{provider_id}/auth/login")
async def start_connected_account_login(
    provider_id: str,
    payload: ConnectedAccountLoginPayload,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    _require_connected_account_provider(provider_id)
    try:
        status = await services.admin.start_connected_account_login(
            provider_id, payload.mode
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(f"Could not start connected-account login ({type(exc).__name__})."),
        ) from exc
    return _no_store(status.as_dict())


@router.post("/admin/api/providers/{provider_id}/auth/cancel")
async def cancel_connected_account_login(
    provider_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    _require_connected_account_provider(provider_id)
    status = await services.admin.cancel_connected_account_login(provider_id)
    return _no_store(status.as_dict())


@router.delete("/admin/api/providers/{provider_id}/auth")
async def disconnect_connected_account(
    provider_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    _require_connected_account_provider(provider_id)
    status = await services.admin.disconnect_connected_account(provider_id)
    return _no_store(status.as_dict())


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


def _filtered_values(values: Mapping[str, JsonValue]) -> JsonObject:
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
) -> JsonObject:
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
        logger.debug(
            "Admin local provider check failed: provider={} exc_type={}",
            provider_id,
            type(exc).__name__,
        )
        return {
            "provider_id": provider_id,
            "status": "offline",
            "label": "Offline",
            "base_url": base_url,
            "message": _LOCAL_PROVIDER_CHECK_FAILURE_MESSAGE,
        }


def _require_connected_account_provider(provider_id: str) -> None:
    descriptor = PROVIDER_CATALOG.get(provider_id)
    if (
        descriptor is None
        or descriptor.auth_kind is not ProviderAuthKind.CONNECTED_ACCOUNT
    ):
        raise HTTPException(
            status_code=404,
            detail="Provider does not support connected-account login.",
        )


def _no_store(payload: JsonValue) -> JSONResponse:
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})
