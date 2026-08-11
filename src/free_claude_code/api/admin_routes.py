"""Local admin UI routes and APIs."""

import asyncio
from collections.abc import Mapping
from pathlib import Path

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
)
from free_claude_code.application.model_metadata import ProviderModelRefreshResult
from free_claude_code.config.admin.manifest import FIELD_BY_KEY
from free_claude_code.config.admin.values import load_config_response, load_value_state
from free_claude_code.config.model_refs import configured_chat_model_refs
from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    ProviderAuthKind,
)
from free_claude_code.core.json_types import JsonObject, JsonValue
from free_claude_code.core.openai_base_url import openai_v1_base_url
from free_claude_code.core.version import package_version

from .admin_security import require_loopback_admin
from .dependencies import get_services
from .ports import ApiServices

router = APIRouter()

STATIC_DIR = Path(__file__).resolve().parent / "admin_static"
PACKAGE_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_ADMIN_ASSET_VERSION_PLACEHOLDER = "__FCC_VERSION__"
_ADMIN_ASSET_FILENAMES = frozenset(
    {
        "admin.css",
        "admin.js",
        "app-icon.svg",
        "chat_sessions.css",
        "chat_sessions.js",
        "model_combobox.js",
    }
)
LOCAL_PROVIDER_PATHS = {
    "lmstudio": "/models",
    "llamacpp": "/models",
    "ollama": "/api/tags",
    "custom": "/models",
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


def _asset_path(filename: str) -> Path:
    asset_dir = PACKAGE_ASSETS_DIR if filename == "app-icon.svg" else STATIC_DIR
    path = asset_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Admin asset not found")
    return path


def _asset_response(filename: str) -> FileResponse:
    return FileResponse(_asset_path(filename))


def admin_page_response() -> HTMLResponse:
    template = _asset_path("index.html").read_text(encoding="utf-8")
    return HTMLResponse(
        template.replace(_ADMIN_ASSET_VERSION_PLACEHOLDER, package_version())
    )


@router.get("/admin", include_in_schema=False)
def admin_page(request: Request):
    require_loopback_admin(request)
    return admin_page_response()


@router.get("/admin/assets/{version}/{filename}", include_in_schema=False)
async def admin_asset(version: str, filename: str, request: Request):
    require_loopback_admin(request)
    if version != package_version() or filename not in _ADMIN_ASSET_FILENAMES:
        raise HTTPException(status_code=404, detail="Admin asset not found")
    return _asset_response(filename)


@router.get("/admin/api/config")
async def get_admin_config(request: Request):
    require_loopback_admin(request)
    return load_config_response()


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

    async def _probe(provider_id: str, path: str) -> JsonObject | None:
        base_url = _local_provider_url(provider_id, values)
        if provider_id == "custom" and not base_url.strip():
            # The custom endpoint only appears once a base URL is configured.
            return None
        api_key = values.get("CUSTOM_API_KEY", "") if provider_id == "custom" else ""
        proxy = values.get("CUSTOM_PROXY", "") if provider_id == "custom" else ""
        return await _check_local_provider(
            provider_id, base_url, path, api_key=api_key, proxy=proxy
        )

    checks = [
        check
        for check in await asyncio.gather(
            *(
                _probe(provider_id, path)
                for provider_id, path in LOCAL_PROVIDER_PATHS.items()
            )
        )
        if check is not None
    ]
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
    if provider_id == "custom":
        # The runtime's custom profile normalizes the configured URL to its
        # OpenAI ``/v1`` API base, so probe the same base the runtime uses.
        configured = values.get("CUSTOM_BASE_URL", "").strip()
        return openai_v1_base_url(configured) if configured else ""
    return ""


async def _check_local_provider(
    provider_id: str,
    base_url: str,
    path: str,
    *,
    api_key: str = "",
    proxy: str = "",
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
    headers = {"Authorization": f"Bearer {api_key.strip()}"} if api_key.strip() else {}
    try:
        # The runtime routes custom provider traffic through CUSTOM_PROXY, so
        # the status probe must use the same transport to agree with it.
        async with httpx.AsyncClient(
            timeout=1.5, proxy=proxy.strip() or None
        ) as client:
            # Only attach headers when a credential is configured — keyless
            # local providers must not receive an Authorization header at all.
            if headers:
                response = await client.get(url, headers=headers)
            else:
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
