"""Installed `fcc-codex` launcher."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from urllib.request import Request, urlopen

from api.admin_urls import local_proxy_root_url
from config.paths import codex_model_catalog_path
from config.settings import Settings, get_settings

from .codex_model_catalog import build_codex_model_catalog, write_codex_model_catalog
from .common import (
    PROXY_PREFLIGHT_TIMEOUT_SECONDS,
    preflight_proxy,
    resolve_client_binary,
    run_client_process,
)

_CODEX_AUTH_ENV_KEY = "FCC_CODEX_API_KEY"
_DISPLAY_NAME = "Codex CLI"
_DEFAULT_BINARY = "codex"
_INSTALL_HINT = "Install Codex with: npm install -g @openai/codex"
_STRIPPED_CODEX_ENV_KEYS = frozenset(
    {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_ORG_ID",
        "OPENAI_ORGANIZATION",
        "CODEX_API_KEY",
        _CODEX_AUTH_ENV_KEY,
    }
)


def launch(argv: Sequence[str] | None = None) -> None:
    """Launch Codex CLI with Free Claude Code proxy configuration."""

    settings = get_settings()
    proxy_root_url = local_proxy_root_url(settings)
    if error := preflight_proxy(proxy_root_url):
        print(
            f"Free Claude Code proxy is not reachable at {proxy_root_url}: {error}",
            file=sys.stderr,
        )
        print("Start it in another terminal with: fcc-server", file=sys.stderr)
        raise SystemExit(1)

    binary_name = codex_binary_name()
    binary_path = resolve_client_binary(
        binary_name=binary_name,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )
    catalog_args = codex_model_catalog_config_args(proxy_root_url, settings)
    args = list(sys.argv[1:] if argv is None else argv)
    run_client_process(
        command=build_codex_launcher_command(
            binary_path=binary_path,
            argv=args,
            settings=settings,
            proxy_root_url=proxy_root_url,
            catalog_config_args=catalog_args,
        ),
        env=build_codex_launcher_env(
            auth_token=settings.anthropic_auth_token,
            base_env=os.environ,
        ),
        binary_name=binary_name,
        display_name=_DISPLAY_NAME,
        install_hint=_INSTALL_HINT,
    )


def codex_binary_name() -> str:
    """Return the Codex CLI binary name."""

    return _DEFAULT_BINARY


def build_codex_launcher_command(
    *,
    binary_path: str,
    argv: Sequence[str],
    settings: Settings,
    proxy_root_url: str,
    catalog_config_args: Sequence[str] = (),
) -> list[str]:
    """Return a Codex command with ephemeral FCC provider config."""

    return [
        binary_path,
        *catalog_config_args,
        *codex_config_args(
            api_url=_ensure_v1_url(proxy_root_url),
            model=getattr(settings, "model", None),
        ),
        *argv,
    ]


def build_codex_launcher_env(
    *,
    auth_token: str,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    """Return a Codex environment that targets the local proxy provider."""

    env = {
        key: value
        for key, value in base_env.items()
        if key not in _STRIPPED_CODEX_ENV_KEYS and not key.startswith("OPENAI_")
    }
    env[_CODEX_AUTH_ENV_KEY] = auth_token.strip() or "fcc-no-auth"
    return env


def codex_model_catalog_config_args(
    proxy_root_url: str, settings: Settings
) -> list[str]:
    """Prepare the generated Codex model catalog and return its config args."""

    try:
        models_response = fetch_proxy_models_response(
            proxy_root_url, settings.anthropic_auth_token
        )
        catalog = build_codex_model_catalog(models_response)
        models = catalog.get("models")
        if not isinstance(models, list) or not models:
            print(
                "Free Claude Code warning: Codex model catalog is empty; "
                "launching without model picker catalog.",
                file=sys.stderr,
            )
            return []
        catalog_path = codex_model_catalog_path()
        write_codex_model_catalog(catalog_path, catalog)
    except Exception as exc:
        print(
            "Free Claude Code warning: could not prepare Codex model catalog "
            f"({exc}); launching without model picker catalog.",
            file=sys.stderr,
        )
        return []

    return build_model_catalog_config_args(str(catalog_path))


def fetch_proxy_models_response(
    proxy_root_url: str, auth_token: str
) -> dict[str, object]:
    """Fetch the local proxy `/v1/models` response for Codex catalog generation."""

    url = f"{proxy_root_url.rstrip('/')}/v1/models"
    headers: dict[str, str] = {}
    if token := auth_token.strip():
        headers["X-API-Key"] = token

    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=PROXY_PREFLIGHT_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("model list response was not a JSON object")
    return payload


def build_model_catalog_config_args(catalog_path: str) -> list[str]:
    """Return Codex config args for a generated model catalog."""

    return ["-c", _toml_assignment("model_catalog_json", catalog_path)]


def codex_config_args(*, api_url: str, model: str | None = None) -> list[str]:
    """Return Codex `-c` assignments for the ephemeral FCC provider."""

    args = [
        "-c",
        _toml_assignment("model_provider", "fcc"),
        "-c",
        _toml_assignment("model_providers.fcc.name", "Free Claude Code"),
        "-c",
        _toml_assignment("model_providers.fcc.base_url", _ensure_v1_url(api_url)),
        "-c",
        _toml_assignment("model_providers.fcc.env_key", _CODEX_AUTH_ENV_KEY),
        "-c",
        _toml_assignment("model_providers.fcc.wire_api", "responses"),
    ]
    if model:
        args.extend(["-c", _toml_assignment("model", model)])
    return args


def _ensure_v1_url(url: str) -> str:
    stripped = url.rstrip("/")
    return stripped if stripped.endswith("/v1") else f"{stripped}/v1"


def _toml_assignment(key: str, value: str) -> str:
    return f"{key}={json.dumps(value)}"
