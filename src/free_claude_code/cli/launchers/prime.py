"""Installed ``fcc-prime`` launcher for Prime Agent."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from free_claude_code.config.loader import get_settings
from free_claude_code.config.server_urls import local_proxy_root_url

from .common import preflight_proxy, run_client_process
from .ensure import ensure_prime_binary, prime_install_hint
from .openai_compat import (
    build_openai_compat_env,
    openai_compat_base_url,
    proxy_bearer_token,
)

_DISPLAY_NAME = "Prime Agent"
_API_KEY_ENV = "FCC_PRIME_API_KEY"


def launch(argv: Sequence[str] | None = None) -> None:
    """Launch Prime Agent with an ephemeral FCC OpenAI-compatible provider."""

    args = list(sys.argv[1:] if argv is None else argv)
    settings = get_settings()
    proxy_root_url = local_proxy_root_url(settings)
    if error := preflight_proxy(proxy_root_url):
        print(
            f"Free Claude Code proxy is not reachable at {proxy_root_url}: {error}",
            file=sys.stderr,
        )
        print("Start it in another terminal with: fcc-server", file=sys.stderr)
        raise SystemExit(1)

    binary_path = ensure_prime_binary()
    agent_dir = Path(tempfile.mkdtemp(prefix="fcc-prime-"))
    write_prime_agent_dir(
        agent_dir,
        proxy_root_url=proxy_root_url,
        model=settings.model,
    )
    env = build_prime_launcher_env(
        proxy_root_url=proxy_root_url,
        auth_token=settings.proxy_auth_token,
        model=settings.model,
        agent_dir=agent_dir,
        base_env=os.environ,
    )
    print(
        f"fcc-prime: using FCC provider 'fcc' model {settings.model} via {proxy_root_url}",
        file=sys.stderr,
    )
    run_client_process(
        command=build_prime_launcher_command(
            binary_path=binary_path,
            argv=args,
            model=settings.model,
        ),
        env=env,
        binary_name=Path(binary_path).name,
        display_name=_DISPLAY_NAME,
        install_hint=prime_install_hint(),
    )


def build_prime_launcher_command(
    *,
    binary_path: str,
    argv: Sequence[str],
    model: str,
) -> list[str]:
    """Return the Prime Agent command, defaulting to the FCC provider/model."""

    args = list(argv)
    extras: list[str] = []
    if not _has_flag(args, "--provider"):
        extras.extend(["--provider", "fcc"])
    if not _has_flag(args, "--model"):
        extras.extend(["--model", f"fcc/{model}"])
    return [binary_path, *extras, *args]


def build_prime_models_document(
    *,
    proxy_root_url: str,
    model: str,
    api_key_env: str = _API_KEY_ENV,
) -> dict[str, object]:
    """Return a Prime ``models.json`` that points at FCC Chat Completions.

    ``apiKey`` is an environment variable *name*, never a secret value.
    """

    return {
        "providers": {
            "fcc": {
                "baseUrl": openai_compat_base_url(proxy_root_url),
                "api": "openai-completions",
                "apiKey": api_key_env,
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                },
                "models": [
                    {
                        "id": model,
                        "name": f"FCC {model}",
                        "reasoning": False,
                        "input": ["text"],
                        "contextWindow": 128000,
                    }
                ],
            }
        }
    }


def write_prime_agent_dir(
    agent_dir: Path,
    *,
    proxy_root_url: str,
    model: str,
) -> Path:
    """Write an ephemeral Prime config dir that contains no secrets."""

    agent_dir.mkdir(parents=True, exist_ok=True)
    models_path = agent_dir / "models.json"
    models_path.write_text(
        json.dumps(
            build_prime_models_document(
                proxy_root_url=proxy_root_url,
                model=model,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return models_path


def build_prime_launcher_env(
    *,
    proxy_root_url: str,
    auth_token: str,
    model: str,
    agent_dir: Path,
    base_env: Mapping[str, str],
) -> dict[str, str]:
    """Return process env for Prime Agent. Secrets stay in memory only."""

    env = build_openai_compat_env(
        proxy_root_url=proxy_root_url,
        auth_token=auth_token,
        base_env=base_env,
        model=model,
    )
    env["PRIME_AGENT_CODING_AGENT_DIR"] = str(agent_dir)
    env[_API_KEY_ENV] = proxy_bearer_token(auth_token)
    return env


def _has_flag(args: Sequence[str], flag: str) -> bool:
    return any(item == flag or item.startswith(f"{flag}=") for item in args)
