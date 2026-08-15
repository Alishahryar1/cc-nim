"""Installed ``fcc-atomic`` launcher for Atomic Agents / Instructor clients."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request

from free_claude_code.cli.local_http import open_local_request
from free_claude_code.config.loader import get_settings
from free_claude_code.config.server_urls import local_proxy_root_url

from .common import preflight_proxy
from .openai_compat import (
    build_openai_compat_env,
    openai_compat_base_url,
    proxy_bearer_token,
)

_DISPLAY_NAME = "Atomic Agents"
_PING_TIMEOUT_SECONDS = 90.0


def launch(argv: Sequence[str] | None = None) -> None:
    """Send one Atomic Agents / Chat Completions turn through the local FCC proxy."""

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

    prompt = " ".join(args).strip() or "ping"
    model = settings.model
    auth_token = settings.proxy_auth_token
    env = build_openai_compat_env(
        proxy_root_url=proxy_root_url,
        auth_token=auth_token,
        base_env=os.environ,
        model=model,
    )
    try:
        text = run_atomic_turn(
            proxy_root_url=proxy_root_url,
            auth_token=auth_token,
            model=model,
            prompt=prompt,
            env=env,
        )
    except (HTTPError, URLError, OSError, RuntimeError, ValueError) as exc:
        print(f"{_DISPLAY_NAME} request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(text)


def run_atomic_turn(
    *,
    proxy_root_url: str,
    auth_token: str,
    model: str,
    prompt: str,
    env: Mapping[str, str] | None = None,
) -> str:
    """Run one turn with Atomic Agents when installed, else Chat Completions."""

    try:
        return run_atomic_agents_turn(
            proxy_root_url=proxy_root_url,
            auth_token=auth_token,
            model=model,
            prompt=prompt,
            env=env,
        )
    except ImportError:
        return run_chat_completions_ping(
            proxy_root_url=proxy_root_url,
            auth_token=auth_token,
            model=model,
            prompt=prompt,
        )


def run_atomic_agents_turn(
    *,
    proxy_root_url: str,
    auth_token: str,
    model: str,
    prompt: str,
    env: Mapping[str, str] | None = None,
) -> str:
    """Call Atomic Agents through Instructor against FCC ``/v1/chat/completions``."""

    import instructor
    from atomic_agents import AgentConfig, AtomicAgent, BasicChatInputSchema
    from atomic_agents.context import ChatHistory
    from openai import OpenAI

    source = os.environ if env is None else env
    client = instructor.from_openai(
        OpenAI(
            base_url=source.get("OPENAI_BASE_URL")
            or openai_compat_base_url(proxy_root_url),
            api_key=source.get("OPENAI_API_KEY") or proxy_bearer_token(auth_token),
        )
    )
    agent = AtomicAgent[BasicChatInputSchema, object](
        config=AgentConfig(
            client=client,
            model=model,
            history=ChatHistory(),
            model_api_parameters={"max_tokens": 64},
        )
    )
    response = agent.run(BasicChatInputSchema(chat_message=prompt))
    text = getattr(response, "chat_message", None)
    if isinstance(text, str) and text.strip():
        return text
    return str(response)


def run_chat_completions_ping(
    *,
    proxy_root_url: str,
    auth_token: str,
    model: str,
    prompt: str,
) -> str:
    """POST one non-streaming Chat Completions request to the local proxy."""

    url = f"{openai_compat_base_url(proxy_root_url)}/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
        }
    ).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {proxy_bearer_token(auth_token)}",
        },
        method="POST",
    )
    with open_local_request(request, timeout=_PING_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Chat Completions returned a non-object body.")
    error = payload.get("error")
    if isinstance(error, dict):
        raise RuntimeError(str(error.get("message") or error))
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Chat Completions returned no choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Chat Completions returned empty content.")
    return content
