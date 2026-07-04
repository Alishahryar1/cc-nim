#!/usr/bin/env python3
"""Concept → shots.json + asset manifest using the Seedance recipe via FCC proxy."""

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_PROXY_URL = "http://localhost:8082/v1/messages"
DEFAULT_MODEL = os.environ.get("HAILUO_MODEL", "claude-opus-4-8")
DEFAULT_MAX_TOKENS = 4096

RECIPE_PATH = Path(__file__).resolve().parent.parent / "reference" / "seedance-recipe.md"


class ArchitectError(Exception):
    """Raised when the architect cannot produce a valid shots.json."""


SHOT_SCHEMA = {
    "type": "object",
    "required": [
        "id",
        "title",
        "duration_seconds",
        "aspect_ratio",
        "resolution",
        "first_frame_asset_id",
        "prompt",
    ],
    "properties": {
        "id": {"type": "string"},
        "title": {"type": "string"},
        "duration_seconds": {"type": "number", "minimum": 1, "maximum": 120},
        "aspect_ratio": {"type": "string"},
        "resolution": {"type": "string"},
        "first_frame_asset_id": {"type": "string"},
        "prompt": {"type": "string", "minLength": 100},
        "continues_from": {"type": ["string", "null"]},
        "audio_note": {"type": "string"},
    },
}

ASSET_SCHEMA = {
    "type": "object",
    "required": ["id", "type", "description", "prompt"],
    "properties": {
        "id": {"type": "string"},
        "type": {"type": "string", "enum": ["character", "location", "prop", "first_frame"]},
        "description": {"type": "string"},
        "prompt": {"type": "string", "minLength": 50},
        "output_settings": {"type": "object"},
    },
}

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["concept", "format_mode", "assets", "shots"],
    "properties": {
        "concept": {"type": "string"},
        "format_mode": {"type": "string"},
        "assets": {"type": "array", "items": ASSET_SCHEMA},
        "shots": {"type": "array", "items": SHOT_SCHEMA},
    },
}


def load_recipe() -> str:
    if not RECIPE_PATH.exists():
        raise ArchitectError(f"Seedance recipe not found at {RECIPE_PATH}")
    return RECIPE_PATH.read_text(encoding="utf-8")


def build_system_prompt(recipe: str, max_shots: int = 8) -> str:
    return (
        "You are the architect for a Hailuo image-to-video short-film pipeline. "
        "Read the user's concept and produce a structured JSON plan for generating the film in Hailuo.\n\n"
        "Rules:\n"
        f"1. Produce at most {max_shots} shots. Warn if the concept needs more.\n"
        "2. Use the 16-section Seedance recipe below for every shot prompt AND for every standalone asset prompt.\n"
        "3. First generate recurring reference assets (hero character, hero locations, signature props). "
        "   Each asset gets a unique kebab-case id, a type (character | location | prop | first_frame), "
        "   a short description, and a full prompt using the 16-section template.\n"
        "4. Each shot references its first-frame asset by first_frame_asset_id and uses the asset ids in ACTIVE REFERENCES.\n"
        "5. Output ONLY a single JSON object matching this schema:\n"
        + json.dumps(OUTPUT_SCHEMA, indent=2)
        + "\n\n"
        "16-section Seedance recipe (canonical):\n"
        "---\n"
        + recipe
        + "\n---\n"
        "Return only the JSON object. No markdown fences, no explanation."
    )


def _parse_sse(raw: str) -> str | None:
    """Parse an Anthropic-style SSE stream into concatenated text_delta fragments.

    Returns the assembled text if the response looks like SSE, else None.
    Only text_delta deltas are collected; thinking_delta is ignored.
    """
    if "data:" not in raw and "event:" not in raw:
        return None
    parts: list[str] = []
    saw_sse = False
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        saw_sse = True
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            evt = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if evt.get("type") != "content_block_delta":
            continue
        delta = evt.get("delta", {})
        if delta.get("type") == "text_delta" and "text" in delta:
            parts.append(delta["text"])
    if not saw_sse:
        return None
    return "".join(parts)


def _extract_envelope_text(raw: str) -> str | None:
    """Parse a leading Anthropic-style JSON envelope and return its first text block.

    The FCC proxy returns a complete JSON envelope followed by a trailing SSE
    diagnostic stream. We want the envelope's text, not the SSE error text.
    Returns None if there is no parseable envelope with a text block.
    """
    # Consider only the part before the first SSE event marker.
    head = raw
    for marker in ("\nevent:", "\r\nevent:", "\ndata:"):
        idx = raw.find(marker)
        if idx != -1:
            head = head[:idx]
            break
    head = head.strip()
    if not head or not head.startswith("{"):
        return None
    try:
        envelope = json.loads(head)
    except json.JSONDecodeError:
        return None
    content = envelope.get("content", [])
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                return block["text"]
    return None


def call_proxy(system: str, user: str, proxy_url: str, model: str, max_tokens: int) -> str:
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "freecc")
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        # Prefer non-streaming; the FCC proxy may still append an SSE tail.
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        proxy_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "x-api-key": token,
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=300) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")[:500]
        raise ArchitectError(f"FCC proxy returned {e.code}: {body}")
    except URLError as e:
        raise ArchitectError(f"Cannot reach FCC proxy at {proxy_url}: {e.reason}. Start the proxy with: fcc")

    # 1. The FCC proxy returns a complete JSON envelope first; its text block is
    #    the real answer. A trailing SSE stream is a diagnostic/error tail — ignore it.
    env_text = _extract_envelope_text(raw)
    if env_text is not None:
        return env_text

    # 2. Pure SSE stream (no leading envelope): collect text_delta fragments.
    sse_text = _parse_sse(raw)
    if sse_text is not None:
        return sse_text

    # 3. Plain non-streaming envelope without SSE tail.
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        # Some proxies may return the raw text directly.
        return raw
    content = envelope.get("content", [])
    if isinstance(content, list) and content:
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
        first = content[0]
        if isinstance(first, dict):
            return first.get("text", "")
    return envelope.get("text", raw)


def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence line and trailing fence.
        text = text[3:].strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


def validate_plan(plan: dict) -> dict:
    if not isinstance(plan, dict):
        raise ArchitectError("architect output is not a JSON object")
    for key in OUTPUT_SCHEMA["required"]:
        if key not in plan:
            raise ArchitectError(f"missing top-level key: {key}")
    for i, asset in enumerate(plan.get("assets", [])):
        for key in ASSET_SCHEMA["required"]:
            if key not in asset:
                raise ArchitectError(f"asset[{i}] missing key: {key}")
    for i, shot in enumerate(plan.get("shots", [])):
        for key in SHOT_SCHEMA["required"]:
            if key not in shot:
                raise ArchitectError(f"shot[{i}] missing key: {key}")
    asset_ids = {a["id"] for a in plan["assets"]}
    for i, shot in enumerate(plan["shots"]):
        fid = shot.get("first_frame_asset_id")
        if fid and fid not in asset_ids:
            raise ArchitectError(f"shot[{i}] references unknown first_frame_asset_id: {fid}")
    return plan


def run(concept: str, *, proxy_url: str | None = None, model: str | None = None, max_tokens: int | None = None, max_shots: int = 8) -> dict:
    recipe = load_recipe()
    system = build_system_prompt(recipe, max_shots=max_shots)
    user = f"Concept: {concept}\n\nGenerate the shots.json plan."
    proxy_url = proxy_url or os.environ.get("HAILUO_PROXY_URL", DEFAULT_PROXY_URL)
    model = model or DEFAULT_MODEL
    max_tokens = max_tokens or int(os.environ.get("HAILUO_MAX_TOKENS", DEFAULT_MAX_TOKENS))

    raw = call_proxy(system, user, proxy_url, model, max_tokens)
    try:
        plan = json.loads(extract_json(raw))
    except json.JSONDecodeError as e:
        raise ArchitectError(f"architect returned invalid JSON: {e}\n---\n{raw[:1000]}")

    validate_plan(plan)
    plan.setdefault("concept", concept)
    return plan


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv:
        print("Usage: architect.py \"concept...\"", file=sys.stderr)
        return 2
    concept = " ".join(argv)
    try:
        plan = run(concept)
        print(json.dumps(plan, indent=2))
        return 0
    except ArchitectError as e:
        print(f"✗ architect: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
