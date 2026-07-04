# `/hailuo-film` Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a `/hailuo-film` Claude Code skill that adapts the Seedance 4K recipe to Hailuo — the FCC-proxied text model architects a 16-section prompt per shot, Chrome DevTools MCP drives Hailuo's web UI to generate assets + clips, and a project folder holds resumable state.

**Architecture:** Three phases. (1) `architect.py` calls the FCC proxy `/v1/messages` to turn a concept into `shots.json` (shot list + per-shot 16-section prompts + asset manifest). (2) The skill (via Chrome DevTools MCP, user signs in once) drives Hailuo image-gen to produce the asset library. (3) The skill drives Hailuo image-gen for each shot's first-frame, then Hailuo I2V to animate it, downloading clips. A `progress.json` makes every run resumable. Output handoff is `manifest.md` for the existing `reel`/`hyperframes` skills.

**Tech Stack:** Python 3.14 stdlib only (`urllib.request`, `json`, `pathlib`, `dataclasses`) — no new dependencies, no `pyproject.toml` changes. pytest for unit tests (already in repo). Chrome DevTools MCP for browser driving (no script imports it — the SKILL.md instructs Claude to call those MCP tools at runtime, reading the playwright/selector docs the skill ships).

## Global Constraints

- **No new dependencies.** Use only Python 3.14 stdlib in `architect.py` / `project.py`. Do not edit `pyproject.toml` (skill scripts are not imported by the fcc proxy package; they are standalone tools the skill runs via `uv run`).
- **Run commands with `uv run`.** Per repo AGENTS.md: `uv run python ...`, `uv run pytest ...`, `uv run ruff format`, `uv run ruff check --fix`, `uv run ty check`.
- **No version bump required.** Skill files under `.claude/skills/` are not production files of the fcc proxy; `tests/` never requires a bump. Do not touch `pyproject.toml` version.
- **No secrets.** The skill uses `ANTHROPIC_AUTH_TOKEN=freecc` (the local proxy token) and the local proxy URL `http://localhost:8082`. Never read, print, or store Hailuo credentials. The sign-in gate is interactive and human-only.
- **Skill location:** `.claude/skills/hailuo-film/`.
- **Project folder default:** `~/hailuo-projects/<slug>-<timestamp>/`.
- **FCC proxy contract:** `POST http://localhost:8082/v1/messages` with header `x-api-key: freecc`, body `{"model": <model-or-omit>, "messages": [{"role":"user","content":<prompt>}], "max_tokens": 8192}`. Response is the Anthropic messages shape: `{"content": [{"type":"text","text": "..."}]}`.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `.claude/skills/hailuo-film/SKILL.md` | Skill entry: frontmatter, when-to-invoke, the 3-phase workflow, sign-in gate, smoke test. Embeds nothing heavy — points at `reference/` and `scripts/`. |
| `.claude/skills/hailuo-film/reference/seedance-recipe.md` | Canonical 16-section prompt template + asset rules. One source of truth; `architect.py` reads it at runtime; SKILL.md references it. |
| `.claude/skills/hailuo-film/scripts/project.py` | Project folder scaffolding + `progress.json` resume helpers. Pure stdlib. |
| `.claude/skills/hailuo-film/scripts/finish.py` | CLI: `finish.py <project_path>` loads `shots.json` + `progress.json`, calls `write_manifest`, prints the manifest path. Pure stdlib. |
| `.claude/skills/hailuo-film/scripts/architect.py` | Concept → `shots.json` via FCC proxy. Builds the architect prompt, calls `/v1/messages`, parses + validates JSON. Pure stdlib. |
| `.claude/skills/hailuo-film/scripts/hailuo_driver.md` | Browser playbook Claude reads at runtime — step-by-step for Hailuo image-gen and I2V. |
| `.claude/skills/hailuo-film/scripts/selectors.json` | UI selector map (text labels / a11y roles). Template skeleton; finalized during the manual smoke test. |
| `.claude/skills/hailuo-film/tests/test_project.py` | Unit tests for `project.py`. |
| `.claude/skills/hailuo-film/tests/test_architect.py` | Unit tests for `architect.py` (mocked HTTP via monkeypatch). |

---

## Task 1: `project.py` — project folder + resumable progress

**Files:**
- Create: `.claude/skills/hailuo-film/scripts/project.py`
- Create: `.claude/skills/hailuo-film/tests/test_project.py`

**Interfaces:**
- Produces: `Project` dataclass with fields `path`, `shots_json`, `assets_dir`, `clips_dir`, `progress_json`, `manifest_md` (all `pathlib.Path`). Functions `create_project(slug, base=None) -> Project`, `load_progress(project) -> dict`, `save_progress(project, progress) -> None`, `mark_asset(progress, asset_id, status, error=None) -> dict`, `mark_shot(progress, shot_id, status, error=None) -> dict`, `next_pending(progress, kind) -> str | None`, `write_manifest(project, shots_data, progress) -> Path`. `kind` is `"asset"` or `"shot"`. `status` is one of `"pending" | "done" | "failed"`.

- [ ] **Step 1: Write the failing tests**

Create `.claude/skills/hailuo-film/tests/test_project.py`:

```python
"""Unit tests for project.py — folder scaffolding + progress.json resume logic."""
from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from project import (
    Project,
    create_project,
    load_progress,
    save_progress,
    mark_asset,
    mark_shot,
    next_pending,
    write_manifest,
)


def test_create_project_scaffolds_folders(tmp_path: Path):
    proj = create_project("my-film", base=tmp_path)
    assert isinstance(proj, Project)
    assert proj.path.is_dir()
    assert proj.assets_dir.is_dir()
    assert proj.clips_dir.is_dir()
    assert proj.path.name.startswith("my-film-")
    # shots.json path exists but file not written yet
    assert not proj.shots_json.exists()
    assert proj.progress_json == proj.path / "progress.json"
    assert proj.manifest_md == proj.path / "manifest.md"


def test_load_progress_empty_when_missing(tmp_path: Path):
    proj = create_project("film", base=tmp_path)
    assert load_progress(proj) == {"assets": {}, "shots": {}}


def test_save_then_load_progress_roundtrip(tmp_path: Path):
    proj = create_project("film", base=tmp_path)
    progress = {"assets": {"char": {"status": "done"}}, "shots": {}}
    save_progress(proj, progress)
    assert load_progress(proj) == progress


def test_mark_asset_sets_status_and_error(tmp_path: Path):
    p = {"assets": {}, "shots": {}}
    p = mark_asset(p, "char", "failed", error="rate limit")
    assert p["assets"]["char"] == {"status": "failed", "error": "rate limit"}
    p = mark_asset(p, "loc", "done")
    assert p["assets"]["loc"] == {"status": "done"}


def test_mark_shot_sets_status(tmp_path: Path):
    p = {"assets": {}, "shots": {}}
    p = mark_shot(p, "shot-01", "done")
    assert p["shots"]["shot-01"] == {"status": "done"}


def test_next_pending_returns_first_pending_then_none(tmp_path: Path):
    p = {"assets": {"a1": {"status": "done"}, "a2": {"status": "pending"}, "a3": {"status": "pending"}}, "shots": {}}
    assert next_pending(p, "asset") == "a2"
    p = mark_asset(p, "a2", "done")
    assert next_pending(p, "asset") == "a3"
    p = mark_asset(p, "a3", "done")
    assert next_pending(p, "asset") is None


def test_next_pending_distinct_kinds(tmp_path: Path):
    p = {"assets": {"a1": {"status": "pending"}}, "shots": {"s1": {"status": "pending"}}}
    assert next_pending(p, "asset") == "a1"
    assert next_pending(p, "shot") == "s1"


def test_write_manifest_creates_markdown(tmp_path: Path):
    proj = create_project("film", base=tmp_path)
    shots_data = {
        "concept": "a cat on a couch",
        "asset_manifest": [{"id": "char-cat", "type": "character", "prompt": "orange tabby"}],
        "shots": [
            {"id": "shot-01", "order": 1, "duration_sec": 5, "sections": {"SCENE_CONTEXT": "x"}, "asset_refs": ["char-cat"]},
        ],
    }
    progress = {"assets": {"char-cat": {"status": "done"}}, "shots": {"shot-01": {"status": "done"}}}
    path = write_manifest(proj, shots_data, progress)
    assert path == proj.manifest_md
    text = path.read_text()
    assert "shot-01" in text
    assert "char-cat" in text
    assert "clips/shot-01.mp4" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest .claude/skills/hailuo-film/tests/test_project.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'project'`

- [ ] **Step 3: Write minimal implementation**

Create `.claude/skills/hailuo-film/scripts/project.py`:

```python
"""Project folder scaffolding + progress.json resume helpers for /hailuo-film.

Pure stdlib. No dependencies on the fcc proxy package.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_BASE = Path.home() / "hailuo-projects"

VALID_STATUSES = ("pending", "done", "failed")


@dataclass(frozen=True)
class Project:
    path: Path
    shots_json: Path
    assets_dir: Path
    clips_dir: Path
    progress_json: Path
    manifest_md: Path


def create_project(slug: str, base: Path | None = None) -> Project:
    """Create a new project folder `<base>/<slug>-<timestamp>/` with subdirs."""
    base_dir = base if base is not None else DEFAULT_BASE
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = Path(slug).as_posix().strip("/").replace("/", "-") or "film"
    path = base_dir / f"{slug}-{stamp}"
    # If the exact timestamp collides, append a counter.
    counter = 1
    while path.exists():
        path = base_dir / f"{slug}-{stamp}-{counter}"
        counter += 1
    assets_dir = path / "assets"
    clips_dir = path / "clips"
    assets_dir.mkdir(parents=True)
    clips_dir.mkdir(parents=True)
    return Project(
        path=path,
        shots_json=path / "shots.json",
        assets_dir=assets_dir,
        clips_dir=clips_dir,
        progress_json=path / "progress.json",
        manifest_md=path / "manifest.md",
    )


def load_progress(project: Project) -> dict:
    """Return progress dict; empty skeleton if file missing."""
    if not project.progress_json.exists():
        return {"assets": {}, "shots": {}}
    data = json.loads(project.progress_json.read_text())
    data.setdefault("assets", {})
    data.setdefault("shots", {})
    return data


def save_progress(project: Project, progress: dict) -> None:
    project.progress_json.write_text(json.dumps(progress, indent=2))


def _mark(bucket: dict, item_id: str, status: str, error: str | None) -> dict:
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}, got {status!r}")
    entry: dict = {"status": status}
    if error is not None:
        entry["error"] = error
    bucket[item_id] = entry
    return bucket


def mark_asset(progress: dict, asset_id: str, status: str, error: str | None = None) -> dict:
    progress["assets"] = progress.get("assets", {})
    return {"assets": _mark(progress["assets"], asset_id, status, error), "shots": progress.get("shots", {})}


def mark_shot(progress: dict, shot_id: str, status: str, error: str | None = None) -> dict:
    progress["shots"] = progress.get("shots", {})
    return {"assets": progress.get("assets", {}), "shots": _mark(progress["shots"], shot_id, status, error)}


def next_pending(progress: dict, kind: str) -> str | None:
    """Return the first item id with status 'pending' in the given bucket, or None.

    kind is "asset" or "shot".
    """
    if kind not in ("asset", "shot"):
        raise ValueError(f"kind must be 'asset' or 'shot', got {kind!r}")
    bucket = progress.get("assets" if kind == "asset" else "shots", {})
    for item_id, entry in bucket.items():
        if entry.get("status") == "pending":
            return item_id
    return None


def write_manifest(project: Project, shots_data: dict, progress: dict) -> Path:
    """Write manifest.md — the human-readable handoff doc for reel/hyperframes."""
    lines: list[str] = []
    lines.append(f"# Film manifest — {project.path.name}")
    lines.append("")
    lines.append(f"**Concept:** {shots_data.get('concept', '')}")
    lines.append("")
    lines.append("## Asset library")
    for asset in shots_data.get("asset_manifest", []):
        status = progress.get("assets", {}).get(asset["id"], {}).get("status", "pending")
        lines.append(f"- `{asset['id']}` ({asset.get('type', 'asset')}) — {status} — `assets/{asset['id']}.png`")
    lines.append("")
    lines.append("## Shots")
    for shot in sorted(shots_data.get("shots", []), key=lambda s: s.get("order", 0)):
        sid = shot["id"]
        status = progress.get("shots", {}).get(sid, {}).get("status", "pending")
        lines.append(f"- {sid} (order {shot.get('order')}, {shot.get('duration_sec')}s) — {status} — `clips/{sid}.mp4`")
    lines.append("")
    lines.append("## Handoff")
    lines.append("Assemble with `reel` (FFmpeg NLE) or `hyperframes`. Shot order is given above.")
    project.manifest_md.write_text("\n".join(lines))
    return project.manifest_md
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest .claude/skills/hailuo-film/tests/test_project.py -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Lint + typecheck**

Run: `uv run ruff format .claude/skills/hailuo-film/scripts/project.py .claude/skills/hailuo-film/tests/test_project.py && uv run ruff check --fix .claude/skills/hailuo-film && uv run ty check .claude/skills/hailuo-film/scripts/project.py`
Expected: clean (or auto-fixed). Re-run pytest if ruff changed formatting.

- [ ] **Step 6: Add `finish.py` CLI + test**

Create `.claude/skills/hailuo-film/scripts/finish.py`:

```python
"""finish.py — write manifest.md for a project after generation completes.

Usage: uv run python .claude/skills/hailuo-film/scripts/finish.py <project_path>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from project import Project, write_manifest


def _project_from_path(path: Path) -> Project:
    return Project(
        path=path,
        shots_json=path / "shots.json",
        assets_dir=path / "assets",
        clips_dir=path / "clips",
        progress_json=path / "progress.json",
        manifest_md=path / "manifest.md",
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: finish.py <project_path>", file=sys.stderr)
        return 2
    path = Path(argv[1]).expanduser().resolve()
    if not path.is_dir():
        print(f"not a directory: {path}", file=sys.stderr)
        return 1
    proj = _project_from_path(path)
    shots_data = json.loads(proj.shots_json.read_text())
    progress = json.loads(proj.progress_json.read_text()) if proj.progress_json.exists() else {"assets": {}, "shots": {}}
    out = write_manifest(proj, shots_data, progress)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

Add to `.claude/skills/hailuo-film/tests/test_project.py`:

```python
def test_finish_cli_writes_manifest(tmp_path: Path, monkeypatch, capsys):
    import finish
    proj = create_project("film", base=tmp_path)
    shots_data = {
        "concept": "a cat on a couch",
        "asset_manifest": [{"id": "char-cat", "type": "character", "prompt": "orange tabby"}],
        "shots": [
            {"id": "shot-01", "order": 1, "duration_sec": 5, "sections": {"SCENE_CONTEXT": "x"}, "asset_refs": ["char-cat"]},
        ],
    }
    proj.shots_json.write_text(json.dumps(shots_data))
    rc = finish.main(["finish.py", str(proj.path)])
    assert rc == 0
    assert proj.manifest_md.exists()
    out = capsys.readouterr().out.strip()
    assert str(proj.manifest_md) in out
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest .claude/skills/hailuo-film/tests/test_project.py -v`
Expected: all 9 tests PASS (8 prior + finish CLI).

- [ ] **Step 8: Lint + typecheck**

Run: `uv run ruff format .claude/skills/hailuo-film/scripts/finish.py && uv run ruff check --fix .claude/skills/hailuo-film && uv run ty check .claude/skills/hailuo-film/scripts/finish.py`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add .claude/skills/hailuo-film/scripts/project.py .claude/skills/hailuo-film/scripts/finish.py .claude/skills/hailuo-film/tests/test_project.py
git commit -m "feat(hailuo-film): project.py + finish.py — folder scaffolding, resumable progress, manifest CLI"
```

---

## Task 2: `reference/seedance-recipe.md` — canonical recipe

**Files:**
- Create: `.claude/skills/hailuo-film/reference/seedance-recipe.md`

**Interfaces:**
- Produces: a markdown doc that `architect.py` reads verbatim and injects into the architect prompt. The doc must define (a) the 16 section keys in order, (b) a one-line description of what each section contains, (c) the asset-library rules, (d) the required output JSON schema.

- [ ] **Step 1: Write the recipe doc**

Create `.claude/skills/hailuo-film/reference/seedance-recipe.md`:

````markdown
# Seedance 4K recipe — adapted for Hailuo

This is the canonical prompt-architecture recipe the `/hailuo-film` architect step uses. It is adapted from the Higgsfield Seedance 4K breakdown and is model-agnostic: it works for any diffusion video model (we drive Hailuo's image + I2V via browser).

## Asset-first principle

Before generating any shot, build a **shared asset library** so recurring entities stay consistent across shots:

- **Character reference sheet** — one image per recurring character: a clear portrait/turnaround showing face, wardrobe, build. Used as an image input on every shot featuring that character.
- **Location plates** — one image per recurring location, shot wide and neutral.
- **Key props** — one image per prop that must stay identical across shots.

Asset library rules:
- Each asset gets a stable `id` (e.g. `char-main`, `loc-livingroom`, `prop-tv-remote`).
- Each asset gets its own generation prompt (descriptive, lit neutral, no action).
- Asset images are generated first and reused as image inputs on every shot that references them.

## The 16-section per-shot prompt

Every shot is described as a single JSON object with exactly these 16 keys, in this order. Every key must be present and non-empty.

1. `SCENE_CONTEXT` — one paragraph: who is in the shot, where, what is happening, the dramatic beat.
2. `ACTIVE_REFERENCES` — which library asset ids appear in this shot (e.g. `["char-main", "loc-livingroom"]`).
3. `LOCATION_MAP` — where the camera is relative to the subject and the room; spatial blocking in words.
4. `FIRST_FRAME_BLOCKING` — the exact composition of the first frame: subject position, gaze direction, framing (close/medium/wide), headroom. This is used to generate the shot's first-frame image.
5. `FORMAT_MODE` — aspect ratio + resolution target (e.g. `16:9, 4K, 24fps`).
6. `OPTICS` — lens choice, depth of field, focus (e.g. `35mm, shallow DOF, focus on subject's eyes`).
7. `CAMERA` — camera movement across the shot (e.g. `slow push-in from medium to close over 5s, handheld micro-drift`).
8. `ACTION` — what physically happens during the shot, beat by beat, tied to time.
9. `PERFORMANCE` — the subject's emotional/physical performance (expression, gesture, breathing).
10. `PHYSICS` — physical rules that must hold (gravity, fabric, hair, reflections, object permanence).
11. `LIGHTING` — light sources, direction, quality, time of day, motivated or practical sources.
12. `COLOR_GRADE` — palette, film stock emulation, contrast, saturation intent.
13. `AUDIO` — diegetic + non-diegetic sound cues for the shot (used at assembly, not generation; still write it).
14. `STYLE` — the overall aesthetic reference (e.g. `shot on Arri Alexa, anamorphic, naturalistic`).
15. `OUTPUT_SETTINGS` — generation parameters as text (e.g. `5s duration, high motion coherence, no text overlays`).
16. `POSITIVE_LOCKS` — a list of must-keep invariants the model should lock: `[subject stays on-model, wardrobe does not change, single continuous take]`.

## Required output JSON schema

The architect must emit a single JSON object (no prose, no code fences) of this shape:

```json
{
  "concept": "<the original concept, echoed>",
  "asset_manifest": [
    {"id": "char-main", "type": "character", "prompt": "<generation prompt for this asset image>"}
  ],
  "shots": [
    {
      "id": "shot-01",
      "order": 1,
      "duration_sec": 5,
      "asset_refs": ["char-main"],
      "sections": {
        "SCENE_CONTEXT": "...",
        "ACTIVE_REFERENCES": "...",
        "LOCATION_MAP": "...",
        "FIRST_FRAME_BLOCKING": "...",
        "FORMAT_MODE": "...",
        "OPTICS": "...",
        "CAMERA": "...",
        "ACTION": "...",
        "PERFORMANCE": "...",
        "PHYSICS": "...",
        "LIGHTING": "...",
        "COLOR_GRADE": "...",
        "AUDIO": "...",
        "STYLE": "...",
        "OUTPUT_SETTINGS": "...",
        "POSITIVE_LOCKS": "..."
      }
    }
  ]
}
```

Rules:
- Shot ids are `shot-01`, `shot-02`, … in `order` sequence.
- `asset_refs` must only contain ids that exist in `asset_manifest`.
- Every shot must contain all 16 section keys, each non-empty.
- `duration_sec` between 3 and 10 (Hailuo clip length limits).
- If the concept implies more than ~12 shots, warn in a top-level `_warning` field and still emit the shots.
````

- [ ] **Step 2: Verify the file is valid markdown**

Run: `uv run python -c "from pathlib import Path; t=Path('.claude/skills/hailuo-film/reference/seedance-recipe.md').read_text(); assert 'POSITIVE_LOCKS' in t and 'asset_manifest' in t and len(t) > 500; print('ok', len(t))"`
Expected: `ok <number>`

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/hailuo-film/reference/seedance-recipe.md
git commit -m "docs(hailuo-film): canonical Seedance 16-section recipe + asset rules"
```

---

## Task 3: `architect.py` — concept → shots.json via FCC proxy

**Files:**
- Create: `.claude/skills/hailuo-film/scripts/architect.py`
- Create: `.claude/skills/hailuo-film/tests/test_architect.py`

**Interfaces:**
- Consumes: `Project` and `write_manifest`-free `Project.shots_json` path from `project.py` (Task 1); the recipe text from `reference/seedance-recipe.md` (Task 2).
- Produces: `build_architect_prompt(concept: str, recipe: str) -> str`, `call_fcc(prompt: str, model: str | None = None, base_url: str = "http://localhost:8082", auth_token: str = "freecc", timeout: float = 120.0) -> str`, `parse_shots(response_text: str) -> dict`, `architect(concept: str, project: Project, model: str | None = None, recipe_path: Path | None = None) -> Path`. `architect()` writes `project.shots_json` and seeds `progress.json` with every asset and shot marked `pending`, then returns `project.shots_json`.

- [ ] **Step 1: Write the failing tests**

Create `.claude/skills/hailuo-film/tests/test_architect.py`:

```python
"""Unit tests for architect.py — prompt building, HTTP call (mocked), JSON parsing."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from architect import build_architect_prompt, call_fcc, parse_shots, architect
from project import create_project

RECIPE = """# Recipe
## 16-section
1. SCENE_CONTEXT
... (16 total) ...
16. POSITIVE_LOCKS
## schema
asset_manifest + shots with sections.
"""

GOOD_RESPONSE = json.dumps({
    "concept": "a cat on a couch",
    "asset_manifest": [{"id": "char-cat", "type": "character", "prompt": "orange tabby"}],
    "shots": [
        {
            "id": "shot-01", "order": 1, "duration_sec": 5, "asset_refs": ["char-cat"],
            "sections": {f"SEC_{i}": "x" for i in range(16)},
        }
    ],
})

# Map the 16 real section names the parser requires onto the good response.
from architect import REQUIRED_SECTIONS
GOOD_RESPONSE_OBJ = json.loads(GOOD_RESPONSE)
GOOD_RESPONSE_OBJ["shots"][0]["sections"] = {name: "x" for name in REQUIRED_SECTIONS}
GOOD_RESPONSE = json.dumps(GOOD_RESPONSE_OBJ)


def test_build_architect_prompt_contains_concept_and_recipe():
    prompt = build_architect_prompt("a cat on a couch", RECIPE)
    assert "a cat on a couch" in prompt
    assert "POSITIVE_LOCKS" in prompt
    assert "JSON" in prompt


def test_parse_shots_accepts_well_formed():
    data = parse_shots(GOOD_RESPONSE)
    assert data["concept"] == "a cat on a couch"
    assert data["asset_manifest"][0]["id"] == "char-cat"
    assert data["shots"][0]["id"] == "shot-01"
    assert set(REQUIRED_SECTIONS).issubset(set(data["shots"][0]["sections"].keys()))


def test_parse_shots_rejects_missing_section():
    obj = json.loads(GOOD_RESPONSE)
    del obj["shots"][0]["sections"]["POSITIVE_LOCKS"]
    try:
        parse_shots(json.dumps(obj))
        assert False, "should have raised"
    except ValueError as e:
        assert "POSITIVE_LOCKS" in str(e)


def test_parse_shots_rejects_unknown_asset_ref():
    obj = json.loads(GOOD_RESPONSE)
    obj["shots"][0]["asset_refs"] = ["does-not-exist"]
    try:
        parse_shots(json.dumps(obj))
        assert False, "should have raised"
    except ValueError as e:
        assert "does-not-exist" in str(e)


def test_parse_shots_strips_code_fences():
    fenced = "```json\n" + GOOD_RESPONSE + "\n```"
    data = parse_shots(fenced)
    assert data["shots"][0]["id"] == "shot-01"


def test_call_fcc_posts_and_returns_text():
    captured = {}

    class FakeResp:
        def __init__(self, body: bytes):
            self._body = body
        def read(self) -> bytes:
            return self._body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode())
        return FakeResp(json.dumps({"content": [{"type": "text", "text": "RESP"}]}).encode())

    with patch("architect.urlopen", fake_urlopen):
        out = call_fcc("hi", model="ollama/glm-5.2:cloud")
    assert out == "RESP"
    assert captured["url"] == "http://localhost:8082/v1/messages"
    assert captured["body"]["model"] == "ollama/glm-5.2:cloud"
    assert captured["body"]["messages"][0]["content"] == "hi"
    assert captured["headers"]["X-Api-key"] == "freecc"


def test_architect_writes_shots_json_and_seeds_progress(tmp_path: Path):
    proj = create_project("film", base=tmp_path)
    with patch("architect.call_fcc", return_value=GOOD_RESPONSE):
        out = architect("a cat on a couch", proj, model="m", recipe_path=None)
    assert out == proj.shots_json
    data = json.loads(proj.shots_json.read_text())
    assert data["shots"][0]["id"] == "shot-01"
    progress = json.loads(proj.progress_json.read_text())
    assert progress["assets"]["char-cat"]["status"] == "pending"
    assert progress["shots"]["shot-01"]["status"] == "pending"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest .claude/skills/hailuo-film/tests/test_architect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'architect'`

- [ ] **Step 3: Write minimal implementation**

Create `.claude/skills/hailuo-film/scripts/architect.py`:

```python
"""architect.py — concept -> shots.json via the FCC proxy.

Pure stdlib. Calls POST http://localhost:8082/v1/messages with x-api-key: freecc.
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from project import Project, save_progress

DEFAULT_RECIPE_PATH = Path(__file__).resolve().parent.parent / "reference" / "seedance-recipe.md"

REQUIRED_SECTIONS = (
    "SCENE_CONTEXT",
    "ACTIVE_REFERENCES",
    "LOCATION_MAP",
    "FIRST_FRAME_BLOCKING",
    "FORMAT_MODE",
    "OPTICS",
    "CAMERA",
    "ACTION",
    "PERFORMANCE",
    "PHYSICS",
    "LIGHTING",
    "COLOR_GRADE",
    "AUDIO",
    "STYLE",
    "OUTPUT_SETTINGS",
    "POSITIVE_LOCKS",
)


def build_architect_prompt(concept: str, recipe: str) -> str:
    return (
        "You are a film prompt architect. Use the recipe below to break the given concept "
        "into a shot list with a shared asset library and a 16-section prompt per shot.\n\n"
        "=== RECIPE ===\n" + recipe + "\n\n=== CONCEPT ===\n" + concept + "\n\n"
        "Emit ONLY the JSON object described by the recipe's schema. No prose, no code fences, "
        "no commentary. Every shot must contain all 16 section keys, each non-empty. "
        "asset_refs must only reference ids present in asset_manifest. duration_sec between 3 and 10."
    )


def call_fcc(
    prompt: str,
    model: str | None = None,
    base_url: str = "http://localhost:8082",
    auth_token: str = "freecc",
    timeout: float = 120.0,
) -> str:
    """POST to the FCC proxy /v1/messages and return the assistant text."""
    body: dict = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 8192}
    if model is not None:
        body["model"] = model
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": auth_token,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — localhost proxy
        raw = resp.read().decode()
    parsed = json.loads(raw)
    parts = parsed.get("content", [])
    texts = [b.get("text", "") for b in parts if b.get("type") == "text"]
    return "\n".join(t for t in texts if t)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def parse_shots(response_text: str) -> dict:
    """Validate the model's JSON response against the recipe schema."""
    text = _strip_fences(response_text)
    # Find the first { ... } JSON object if there's any surrounding noise.
    if not text.startswith("{"):
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in architect response")
        text = text[start:]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Architect response is not valid JSON: {e}") from e

    if "concept" not in data:
        raise ValueError("Architect response missing 'concept'")
    if "asset_manifest" not in data or not isinstance(data["asset_manifest"], list):
        raise ValueError("Architect response missing 'asset_manifest' list")
    if "shots" not in data or not isinstance(data["shots"], list):
        raise ValueError("Architect response missing 'shots' list")

    asset_ids = {a["id"] for a in data["asset_manifest"] if isinstance(a, dict) and "id" in a}
    for shot in data["shots"]:
        sid = shot.get("id", "<no-id>")
        sections = shot.get("sections", {})
        missing = [s for s in REQUIRED_SECTIONS if s not in sections or not str(sections[s]).strip()]
        if missing:
            raise ValueError(f"Shot {sid} missing/empty sections: {missing}")
        for ref in shot.get("asset_refs", []):
            if ref not in asset_ids:
                raise ValueError(f"Shot {sid} references unknown asset '{ref}'")
        if not (3 <= int(shot.get("duration_sec", 0)) <= 10):
            raise ValueError(f"Shot {sid} duration_sec must be 3..10, got {shot.get('duration_sec')}")
    return data


def architect(
    concept: str,
    project: Project,
    model: str | None = None,
    recipe_path: Path | None = None,
) -> Path:
    """Run the full architect step: call FCC, parse, write shots.json, seed progress.json."""
    recipe_file = recipe_path if recipe_path is not None else DEFAULT_RECIPE_PATH
    recipe = recipe_file.read_text()
    prompt = build_architect_prompt(concept, recipe)
    response = call_fcc(prompt, model=model)
    data = parse_shots(response)
    project.shots_json.write_text(json.dumps(data, indent=2))
    # Seed progress.json with every asset and shot pending.
    progress = {
        "assets": {a["id"]: {"status": "pending"} for a in data["asset_manifest"]},
        "shots": {s["id"]: {"status": "pending"} for s in data["shots"]},
    }
    save_progress(project, progress)
    return project.shots_json
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest .claude/skills/hailuo-film/tests/test_architect.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Lint + typecheck**

Run: `uv run ruff format .claude/skills/hailuo-film/scripts/architect.py .claude/skills/hailuo-film/tests/test_architect.py && uv run ruff check --fix .claude/skills/hailuo-film && uv run ty check .claude/skills/hailuo-film/scripts/architect.py`
Expected: clean. Re-run pytest if ruff changed anything. Note: `urllib.request.urlopen` may trip ruff S310 — the `# noqa: S310` comment handles it; if ruff still complains, run `uv run ruff check --fix` again and confirm the noqa is respected.

- [ ] **Step 6: Commit**

```bash
git add .claude/skills/hailuo-film/scripts/architect.py .claude/skills/hailuo-film/tests/test_architect.py
git commit -m "feat(hailuo-film): architect.py — concept -> shots.json via FCC proxy"
```

---

## Task 4: `selectors.json` — UI selector map skeleton

**Files:**
- Create: `.claude/skills/hailuo-film/scripts/selectors.json`

**Interfaces:**
- Produces: a JSON file the SKILL.md and `hailuo_driver.md` reference. Structure: `{ "image_gen": {...}, "video_gen": {...}, "common": {...} }`. Values are human-readable cues (button text, a11y role + name) the agent matches against the Chrome DevTools MCP a11y snapshot — not CSS selectors. Filled with TODO-marker strings the smoke test replaces.

- [ ] **Step 1: Write the selector skeleton**

Create `.claude/skills/hailuo-film/scripts/selectors.json`:

```json
{
  "_comment": "UI selector map for driving Hailuo via Chrome DevTools MCP. Values are a11y cues (role + name / button text), NOT CSS selectors. Finalize during the manual smoke test (SKILL.md Task 6). Update here when Hailuo changes the UI.",
  "common": {
    "signed_in_indicator": "text: 'Create' present in top nav",
    "home_url": "https://hailuoai.com/"
  },
  "image_gen": {
    "entry": "link/button named 'Image' or 'Image Generation'",
    "prompt_input": "textarea with placeholder about describing the image",
    "reference_upload": "button labeled 'Upload reference' or area accepting image drop",
    "generate_button": "button labeled 'Generate' or 'Create image'",
    "result_image": "img with alt containing the prompt, or a download button appearing after render",
    "download": "button labeled 'Download'"
  },
  "video_gen": {
    "entry": "link/button named 'Video' or 'Image to Video' or 'I2V'",
    "firstframe_upload": "button/area labeled 'Upload first frame' or 'Image to Video'",
    "reference_upload": "button labeled 'Upload reference' (optional character/location refs)",
    "prompt_input": "textarea for the shot prompt",
    "generate_button": "button labeled 'Generate' or 'Create video'",
    "result_video": "video element or download button appearing after render",
    "download": "button labeled 'Download'"
  }
}
```

- [ ] **Step 2: Verify it parses**

Run: `uv run python -c "import json; json.load(open('.claude/skills/hailuo-film/scripts/selectors.json')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/hailuo-film/scripts/selectors.json
git commit -m "feat(hailuo-film): selectors.json — a11y cue map skeleton for Hailuo UI"
```

---

## Task 5: `hailuo_driver.md` — browser playbook

**Files:**
- Create: `.claude/skills/hailuo-film/scripts/hailuo_driver.md`

**Interfaces:**
- Consumes: `selectors.json` (Task 4). The agent reads this playbook at runtime and calls Chrome DevTools MCP tools (`new_page`, `take_snapshot`, `click`, `fill`, `upload_file`, `wait_for`, `take_screenshot`).
- Produces: a step-by-step markdown doc covering sign-in gate, asset generation, first-frame generation, I2V animation, download, and failure handling.

- [ ] **Step 1: Write the playbook**

Create `.claude/skills/hailuo-film/scripts/hailuo_driver.md`:

````markdown
# Hailuo browser driver playbook

Read `selectors.json` first — it maps each on-page control to an a11y cue (role + name / button text). Match those cues against the Chrome DevTools MCP a11y snapshot (`take_snapshot`), then act by `uid`.

## Conventions

- Use Chrome DevTools MCP tools only: `new_page`, `take_snapshot`, `click` (by uid), `fill` (by uid), `upload_file` (by uid), `wait_for`, `take_screenshot`, `evaluate_script`.
- Never type or store credentials. The human signs in; you drive after.
- After every generate, `wait_for` a completion cue (the Download button appears) before downloading. Hailuo renders can take 30s–3min.
- Save downloads into the project folder using `evaluate_script` to fetch the blob URL and write via the script, or use `take_screenshot`/network capture as fallback. Prefer the explicit Download button.

## Phase 0 — Sign-in gate

1. `new_page` to `selectors.common.home_url`.
2. `take_snapshot`. If the snapshot shows the signed-in indicator (`selectors.common.signed_in_indicator`), skip to Phase 1.
3. Otherwise: tell the user in chat: "Hailuo is open in Chrome — please sign in, then reply 'done'." Wait for the user's reply. Do not click any sign-in button yourself.
4. Once the user confirms, `take_snapshot` again to confirm the indicator is present. If not, ask them to confirm again.

## Phase 1 — Asset generation

For each asset id in `next_pending(progress, "asset")`:

1. Read the asset's prompt from `shots.json` `asset_manifest`.
2. Navigate to image-gen (`selectors.image_gen.entry`).
3. `take_snapshot`; find the prompt textarea uid (`selectors.image_gen.prompt_input`).
4. `fill` the asset prompt.
5. If the asset is a character and a reference image exists on disk, `upload_file` to `selectors.image_gen.reference_upload`. (First assets have no reference; skip.)
6. `click` the generate button (`selectors.image_gen.generate_button`).
7. `wait_for` the Download button (`selectors.image_gen.download`) or a completion cue.
8. Download the image to `assets/<asset_id>.png`.
9. `mark_asset(progress, asset_id, "done")`; `save_progress`. On any failure, `mark_asset(..., "failed", error=str(e))`, log, continue.

When `next_pending(progress, "asset")` returns None, move to Phase 2.

## Phase 2 — Shot first-frames

For each shot id in `next_pending(progress, "shot")`:

1. Read the shot's `FIRST_FRAME_BLOCKING` section + `asset_refs` from `shots.json`.
2. Navigate to image-gen.
3. `fill` the `FIRST_FRAME_BLOCKING` text into the prompt textarea.
4. For each `asset_ref`, `upload_file` the corresponding `assets/<ref>.png` to the reference upload.
5. Generate, wait, download to `assets/<shot_id>-firstframe.png`.

## Phase 3 — Animate (I2V)

Still inside the same shot iteration:

1. Navigate to video-gen / image-to-video (`selectors.video_gen.entry`).
2. `upload_file` `assets/<shot_id>-firstframe.png` to `selectors.video_gen.firstframe_upload`.
3. `fill` the full 16-section prompt (serialize the shot's `sections` as `KEY: value` lines) into `selectors.video_gen.prompt_input`.
4. Optionally `upload_file` asset reference images to `selectors.video_gen.reference_upload`.
5. `click` `selectors.video_gen.generate_button`.
6. `wait_for` `selectors.video_gen.download` (renders can take minutes; poll with `take_snapshot`).
7. Download the clip to `clips/<shot_id>.mp4`.
8. `mark_shot(progress, shot_id, "done")`; `save_progress`. On failure, `mark_shot(..., "failed", error=...)`, continue.

When `next_pending(progress, "shot")` returns None, run `write_manifest(project, shots_data, progress)` and tell the user the project is ready for assembly with `reel`/`hyperframes`.

## Failure handling

- **Element not found in snapshot:** halt the current item, mark it `failed` with `"error": "selector not found: <name>; update selectors.json"`, continue to next. Report all failures at the end. Do not guess uids.
- **Generation error (NSFW / rate limit):** capture the on-page error text via `take_snapshot`, mark `failed` with that text, continue.
- **Page closed mid-run:** `new_page` again, re-run the sign-in gate (the user may need to re-sign-in), then resume from `progress.json` — pending items only.
````

- [ ] **Step 2: Verify the file exists and references selectors.json**

Run: `uv run python -c "from pathlib import Path; t=Path('.claude/skills/hailuo-film/scripts/hailuo_driver.md').read_text(); assert 'selectors.json' in t and 'Phase 3' in t and 'sign-in' in t; print('ok', len(t))"`
Expected: `ok <number>`

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/hailuo-film/scripts/hailuo_driver.md
git commit -m "docs(hailuo-film): hailuo_driver.md — browser playbook for asset + clip generation"
```

---

## Task 6: `SKILL.md` — skill entry + smoke test

**Files:**
- Create: `.claude/skills/hailuo-film/SKILL.md`

**Interfaces:**
- Consumes: `project.py`, `architect.py`, `hailuo_driver.md`, `selectors.json`, `reference/seedance-recipe.md` (all earlier tasks).
- Produces: the skill definition with frontmatter, when-to-invoke, the 3-phase workflow, sign-in gate, and a documented manual smoke test.

- [ ] **Step 1: Write SKILL.md**

Create `.claude/skills/hailuo-film/SKILL.md`:

````markdown
---
name: hailuo-film
description: >
  Use when the user wants to generate a high-quality, cross-shot-consistent AI short film from a
  concept using Hailuo (MiniMax) — the Seedance 4K recipe adapted to this stack. The skill uses the
  FCC-proxied text model as a prompt architect (16-section per-shot prompts + shared asset library),
  then drives Hailuo's web UI via Chrome DevTools MCP to generate asset images and video clips into
  a resumable project folder. The user signs in to Hailuo once in a real Chrome; the skill never
  touches credentials. Output is a project folder + manifest.md for assembly with the reel or
  hyperframes skills (out of scope here). Do not use for single-image generation or for non-Hailuo
  generators.
metadata: { "tags": "video, ai-generation, hailuo, seedance, film, browser-automation, fcc-proxy" }
---

# `/hailuo-film` — Seedance-recipe film generation via Hailuo

Adapts the Higgsfield Seedance 4K breakdown to this stack: the FCC-proxied text model architects
prompts, Hailuo's web UI renders assets + clips, and a project folder holds resumable state.

## When to invoke

- User asks to make/generate an AI short film, AI video sequence, or a set of consistent AI clips
  from a concept, and names Hailuo (or accepts Hailuo when proposed).
- User references "the Seedance recipe" or "the 4K breakdown" and wants it run here.

Do **not** use for: a single image, a single clip with no cross-shot consistency need, or a non-Hailuo
generator. Use `hyperframes` for HTML-rendered video, `reel` for NLE assembly of existing clips.

## Inputs

- A **concept**: a short paragraph describing the film (one character, one location, a beat). Pass
  it as the argument, or point at a file path.

## The workflow

### Phase 1 — Architect (text model via FCC proxy)

1. Confirm the proxy is up: `curl -sf http://localhost:8082/health || echo "start it: fcc"`. If down, tell the user to run `fcc` and stop.
2. Pick a project folder: `uv run python .claude/skills/hailuo-film/scripts/architect.py "<concept>"` (the script creates the project under `~/hailuo-projects/`, writes `shots.json`, seeds `progress.json`, prints the project path).
   - Override the brain model with `MODEL=<fcc-model-ref>` if the user wants a specific one; default is the proxy's current `MODEL`.
3. Read `shots.json` and show the user the shot list + asset manifest. Confirm before generating.

The 16-section recipe the architect uses lives in `reference/seedance-recipe.md` — read it if you need to explain any section.

### Phase 2 — Sign-in gate + asset generation (Hailuo web UI)

1. Read `scripts/hailuo_driver.md` and `scripts/selectors.json`.
2. Open Hailuo via Chrome DevTools MCP `new_page` to the home URL.
3. Run the sign-in gate: ask the user in chat to sign in and reply "done". **Never click sign-in buttons yourself; never read or store credentials.**
4. Drive asset generation per `hailuo_driver.md` Phase 1, marking `progress.json` after each asset. Resume on re-run skips `done` assets.

### Phase 3 — Shot first-frames + animate (I2V)

1. Per `hailuo_driver.md` Phases 2 & 3: for each pending shot, generate its first-frame image (using the library assets as references), then drive image-to-video on that frame with the full 16-section prompt.
2. Download each clip to `clips/<shot_id>.mp4`, mark `progress.json`.

### Handoff

When all shots are `done`, write `manifest.md`:

```bash
uv run python .claude/skills/hailuo-film/scripts/finish.py <project_path>
```

Tell the user the project folder is ready and suggest assembling with `reel` (FFmpeg NLE) or `hyperframes`.

## Resumability

Every run reads `progress.json` first. `done` items are skipped, `failed` items are retried (or skipped if `--skip-failed` is passed — to be added if requested), `pending` items are processed in order. A crashed browser or interrupted run loses no completed work.

## Smoke test (manual, run once after first install and whenever Hailuo's UI changes)

1. Start the proxy: `fcc` (in another shell).
2. Run the architect on a tiny concept:
   `uv run python .claude/skills/hailuo-film/scripts/architect.py "a single 5-second shot of an orange cat sitting on a couch, neutral lighting"`
3. Confirm `shots.json` has 1 shot and 1 asset.
4. Open Hailuo via Chrome DevTools MCP, complete the sign-in gate, and drive one asset + one shot end-to-end per `hailuo_driver.md`.
5. If any selector in `selectors.json` does not match the live UI, update `selectors.json` with the correct a11y cue and re-run.

## Security

- The skill never enters, reads back, or stores any credential. Sign-in is human-only.
- All generated files are local to the project folder. The skill uploads nothing except the asset/first-frame images the user generated, and only to Hailuo's own UI.
````

- [ ] **Step 2: Verify frontmatter parses**

Run: `uv run python -c "from pathlib import Path; t=Path('.claude/skills/hailuo-film/SKILL.md').read_text(); assert t.startswith('---') and 'name: hailuo-film' in t and 'Phase 3' in t; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Run the full skill test suite**

Run: `uv run pytest .claude/skills/hailuo-film/tests/ -v`
Expected: all tests (project + architect) PASS.

- [ ] **Step 4: Lint the whole skill**

Run: `uv run ruff format .claude/skills/hailuo-film && uv run ruff check --fix .claude/skills/hailuo-film && uv run ty check .claude/skills/hailuo-film/scripts`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add .claude/skills/hailuo-film/SKILL.md
git commit -m "feat(hailuo-film): SKILL.md — 3-phase workflow + sign-in gate + smoke test"
```

- [ ] **Step 6: Final verification — full CI on the skill subtree**

Run: `uv run pytest .claude/skills/hailuo-film/tests/ -v && uv run ruff check .claude/skills/hailuo-film && uv run ty check .claude/skills/hailuo-film/scripts`
Expected: all green. If the repo-wide `./scripts/ci.sh` is convenient, run it too — but the skill is isolated from the fcc proxy package, so it should not affect package checks.

- [ ] **Step 7: Merge to main**

```bash
git checkout main && git merge --no-ff hailuo-film-design -m "Merge hailuo-film skill"
```
(Only after the user confirms the smoke test passes against the real Hailuo UI. Do not merge before the manual smoke test in Task 6 has been run at least once.)