"""Project scaffolding and progress.json resume helpers for hailuo-film."""

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path.home() / "hailuo-projects"


def slugify(text: str) -> str:
    """Create a filesystem-safe slug from a concept string."""
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    s = re.sub(r"[-\s]+", "-", s)
    return s[:60]


def project_dir(concept: str, root: Path | str | None = None, timestamp: bool = True) -> Path:
    """Return (and create) a project directory for a concept."""
    root_path = Path(root or DEFAULT_ROOT)
    slug = slugify(concept)
    if timestamp:
        slug = f"{slug}-{int(time.time())}"
    proj = root_path / slug
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "assets").mkdir(exist_ok=True)
    (proj / "clips").mkdir(exist_ok=True)
    (proj / "reference").mkdir(exist_ok=True)
    return proj


def copy_skill_files(proj: Path, skill_root: Path | None = None) -> None:
    """Copy SKILL.md and reference docs into the project folder for offline readability."""
    if skill_root is None:
        skill_root = Path(__file__).resolve().parent.parent
    src_skill = skill_root / "SKILL.md"
    src_ref = skill_root / "reference" / "seedance-recipe.md"
    if src_skill.exists():
        shutil.copy2(src_skill, proj / "SKILL.md")
    if src_ref.exists():
        shutil.copy2(src_ref, proj / "reference" / "seedance-recipe.md")


class Progress:
    """Tiny resume-state manager backed by progress.json."""

    def __init__(self, proj: Path, reset: bool = False):
        self.path = proj / "progress.json"
        if reset or not self.path.exists():
            self.data: dict[str, Any] = {
                "assets": {},  # asset_id -> {"status": pending|done|failed, "error": str, "path": str}
                "shots": {},   # shot_id -> {"status": pending|done|failed, "error": str, "path": str}
                "started_at": time.time(),
                "updated_at": time.time(),
            }
            self.save()
        else:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.data["updated_at"] = time.time()
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def is_done(self, kind: str, item_id: str) -> bool:
        return self.data.get(kind, {}).get(item_id, {}).get("status") == "done"

    def is_failed(self, kind: str, item_id: str) -> bool:
        return self.data.get(kind, {}).get(item_id, {}).get("status") == "failed"

    def status(self, kind: str, item_id: str) -> str:
        return self.data.get(kind, {}).get(item_id, {}).get("status", "pending")

    def set_status(self, kind: str, item_id: str, status: str, path: str | None = None, error: str | None = None) -> None:
        self.data.setdefault(kind, {})[item_id] = {
            "status": status,
            "path": path,
            "error": error,
            "updated_at": time.time(),
        }
        self.save()

    def pending_assets(self, assets: list[dict]) -> list[dict]:
        """Return assets that are not done. If retry_failed is desired, callers can also include failed."""
        return [a for a in assets if not self.is_done("assets", a["id"])]

    def pending_shots(self, shots: list[dict], retry_failed: bool = False) -> list[dict]:
        """Return shots that are not done. Optionally include failed ones for retry."""
        out = []
        for s in shots:
            st = self.status("shots", s["id"])
            if st == "done":
                continue
            if st == "failed" and not retry_failed:
                continue
            out.append(s)
        return out

    def summary(self) -> dict[str, Any]:
        """Return counts of done/failed/pending items."""
        def counts(d: dict) -> dict[str, int]:
            total = len(d)
            done = sum(1 for v in d.values() if v.get("status") == "done")
            failed = sum(1 for v in d.values() if v.get("status") == "failed")
            return {"total": total, "done": done, "failed": failed, "pending": total - done - failed}
        return {"assets": counts(self.data.get("assets", {})), "shots": counts(self.data.get("shots", {}))}


def write_shots_json(proj: Path, payload: dict) -> Path:
    """Write the architect output to shots.json."""
    path = proj / "shots.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_shots_json(proj: Path) -> dict:
    """Read shots.json."""
    return json.loads((proj / "shots.json").read_text(encoding="utf-8"))


def write_manifest(proj: Path, concept: str, shots: list[dict], assets: list[dict]) -> Path:
    """Write manifest.md handoff doc for reel/hyperframes."""
    lines = [
        f"# Hailuo Film Manifest: {concept}",
        "",
        "## Assets",
        "",
    ]
    for a in assets:
        path = a.get("path") or f"assets/{a['id']}.png"
        lines.append(f"- **{a['id']}** ({a.get('type', 'image')}): {a.get('description', '')}")
        lines.append(f"  - File: `{path}`")
        lines.append("")
    lines.append("## Shots")
    lines.append("")
    for s in shots:
        clip_path = s.get("clip_path") or f"clips/{s['id']}.mp4"
        first_frame = s.get("first_frame_path") or f"assets/{s.get('first_frame_asset_id', '')}.png"
        lines.append(f"### {s['id']}: {s.get('title', 'Untitled')}")
        lines.append(f"- Duration: {s.get('duration_seconds', '?')}s")
        lines.append(f"- First frame asset: `{first_frame}`")
        lines.append(f"- Clip: `{clip_path}`")
        lines.append(f"- Prompt excerpt: {s.get('prompt', '')[:200].strip()}...")
        lines.append("")
    path = proj / "manifest.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
