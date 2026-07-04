"""Tests for scripts/project.py — resume logic and scaffolding."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

import project


def test_slugify():
    assert project.slugify("A B C!!!") == "a-b-c"
    assert project.slugify("  spaced--text  ") == "spaced-text"


def test_project_dir_creates_structure(tmp_path):
    proj = project.project_dir("Test Concept", root=tmp_path, timestamp=False)
    assert proj.exists()
    assert (proj / "assets").exists()
    assert (proj / "clips").exists()
    assert (proj / "reference").exists()
    assert proj.name == "test-concept"


def test_progress_skip_done(tmp_path):
    proj = project.project_dir("x", root=tmp_path, timestamp=False)
    p = project.Progress(proj)
    p.set_status("assets", "a1", "done", path="assets/a1.png")
    p.set_status("assets", "a2", "pending")

    assets = [{"id": "a1"}, {"id": "a2"}]
    pending = p.pending_assets(assets)
    assert len(pending) == 1
    assert pending[0]["id"] == "a2"


def test_progress_retry_failed(tmp_path):
    proj = project.project_dir("x", root=tmp_path, timestamp=False)
    p = project.Progress(proj)
    p.set_status("shots", "s1", "failed", error="nsfw")
    p.set_status("shots", "s2", "done")

    shots = [{"id": "s1"}, {"id": "s2"}]
    assert len(p.pending_shots(shots)) == 0
    assert len(p.pending_shots(shots, retry_failed=True)) == 1


def test_progress_summary(tmp_path):
    proj = project.project_dir("x", root=tmp_path, timestamp=False)
    p = project.Progress(proj)
    p.set_status("assets", "a1", "done")
    p.set_status("assets", "a2", "failed")
    p.set_status("assets", "a3", "pending")
    summary = p.summary()
    assert summary["assets"] == {"total": 3, "done": 1, "failed": 1, "pending": 1}


def test_write_manifest(tmp_path):
    proj = project.project_dir("x", root=tmp_path, timestamp=False)
    path = project.write_manifest(
        proj,
        "Test Film",
        shots=[
            {
                "id": "shot-01",
                "title": "Open",
                "duration_seconds": 6,
                "first_frame_asset_id": "loc-01",
                "prompt": "Wide establishing shot...",
            }
        ],
        assets=[{"id": "loc-01", "type": "location", "description": "City rooftop"}],
    )
    text = path.read_text(encoding="utf-8")
    assert "Test Film" in text
    assert "shot-01" in text
    assert "loc-01" in text
