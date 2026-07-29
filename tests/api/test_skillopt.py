from __future__ import annotations

import json

from api import skillopt


def _write_policy(tmp_path, payload) -> None:
    (tmp_path / "skillopt_policy.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_disabled_by_default_returns_none(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SKILLOPT_ENABLED", raising=False)
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    _write_policy(
        tmp_path,
        {
            "version": 1,
            "policies": {"edit": {"primary": "openai/gpt-4o", "fallbacks": []}},
        },
    )
    skillopt.invalidate_cache()

    assert skillopt.lookup("edit") is None
    assert skillopt.is_enabled() is False


def test_enabled_lookup_returns_policy(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SKILLOPT_ENABLED", "1")
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    _write_policy(
        tmp_path,
        {
            "version": 1,
            "policies": {
                "edit": {
                    "primary": "openai/gpt-4o",
                    "fallbacks": ["deepseek/deepseek-chat"],
                }
            },
        },
    )
    skillopt.invalidate_cache()

    policy = skillopt.lookup("edit")
    assert policy is not None
    assert policy.primary == "openai/gpt-4o"
    assert policy.fallbacks == ("deepseek/deepseek-chat",)


def test_missing_skill_returns_none(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SKILLOPT_ENABLED", "1")
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    _write_policy(tmp_path, {"version": 1, "policies": {}})
    skillopt.invalidate_cache()

    assert skillopt.lookup("chat") is None


def test_missing_policy_file_returns_none(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SKILLOPT_ENABLED", "1")
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    skillopt.invalidate_cache()

    assert skillopt.lookup("edit") is None


def test_malformed_policy_returns_none(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SKILLOPT_ENABLED", "1")
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    (tmp_path / "skillopt_policy.json").write_text("{not json", encoding="utf-8")
    skillopt.invalidate_cache()

    assert skillopt.lookup("edit") is None


def test_cache_refreshes_on_mtime_change(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SKILLOPT_ENABLED", "1")
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    _write_policy(
        tmp_path,
        {"version": 1, "policies": {"edit": {"primary": "openai/gpt-4o"}}},
    )
    skillopt.invalidate_cache()
    first = skillopt.lookup("edit")
    assert first is not None and first.primary == "openai/gpt-4o"

    # Overwrite the policy file with a fresh mtime.
    _write_policy(
        tmp_path,
        {
            "version": 2,
            "policies": {
                "edit": {"primary": "deepseek/deepseek-chat", "fallbacks": []}
            },
        },
    )
    import os

    now = skillopt.policy_path().stat().st_mtime_ns
    os.utime(skillopt.policy_path(), ns=(now + 1_000_000_000, now + 1_000_000_000))

    second = skillopt.lookup("edit")
    assert second is not None and second.primary == "deepseek/deepseek-chat"


def test_snapshot_reports_state(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SKILLOPT_ENABLED", "1")
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    _write_policy(
        tmp_path,
        {
            "version": 1,
            "policies": {
                "edit": {
                    "primary": "openai/gpt-4o",
                    "fallbacks": ["deepseek/deepseek-chat"],
                }
            },
        },
    )
    skillopt.invalidate_cache()

    snap = skillopt.snapshot()
    assert snap["enabled"] is True
    assert snap["loaded"] is True
    assert snap["version"] == 1
    assert snap["policies"]["edit"]["primary"] == "openai/gpt-4o"


def test_snapshot_when_disabled(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SKILLOPT_ENABLED", raising=False)
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    skillopt.invalidate_cache()

    snap = skillopt.snapshot()
    assert snap["enabled"] is False
    assert snap["loaded"] is False
