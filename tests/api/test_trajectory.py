from __future__ import annotations

import json

from api import trajectory


def _enable(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TRAJECTORY_LOG_ENABLED", "1")
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    trajectory.reload_config()
    trajectory.clear()


def test_disabled_by_default_is_a_noop(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TRAJECTORY_LOG_ENABLED", raising=False)
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    trajectory.reload_config()
    trajectory.clear()

    trajectory.record(
        request_id="r1",
        provider_id="openai",
        model="gpt-4o",
        skill="chat",
        thinking_enabled=False,
        input_tokens=10,
        output_tokens=5,
        latency_ms=42.0,
        cost_usd=0.001,
        status="ok",
    )

    assert trajectory.summary()["total"] == 0
    assert not (tmp_path / "trajectories.jsonl").exists()


def test_record_persists_and_summary_rolls_up(monkeypatch, tmp_path) -> None:
    _enable(monkeypatch, tmp_path)

    trajectory.record(
        request_id="r1",
        provider_id="openai",
        model="gpt-4o",
        skill="edit",
        thinking_enabled=True,
        input_tokens=100,
        output_tokens=50,
        latency_ms=125.4,
        cost_usd=0.0002,
        status="ok",
    )
    trajectory.record(
        request_id="r2",
        provider_id="openai",
        model="gpt-4o-mini",
        skill="chat",
        thinking_enabled=False,
        input_tokens=20,
        output_tokens=10,
        latency_ms=90.0,
        cost_usd=0.00001,
        status="ok",
    )

    summary = trajectory.summary()
    assert summary["enabled"] is True
    assert summary["total"] == 2
    assert summary["per_skill"] == {"edit": 1, "chat": 1}
    assert summary["per_provider"] == {"openai": 2}
    assert summary["tokens_in"] == 120
    assert summary["tokens_out"] == 60

    log = tmp_path / "trajectories.jsonl"
    assert log.exists()
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    assert rows[0]["skill"] == "edit"
    assert rows[1]["model"] == "gpt-4o-mini"


def test_rotation_kicks_in_at_byte_cap(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TRAJECTORY_LOG_ENABLED", "1")
    monkeypatch.setenv("FCC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRAJECTORY_LOG_MAX_BYTES", "300")
    trajectory.reload_config()
    trajectory.clear()

    for i in range(20):
        trajectory.record(
            request_id=f"r{i}",
            provider_id="openai",
            model="gpt-4o",
            skill="chat",
            thinking_enabled=False,
            input_tokens=1,
            output_tokens=1,
            latency_ms=1.0,
            cost_usd=0.0,
            status="ok",
        )

    log = tmp_path / "trajectories.jsonl"
    rotated = tmp_path / "trajectories.jsonl.1"
    assert log.exists()
    assert rotated.exists()


def test_infer_skill_probe_when_tiny_and_no_tools() -> None:
    assert trajectory.infer_skill([], None, 10) == "probe"


def test_infer_skill_edit_from_tool_names() -> None:
    tools = [{"name": "Edit"}, {"name": "TodoWrite"}]
    assert trajectory.infer_skill([{"content": "x"}], tools, 5000) == "edit"


def test_infer_skill_question_from_leading_word() -> None:
    messages = [{"role": "user", "content": "what does this function do"}]
    assert trajectory.infer_skill(messages, None, 5000) == "question"


def test_infer_skill_plan_from_keyword() -> None:
    messages = [{"role": "user", "content": "plan the migration in three phases"}]
    assert trajectory.infer_skill(messages, None, 5000) == "plan"
