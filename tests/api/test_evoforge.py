from __future__ import annotations

import json

from api import evoforge


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _row(provider, model, skill, latency, cost, status="ok"):
    return {
        "provider_id": provider,
        "model": model,
        "skill": skill,
        "latency_ms": latency,
        "cost_usd": cost,
        "status": status,
        "input_tokens": 100,
        "output_tokens": 100,
    }


def test_load_rows_reads_valid_jsonl(tmp_path) -> None:
    p = tmp_path / "trajectories.jsonl"
    _write_jsonl(
        p,
        [
            _row("openai", "gpt-4o", "edit", 500, 0.001),
            _row("deepseek", "deepseek-chat", "edit", 700, 0.0001),
        ],
    )
    rows = evoforge.load_rows((p,))
    assert len(rows) == 2


def test_load_rows_silently_skips_bad_lines(tmp_path) -> None:
    p = tmp_path / "trajectories.jsonl"
    p.write_text(
        "not-json\n"
        + json.dumps(_row("openai", "gpt-4o", "chat", 100, 0.0))
        + "\n"
        + json.dumps({"partial": True})  # missing required keys
        + "\n",
        encoding="utf-8",
    )
    rows = evoforge.load_rows((p,))
    assert len(rows) == 1


def test_under_sampled_candidates_are_dropped(tmp_path) -> None:
    p = tmp_path / "trajectories.jsonl"
    _write_jsonl(
        p,
        [_row("openai", "gpt-4o", "edit", 500, 0.001)] * 2
        + [_row("deepseek", "deepseek-chat", "edit", 700, 0.0001)] * 6,
    )
    rows = evoforge.load_rows((p,))
    stats = evoforge.aggregate(rows, evoforge.ForgeParams(min_samples=5))
    # only deepseek survives (2 openai rows < min_samples=5)
    assert {(s.provider_id, s.model) for s in stats} == {
        ("deepseek", "deepseek-chat")
    }


def test_cheaper_and_faster_wins(tmp_path) -> None:
    # openai/gpt-4o : $0.001, 500ms, ok=1.0 (expensive)
    # deepseek/deepseek-chat: $0.0001, 300ms, ok=1.0 (cheap+fast)
    rows = (
        [_row("openai", "gpt-4o", "edit", 500, 0.001)] * 10
        + [_row("deepseek", "deepseek-chat", "edit", 300, 0.0001)] * 10
    )
    stats = evoforge.aggregate(rows, evoforge.ForgeParams(min_samples=5))
    policy = evoforge.build_policy(stats, evoforge.ForgeParams())

    assert policy["policies"]["edit"]["primary"] == "deepseek/deepseek-chat"
    assert "openai/gpt-4o" in policy["policies"]["edit"]["fallbacks"]


def test_policy_shape_versioned_and_deterministic(tmp_path) -> None:
    rows = [_row("openai", "gpt-4o", "chat", 400, 0.0005)] * 8
    policy1 = evoforge.build_policy(
        evoforge.aggregate(rows, evoforge.ForgeParams(min_samples=5)),
        evoforge.ForgeParams(),
    )
    policy2 = evoforge.build_policy(
        evoforge.aggregate(rows, evoforge.ForgeParams(min_samples=5)),
        evoforge.ForgeParams(),
    )
    # Same input → same policies (generated_ts differs; check shape).
    assert policy1["version"] == evoforge.POLICY_SCHEMA_VERSION
    assert policy1["policies"] == policy2["policies"]
    assert set(policy1["policies"]["chat"]) == {"primary", "fallbacks"}


def test_write_policy_is_atomic(tmp_path) -> None:
    out = tmp_path / "skillopt_policy.json"
    policy = {
        "version": 1,
        "generated_ts": 123,
        "params": {},
        "policies": {"edit": {"primary": "openai/gpt-4o", "fallbacks": []}},
    }
    evoforge.write_policy(policy, out)
    assert out.is_file()
    # No leftover .tmp
    assert not (tmp_path / "skillopt_policy.json.tmp").exists()
    loaded = json.loads(out.read_text())
    assert loaded["policies"]["edit"]["primary"] == "openai/gpt-4o"


def test_run_end_to_end(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "trajectories.jsonl"
    output_path = tmp_path / "skillopt_policy.json"
    _write_jsonl(input_path, [_row("openai", "gpt-4o", "chat", 200, 0.0002)] * 6)

    policy = evoforge.run(inputs=(input_path,), output=output_path)

    assert output_path.is_file()
    assert policy["policies"]["chat"]["primary"] == "openai/gpt-4o"


def test_status_error_lowers_ok_rate(tmp_path) -> None:
    rows = [_row("openai", "gpt-4o", "edit", 500, 0.001)] * 8 + [
        _row("openai", "gpt-4o", "edit", 500, 0.001, status="error")
    ] * 2
    stats = evoforge.aggregate(rows, evoforge.ForgeParams(min_samples=5))
    entry = next(s for s in stats if s.provider_id == "openai")
    assert entry.ok_rate == 0.8
