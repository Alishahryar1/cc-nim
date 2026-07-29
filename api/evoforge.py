"""EvoForge — offline processor that converts a trajectory JSONL corpus
into a versioned SkillOpt policy file.

Design contract:

    input  : ``${FCC_CACHE_DIR}/trajectories.jsonl`` (+ optional ``.1``)
    output : ``${FCC_CACHE_DIR}/skillopt_policy.json``

The pipeline is deterministic and re-runnable. It never touches live
traffic; the gateway only reads the output file (see ``api/skillopt.py``).
Kept as a plain module so both the CLI shim (``scripts/evoforge.py``) and
unit tests can drive it directly.

Scoring: for each ``(skill, provider/model)`` candidate observed
``min_samples`` or more times, compute a utility score

    score = -λ_cost · avg_cost - λ_latency · p95_latency + λ_success · ok_rate

The primary is the argmax; ``top_k - 1`` runners-up become ``fallbacks``.
Ordering ties break by provider id (stable output across runs).
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import quantiles
from typing import Any

POLICY_SCHEMA_VERSION = 1
DEFAULT_MIN_SAMPLES = 5
DEFAULT_TOP_K = 3
# Per-unit weights chosen so a request that costs $0.001 and runs 500 ms
# with 95 % ok rate scores near zero — tune when the corpus grows.
DEFAULT_LAMBDA_COST = 1000.0  # score punishes $/req heavily
DEFAULT_LAMBDA_LATENCY_MS = 0.001  # 500 ms costs ~0.5 score points
DEFAULT_LAMBDA_SUCCESS = 1.0  # ok_rate 1.0 adds a full point


@dataclass(frozen=True, slots=True)
class ForgeParams:
    """All knobs for a single EvoForge run."""

    min_samples: int = DEFAULT_MIN_SAMPLES
    top_k: int = DEFAULT_TOP_K
    lambda_cost: float = DEFAULT_LAMBDA_COST
    lambda_latency_ms: float = DEFAULT_LAMBDA_LATENCY_MS
    lambda_success: float = DEFAULT_LAMBDA_SUCCESS


@dataclass(frozen=True, slots=True)
class ForgeStats:
    """Aggregated per-candidate statistics."""

    skill: str
    provider_id: str
    model: str
    samples: int
    ok_rate: float
    avg_cost_usd: float
    p95_latency_ms: float
    score: float


def default_input_paths() -> tuple[Path, ...]:
    base = os.environ.get("FCC_CACHE_DIR")
    root = Path(base) if base else Path.home() / ".fcc-cache"
    return (root / "trajectories.jsonl", root / "trajectories.jsonl.1")


def default_output_path() -> Path:
    base = os.environ.get("FCC_CACHE_DIR")
    root = Path(base) if base else Path.home() / ".fcc-cache"
    return root / "skillopt_policy.json"


def load_rows(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    """Read every JSONL row across the given files. Silently skips bad lines."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and _looks_valid(obj):
                rows.append(obj)
    return rows


def _looks_valid(row: dict[str, Any]) -> bool:
    required = ("skill", "provider_id", "model", "latency_ms", "status")
    return all(k in row for k in required)


def aggregate(rows: list[dict[str, Any]], params: ForgeParams) -> list[ForgeStats]:
    """Group rows by (skill, provider, model); drop under-sampled candidates."""
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["skill"], row["provider_id"], row["model"])
        buckets[key].append(row)

    stats: list[ForgeStats] = []
    for (skill, provider, model), entries in buckets.items():
        if len(entries) < params.min_samples:
            continue
        latencies = [float(e.get("latency_ms") or 0.0) for e in entries]
        costs = [float(e.get("cost_usd") or 0.0) for e in entries]
        oks = sum(1 for e in entries if e.get("status") == "ok")
        ok_rate = oks / len(entries)
        avg_cost = sum(costs) / len(costs)
        p95 = _p95(latencies)
        score = (
            -params.lambda_cost * avg_cost
            - params.lambda_latency_ms * p95
            + params.lambda_success * ok_rate
        )
        stats.append(
            ForgeStats(
                skill=skill,
                provider_id=provider,
                model=model,
                samples=len(entries),
                ok_rate=round(ok_rate, 4),
                avg_cost_usd=round(avg_cost, 6),
                p95_latency_ms=round(p95, 1),
                score=round(score, 6),
            )
        )
    return stats


def _p95(latencies: list[float]) -> float:
    if not latencies:
        return 0.0
    if len(latencies) < 2:
        return latencies[0]
    # ``quantiles(n=20)`` returns 19 cut points; index 18 is the 95th percentile.
    cuts = quantiles(sorted(latencies), n=20, method="inclusive")
    return cuts[-1]


def build_policy(
    stats: list[ForgeStats],
    params: ForgeParams,
) -> dict[str, Any]:
    """Return the JSON-serialisable policy — sorted, deterministic."""
    per_skill: dict[str, list[ForgeStats]] = defaultdict(list)
    for s in stats:
        per_skill[s.skill].append(s)

    policies: dict[str, dict[str, Any]] = {}
    for skill in sorted(per_skill):
        ranked = sorted(
            per_skill[skill],
            key=lambda s: (-s.score, s.provider_id, s.model),
        )
        winners = ranked[: params.top_k]
        if not winners:
            continue
        primary = f"{winners[0].provider_id}/{winners[0].model}"
        fallbacks = [f"{w.provider_id}/{w.model}" for w in winners[1:]]
        policies[skill] = {"primary": primary, "fallbacks": fallbacks}

    return {
        "version": POLICY_SCHEMA_VERSION,
        "generated_ts": int(time.time()),
        "params": {
            "min_samples": params.min_samples,
            "top_k": params.top_k,
            "lambda_cost": params.lambda_cost,
            "lambda_latency_ms": params.lambda_latency_ms,
            "lambda_success": params.lambda_success,
        },
        "policies": policies,
    }


def write_policy(policy: dict[str, Any], out: Path) -> None:
    """Write the policy JSON atomically (write to .tmp, then rename)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(
        json.dumps(policy, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    tmp.replace(out)


def run(
    *,
    inputs: tuple[Path, ...] | None = None,
    output: Path | None = None,
    params: ForgeParams | None = None,
) -> dict[str, Any]:
    """End-to-end EvoForge run. Returns the policy for logging/inspection."""
    real_inputs = inputs or default_input_paths()
    real_output = output or default_output_path()
    real_params = params or ForgeParams()
    rows = load_rows(real_inputs)
    stats = aggregate(rows, real_params)
    policy = build_policy(stats, real_params)
    write_policy(policy, real_output)
    return policy
