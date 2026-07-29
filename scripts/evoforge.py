#!/usr/bin/env python
"""EvoForge CLI — turn trajectories into a SkillOpt policy.

Usage::

    uv run python scripts/evoforge.py            # defaults from FCC_CACHE_DIR
    uv run python scripts/evoforge.py --dry-run  # print policy, don't write
    uv run python scripts/evoforge.py --input trajectories.jsonl --output policy.json

Zero effect on the running gateway — only writes the policy file that
``api/skillopt.py`` picks up on the next request.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from api import evoforge


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="EvoForge — offline SkillOpt policy builder"
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help="Trajectory JSONL file (repeatable)",
    )
    parser.add_argument(
        "--output", type=Path, help="Where to write the SkillOpt policy JSON"
    )
    parser.add_argument(
        "--min-samples", type=int, default=evoforge.DEFAULT_MIN_SAMPLES
    )
    parser.add_argument("--top-k", type=int, default=evoforge.DEFAULT_TOP_K)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print policy, do not write"
    )
    args = parser.parse_args()

    inputs = tuple(args.input) if args.input else evoforge.default_input_paths()
    output = args.output or evoforge.default_output_path()
    params = evoforge.ForgeParams(min_samples=args.min_samples, top_k=args.top_k)

    rows = evoforge.load_rows(inputs)
    if not rows:
        print("EvoForge: no trajectory rows found — nothing to do", file=sys.stderr)
        return 0
    stats = evoforge.aggregate(rows, params)
    policy = evoforge.build_policy(stats, params)

    if args.dry_run:
        json.dump(policy, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    evoforge.write_policy(policy, output)
    n_policies = len(policy["policies"])
    print(
        f"EvoForge: {len(rows)} rows → {n_policies} skill policies → {output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
