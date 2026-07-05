"""Small JSON report writer for smoke runs."""

import json
import time
from dataclasses import asdict, dataclass

from .config import SmokeConfig, redacted


@dataclass(slots=True)
class SmokeOutcome:
    nodeid: str
    outcome: str
    classification: str
    duration_s: float
    markers: list[str]
    detail: str


class SmokeReport:
    def __init__(self, config: SmokeConfig) -> None:
        self.config = config
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.outcomes: list[SmokeOutcome] = []

    def add(
        self,
        *,
        nodeid: str,
        outcome: str,
        duration_s: float,
        markers: list[str],
        detail: str = "",
    ) -> None:
        self.outcomes.append(
            SmokeOutcome(
                nodeid=nodeid,
                outcome=outcome,
                classification=classify_outcome(
                    nodeid=nodeid, outcome=outcome, detail=detail
                ),
                duration_s=duration_s,
                markers=markers,
                detail=redacted(detail),
            )
        )

    def write(self) -> None:
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        path = (
            self.config.results_dir
            / f"report-{self.config.worker_id}-{int(time.time())}.json"
        )
        payload = {
            "started_at": self.started_at,
            "worker_id": self.config.worker_id,
            "targets": sorted(self.config.targets),
            "outcomes": [asdict(outcome) for outcome in self.outcomes],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def classify_outcome(*, nodeid: str, outcome: str, detail: str) -> str:
    """Classify smoke outcomes for triage reports."""
    if outcome == "passed":
        return "passed"

    text = f"{nodeid}\n{detail}".lower()
    if outcome == "skipped":
        if "smoke target disabled" in text:
            return "target_disabled"
        if "missing_env" in text:
            return "missing_env"
        if any(
            marker in text
            for marker in (
                "upstream_unavailable",
                "connection refused",
                "connecterror",
                "readtimeout",
                "timed out",
                "not reachable",
            )
        ):
            return "upstream_unavailable"
        return "missing_env"

    if "harness_bug" in text:
        return "harness_bug"
    if "missing_env" in text:
        return "missing_env"
    if any(
        marker in text
        for marker in (
            "upstream_unavailable",
            "connection refused",
            "connecterror",
            "readtimeout",
            "timed out",
            "not reachable",
            "upstream provider",
        )
    ):
        return "upstream_unavailable"
    return "product_failure"
