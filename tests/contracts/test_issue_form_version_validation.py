import json
import re
from pathlib import Path

import pytest

BUG_FORM = Path(".github/ISSUE_TEMPLATE/bug-report.yml")
WORKFLOW = Path(".github/workflows/validate-bug-report-version.yml")


def _workflow_pattern(name: str) -> str:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(rf"const {name} = (\"[^\"]+\");", workflow)
    assert match is not None
    return json.loads(match.group(1))


def test_bug_form_requires_an_exact_version_or_none() -> None:
    form = BUG_FORM.read_text(encoding="utf-8")

    assert "Run `fcc-server --version`" in form
    assert "enter only the version number" in form
    assert "enter `None`" in form
    assert 'placeholder: "4.6.1 or None"' in form
    assert "not installed" not in form


@pytest.mark.parametrize("value", ["0.0.0", "4.6.1", "123.45.678", "None"])
def test_version_pattern_accepts_supported_values(value: str) -> None:
    assert re.fullmatch(_workflow_pattern("versionPattern"), value)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "latest",
        "4.6",
        "4.6.1.2",
        "v4.6.1",
        "4.6.x",
        "none",
        "free-claude-code 4.6.1",
    ],
)
def test_version_pattern_rejects_ambiguous_values(value: str) -> None:
    assert re.fullmatch(_workflow_pattern("versionPattern"), value) is None


def test_field_pattern_extracts_the_issue_form_value() -> None:
    body = """### FCC version

4.6.1

### CLI

Claude Code (fcc-claude)
"""

    match = re.search(_workflow_pattern("fieldPattern"), body, flags=re.MULTILINE)

    assert match is not None
    assert match.group(1) == "4.6.1"


def test_workflow_owns_one_idempotent_triage_state() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "types: [opened, edited]" in workflow
    assert "issues: write" in workflow
    assert "needs-fcc-version" in workflow
    assert "<!-- fcc-version-validator -->" in workflow
    assert "github.rest.issues.createLabel" in workflow
    assert "github.rest.issues.addLabels" in workflow
    assert "github.rest.issues.removeLabel" in workflow
    assert "comments.some" in workflow
