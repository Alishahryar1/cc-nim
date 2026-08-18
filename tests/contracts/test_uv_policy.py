import tomllib
from pathlib import Path

UV_MINIMUM = "0.11.16"
UV_WORKFLOWS = (
    Path(".github/workflows/tests.yml"),
    Path(".github/workflows/dependency-cache.yml"),
)


def test_supported_uv_minimum_is_consistent() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    install_sh = Path("scripts/install.sh").read_text(encoding="utf-8")
    install_ps1 = Path("scripts/install.ps1").read_text(encoding="utf-8")

    assert pyproject["tool"]["uv"]["required-version"] == f">={UV_MINIMUM}"
    assert f'MIN_UV_VERSION="{UV_MINIMUM}"' in install_sh
    assert f'$MinUvVersion = "{UV_MINIMUM}"' in install_ps1
    for workflow_path in UV_WORKFLOWS:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert f'version: "{UV_MINIMUM}"' in workflow


def test_every_uv_workflow_inherits_malware_check() -> None:
    workflow_env = (
        'env:\n  UV_MALWARE_CHECK: "1"\n  UV_PREVIEW_FEATURES: "malware-check"\n'
    )

    for workflow_path in UV_WORKFLOWS:
        workflow = workflow_path.read_text(encoding="utf-8")
        assert workflow.count(workflow_env) == 1
        assert workflow.index(workflow_env) < workflow.index("jobs:\n")
