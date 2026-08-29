"""Executable contracts for Pi's bundled TypeScript extension."""

import json
import shutil
import subprocess

import pytest

from free_claude_code.cli.launchers.pi import pi_extension_path


def test_pi_extension_projects_known_capabilities_and_preserves_unknown_defaults() -> (
    None
):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    payload = {
        "object": "list",
        "data": [
            {
                "id": "provider/vision-reasoning",
                "provider_model_ref": "provider/vision-reasoning",
                "supportsReasoning": True,
                "inputModalities": ["text", "image"],
                "contextWindow": 131072,
                "maxCompletionTokens": 8192,
            },
            {
                "id": "claude-3-freecc-no-thinking/provider/text-only",
                "provider_model_ref": "provider/text-only",
                "supportsReasoning": False,
                "inputModalities": ["text"],
                "contextWindow": 65536,
            },
            {
                "id": "provider/unknown",
                "provider_model_ref": "provider/unknown",
            },
        ],
    }
    script = """
const { projectFccModels } = await import(process.argv[1]);
const payload = JSON.parse(process.argv[2]);
console.log(JSON.stringify(projectFccModels(payload)));
"""

    result = subprocess.run(
        [
            node,
            "--experimental-strip-types",
            "--input-type=module",
            "--eval",
            script,
            pi_extension_path().as_uri(),
            json.dumps(payload),
        ],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    projected = json.loads(result.stdout)
    assert [(model["reasoning"], model["input"]) for model in projected] == [
        (True, ["text", "image"]),
        (False, ["text"]),
        (True, ["text"]),
    ]
    assert [(model["contextWindow"], model["maxTokens"]) for model in projected] == [
        (131072, 8192),
        (65536, 16384),
        (128000, 16384),
    ]
