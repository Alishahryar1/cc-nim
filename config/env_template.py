"""Canonical env template loading for init and Admin UI defaults."""

from __future__ import annotations

import importlib.resources
from pathlib import Path


def load_env_template() -> str:
    """Load the root ``.env.example`` template from wheel resources or checkout."""

    packaged = importlib.resources.files("config").joinpath("env.example")
    if packaged.is_file():
        return packaged.read_text("utf-8")

    source_template = Path(__file__).resolve().parents[1] / ".env.example"
    if source_template.is_file():
        return source_template.read_text(encoding="utf-8")

    raise FileNotFoundError("Could not find bundled or source .env.example template.")


def load_env_template_or_empty() -> str:
    """Return the env template, or an empty template when unavailable."""

    try:
        return load_env_template()
    except FileNotFoundError:
        return ""
