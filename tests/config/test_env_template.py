from io import StringIO
from pathlib import Path

from dotenv import dotenv_values

from free_claude_code.config.admin.manifest import FIELD_BY_KEY
from free_claude_code.config.constants import (
    PROVIDER_MAX_CONCURRENCY_DEFAULT,
    PROVIDER_RATE_LIMIT_DEFAULT,
    PROVIDER_RATE_WINDOW_DEFAULT,
)
from free_claude_code.config.env_template import load_env_template


def test_env_template_loader_uses_root_template_in_source_checkout() -> None:
    """Source checkout fallback uses the root .env.example as the single source."""

    template = (Path(__file__).resolve().parents[2] / ".env.example").read_text(
        encoding="utf-8"
    )

    assert load_env_template() == template


def test_provider_admission_defaults_match_every_config_surface() -> None:
    """Runtime, Admin, and the generated env template expose one contract."""

    expected = {
        "PROVIDER_RATE_LIMIT": str(PROVIDER_RATE_LIMIT_DEFAULT),
        "PROVIDER_RATE_WINDOW": str(PROVIDER_RATE_WINDOW_DEFAULT),
        "PROVIDER_MAX_CONCURRENCY": str(PROVIDER_MAX_CONCURRENCY_DEFAULT),
    }
    template = dotenv_values(stream=StringIO(load_env_template()))

    assert {key: template[key] for key in expected} == expected
    assert {key: FIELD_BY_KEY[key].default for key in expected} == expected
