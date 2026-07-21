from pathlib import Path

from free_claude_code.config.env_template import load_env_template


def test_env_template_loader_uses_root_template_in_source_checkout() -> None:
    """Source checkout fallback uses the root .env.example as the single source."""

    template = (Path(__file__).resolve().parents[2] / ".env.example").read_text(
        encoding="utf-8"
    )

    assert load_env_template() == template
