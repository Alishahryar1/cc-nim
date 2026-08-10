import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit


def test_architecture_document_exists() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert (repo_root / "ARCHITECTURE.md").is_file()


def test_architecture_document_relative_links_resolve() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    architecture = repo_root / "ARCHITECTURE.md"
    text = architecture.read_text(encoding="utf-8")

    missing: list[str] = []
    for match in re.finditer(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text):
        raw_target = match.group(1).strip()
        target = raw_target.split("#", 1)[0]
        if not target or urlsplit(target).scheme:
            continue
        if not (repo_root / unquote(target)).exists():
            missing.append(raw_target)

    assert missing == []


def test_root_env_example_is_the_single_template_source() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    root_example = repo_root / ".env.example"
    duplicate_example = (
        repo_root / "src" / "free_claude_code" / "config" / "env.example"
    )

    assert root_example.is_file()
    assert not duplicate_example.exists()


def test_root_env_example_is_packaged_for_config_template_loader() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text("utf-8"))

    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"][
        "force-include"
    ]

    assert force_include[".env.example"] == "free_claude_code/config/env.example"


def test_admin_template_defaults_align_with_settings_for_compact_reliability() -> None:
    """Admin Apply seeds ~/.fcc/.env from .env.example + manifest; keep them
    aligned with Settings so Apply cannot reintroduce RATE_LIMIT=1."""
    from free_claude_code.config.admin.manifest import FIELD_BY_KEY
    from free_claude_code.config.settings import Settings

    repo_root = Path(__file__).resolve().parents[2]
    example = (repo_root / ".env.example").read_text(encoding="utf-8")
    settings = Settings.model_construct()

    expected = {
        "PROVIDER_RATE_LIMIT": str(settings.provider_rate_limit),
        "PROVIDER_RATE_WINDOW": str(settings.provider_rate_window),
        "HTTP_READ_TIMEOUT": str(int(settings.http_read_timeout))
        if float(settings.http_read_timeout).is_integer()
        else str(settings.http_read_timeout),
        "HTTP_WRITE_TIMEOUT": str(int(settings.http_write_timeout))
        if float(settings.http_write_timeout).is_integer()
        else str(settings.http_write_timeout),
    }
    for key, value in expected.items():
        assert FIELD_BY_KEY[key].default == value, key
        assert re.search(rf"(?m)^{re.escape(key)}={re.escape(value)}$", example), key


def test_pyproject_first_party_packages_match_packaged_roots() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"known-first-party = \[(?P<items>[^\]]+)\]", pyproject)

    assert match is not None
    configured = {
        item.strip().strip('"')
        for item in match.group("items").split(",")
        if item.strip()
    }
    expected = {"free_claude_code", "smoke"}
    assert configured == expected
