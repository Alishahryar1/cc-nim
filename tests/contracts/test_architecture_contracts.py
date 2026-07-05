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


def test_smoke_lib_has_no_sse_shim_module() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert not (repo_root / "smoke" / "lib" / "sse.py").exists()


def test_api_package_exports() -> None:
    import api

    assert set(api.__all__) == {
        "MessagesRequest",
        "MessagesResponse",
        "TokenCountRequest",
        "TokenCountResponse",
        "create_app",
    }


def test_root_env_example_is_the_single_template_source() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    root_example = repo_root / ".env.example"
    duplicate_example = repo_root / "config" / "env.example"

    assert root_example.is_file()
    assert not duplicate_example.exists()


def test_root_env_example_is_packaged_for_config_template_loader() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text("utf-8"))

    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"][
        "force-include"
    ]

    assert force_include[".env.example"] == "config/env.example"


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
    expected = {"api", "cli", "config", "core", "messaging", "providers", "smoke"}
    assert configured == expected
