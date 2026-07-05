import re
from pathlib import Path

from messaging.platforms.factory import create_messaging_components
from providers.base import BaseProvider
from providers.cerebras import CerebrasProvider
from providers.cloudflare import CloudflareProvider
from providers.codestral import CodestralProvider
from providers.deepseek import DeepSeekProvider
from providers.fireworks import FireworksProvider
from providers.gemini import GeminiProvider
from providers.groq import GroqProvider
from providers.kimi import KimiProvider
from providers.llamacpp import LlamaCppProvider
from providers.lmstudio import LMStudioProvider
from providers.minimax import MiniMaxProvider
from providers.mistral import MistralProvider
from providers.nvidia_nim import NvidiaNimProvider
from providers.ollama import OllamaProvider
from providers.open_router import OpenRouterProvider
from providers.opencode import OpenCodeProvider
from providers.wafer import WaferProvider
from providers.zai import ZaiProvider
from smoke.features import FEATURE_INVENTORY, README_FEATURES, feature_ids

VALID_SOURCE = {"readme", "public_surface"}


def test_every_readme_feature_has_inventory_entry() -> None:
    missing = sorted(set(README_FEATURES) - feature_ids(source="readme"))
    extra_readme = sorted(feature_ids(source="readme") - set(README_FEATURES))
    assert not missing, f"README features missing inventory entries: {missing}"
    assert not extra_readme, (
        f"README inventory entries not in README_FEATURES: {extra_readme}"
    )


def test_feature_inventory_is_unique_and_decision_complete() -> None:
    ids = [feature.feature_id for feature in FEATURE_INVENTORY]
    assert len(ids) == len(set(ids))
    assert "claude_pick" not in ids

    for feature in FEATURE_INVENTORY:
        assert feature.source in VALID_SOURCE, feature
        assert feature.title.strip(), feature
        assert feature.skip_policy.strip(), feature
        assert feature.pytest_contract_tests, feature
        assert feature.has_pytest_coverage, feature
        if feature.product_e2e_tests:
            assert feature.smoke_targets, feature
            assert not feature.product_e2e_reason, feature
        else:
            assert feature.product_e2e_reason.strip(), feature
        if feature.live_prereq_tests:
            assert feature.smoke_targets, feature


def test_feature_inventory_test_owners_exist() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pytest_names = _collect_test_names(repo_root / "tests")
    smoke_names = _collect_test_names(repo_root / "smoke")

    for feature in FEATURE_INVENTORY:
        for owner in feature.pytest_contract_tests:
            _assert_owner_exists(owner, repo_root, pytest_names)
        for owner in feature.live_prereq_tests + feature.product_e2e_tests:
            assert owner in smoke_names or owner in pytest_names, (feature, owner)


def test_product_coverage_is_not_satisfied_by_prereq_probes() -> None:
    for feature in FEATURE_INVENTORY:
        overlap = set(feature.live_prereq_tests) & set(feature.product_e2e_tests)
        assert not overlap, (feature.feature_id, sorted(overlap))
        if feature.product_e2e_tests:
            assert all("_e2e" in name for name in feature.product_e2e_tests), feature


def test_provider_and_platform_registries_include_advertised_builtins() -> None:
    provider_classes = {
        "nvidia_nim": NvidiaNimProvider,
        "open_router": OpenRouterProvider,
        "mistral": MistralProvider,
        "mistral_codestral": CodestralProvider,
        "deepseek": DeepSeekProvider,
        "kimi": KimiProvider,
        "minimax": MiniMaxProvider,
        "fireworks": FireworksProvider,
        "cloudflare": CloudflareProvider,
        "lmstudio": LMStudioProvider,
        "llamacpp": LlamaCppProvider,
        "ollama": OllamaProvider,
        "wafer": WaferProvider,
        "opencode": OpenCodeProvider,
        "opencode_go": OpenCodeProvider,
        "zai": ZaiProvider,
        "gemini": GeminiProvider,
        "groq": GroqProvider,
        "cerebras": CerebrasProvider,
    }
    for provider_class in provider_classes.values():
        assert issubclass(provider_class, BaseProvider)

    assert create_messaging_components("not-a-platform") is None


def _collect_test_names(root: Path) -> set[str]:
    names: set[str] = set()
    for path in root.rglob("test_*.py"):
        text = path.read_text(encoding="utf-8")
        names.update(re.findall(r"^\s*(?:async\s+)?def (test_[^(]+)", text, re.M))
    return names


def _assert_owner_exists(owner: str, repo_root: Path, test_names: set[str]) -> None:
    if owner.startswith("test_"):
        assert owner in test_names, owner
        return

    path_part, _, node_name = owner.partition("::")
    path = repo_root / path_part
    assert path.exists(), owner
    if node_name:
        assert node_name in test_names, owner
