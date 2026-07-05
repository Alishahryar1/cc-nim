from pathlib import Path
from types import SimpleNamespace

from smoke.conftest import (
    DISABLED_PROVIDER_MODEL,
    provider_model_params,
    provider_xdist_group,
)
from smoke.lib.config import (
    ALL_TARGETS,
    DEFAULT_TARGETS,
    NVIDIA_NIM_CLI_DEFAULT_MODELS,
    OPENROUTER_FREE_CLI_DEFAULT_MODELS,
    OPT_IN_TARGETS,
    PROVIDER_SMOKE_DEFAULT_MODELS,
    TARGET_REQUIRED_ENV,
    SmokeConfig,
    nvidia_nim_cli_model_refs,
    openrouter_free_cli_model_refs,
)


def _settings(**overrides):
    values = {
        "model": "ollama/llama3.1",
        "model_opus": None,
        "model_sonnet": None,
        "model_haiku": None,
        "nvidia_nim_api_key": "",
        "open_router_api_key": "",
        "mistral_api_key": "",
        "codestral_api_key": "",
        "deepseek_api_key": "",
        "kimi_api_key": "",
        "wafer_api_key": "",
        "minimax_api_key": "",
        "opencode_api_key": "",
        "zai_api_key": "",
        "gemini_api_key": "",
        "groq_api_key": "",
        "cerebras_api_key": "",
        "fireworks_api_key": "",
        "cloudflare_api_token": "",
        "cloudflare_account_id": "",
        "lm_studio_base_url": "",
        "llamacpp_base_url": "",
        "ollama_base_url": "http://localhost:11434",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _smoke_config(**overrides) -> SmokeConfig:
    values = {
        "root": Path("."),
        "results_dir": Path(".smoke-results"),
        "live": False,
        "interactive": False,
        "targets": DEFAULT_TARGETS,
        "provider_matrix": frozenset(),
        "timeout_s": 45.0,
        "prompt": "Reply with exactly: FCC_SMOKE_PONG",
        "claude_bin": "claude",
        "worker_id": "main",
        "settings": _settings(),
    }
    values.update(overrides)
    return SmokeConfig(**values)


def test_ollama_is_default_smoke_target() -> None:
    assert "ollama" in DEFAULT_TARGETS
    assert "ollama" in TARGET_REQUIRED_ENV


def test_nvidia_nim_cli_is_opt_in_smoke_target() -> None:
    assert "nvidia_nim_cli" not in DEFAULT_TARGETS
    assert "nvidia_nim_cli" in OPT_IN_TARGETS
    assert "nvidia_nim_cli" in ALL_TARGETS
    assert "nvidia_nim_cli" in TARGET_REQUIRED_ENV
    assert "openrouter_free_cli" not in DEFAULT_TARGETS
    assert "openrouter_free_cli" in OPT_IN_TARGETS
    assert "openrouter_free_cli" in ALL_TARGETS
    assert "openrouter_free_cli" in TARGET_REQUIRED_ENV


def test_ollama_provider_configuration_uses_base_url() -> None:
    config = _smoke_config()

    assert config.has_provider_configuration("ollama")
    assert config.provider_models()[0].full_model == "ollama/llama3.1"


def test_ollama_provider_matrix_filters_models() -> None:
    config = _smoke_config(provider_matrix=frozenset({"ollama"}))

    assert [model.provider for model in config.provider_models()] == ["ollama"]


def test_provider_smoke_models_cover_configured_providers_independent_of_model_mapping(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_DEEPSEEK", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            deepseek_api_key="deepseek-key",
            ollama_base_url="",
        )
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["deepseek"]
    assert models[0].full_model == PROVIDER_SMOKE_DEFAULT_MODELS["deepseek"]
    assert models[0].source == "provider_default"


def test_openrouter_provider_smoke_uses_concrete_free_model(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_OPEN_ROUTER", raising=False)
    config = _smoke_config(
        settings=_settings(open_router_api_key="openrouter-key", ollama_base_url="")
    )

    models = config.provider_smoke_models()

    assert [model.provider for model in models] == ["open_router"]
    assert models[0].full_model == "open_router/moonshotai/kimi-k2.6:free"
    assert models[0].source == "provider_default"


def test_wafer_provider_configuration_uses_api_key(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_WAFER", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            wafer_api_key="wafer-key",
        )
    )

    assert config.has_provider_configuration("wafer")
    models = config.provider_smoke_models()
    assert models[0].provider == "wafer"
    assert models[0].full_model == PROVIDER_SMOKE_DEFAULT_MODELS["wafer"]


def test_minimax_provider_configuration_uses_api_key(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_MINIMAX", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            minimax_api_key="minimax-key",
        )
    )

    assert config.has_provider_configuration("minimax")
    models = config.provider_smoke_models()
    assert models[0].provider == "minimax"
    assert models[0].full_model == PROVIDER_SMOKE_DEFAULT_MODELS["minimax"]


def test_cloudflare_provider_configuration_requires_token_and_account(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_CLOUDFLARE", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="ollama/llama3.1",
            ollama_base_url="",
            cloudflare_api_token="cf-token",
            cloudflare_account_id="cf-account",
        )
    )

    assert config.has_provider_configuration("cloudflare")
    models = config.provider_smoke_models()
    assert models[0].provider == "cloudflare"
    assert models[0].full_model == PROVIDER_SMOKE_DEFAULT_MODELS["cloudflare"]


def test_cloudflare_provider_configuration_missing_account_is_unconfigured() -> None:
    config = _smoke_config(
        settings=_settings(
            ollama_base_url="",
            cloudflare_api_token="cf-token",
            cloudflare_account_id="",
        )
    )

    assert not config.has_provider_configuration("cloudflare")


def test_provider_smoke_model_override_accepts_model_name_without_prefix(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FCC_SMOKE_MODEL_DEEPSEEK", "deepseek-reasoner")
    config = _smoke_config(
        settings=_settings(
            deepseek_api_key="deepseek-key",
            ollama_base_url="",
        ),
        provider_matrix=frozenset({"deepseek"}),
    )

    models = config.provider_smoke_models()

    assert models[0].full_model == "deepseek/deepseek-reasoner"
    assert models[0].source == "FCC_SMOKE_MODEL_DEEPSEEK"


def test_provider_smoke_model_override_accepts_owner_model_name(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "FCC_SMOKE_MODEL_NVIDIA_NIM", "nvidia/nemotron-3-super-120b-a12b"
    )
    config = _smoke_config(
        settings=_settings(
            model="deepseek/deepseek-chat",
            deepseek_api_key="",
            nvidia_nim_api_key="nim-key",
            ollama_base_url="",
        ),
        provider_matrix=frozenset({"nvidia_nim"}),
    )

    models = config.provider_smoke_models()

    assert models[0].full_model == "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"
    assert models[0].source == "FCC_SMOKE_MODEL_NVIDIA_NIM"


def test_provider_smoke_model_override_rejects_wrong_provider_prefix(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FCC_SMOKE_MODEL_DEEPSEEK", "ollama/llama3.1")
    config = _smoke_config(
        settings=_settings(
            deepseek_api_key="deepseek-key",
            ollama_base_url="",
        ),
        provider_matrix=frozenset({"deepseek"}),
    )

    try:
        config.provider_smoke_models()
    except ValueError as exc:
        assert "FCC_SMOKE_MODEL_DEEPSEEK" in str(exc)
    else:
        raise AssertionError("expected wrong provider prefix to fail")


def test_provider_smoke_matrix_filters_provider_catalog(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_DEEPSEEK", raising=False)
    config = _smoke_config(
        settings=_settings(
            deepseek_api_key="deepseek-key",
            nvidia_nim_api_key="nim-key",
            ollama_base_url="",
        ),
        provider_matrix=frozenset({"nvidia_nim"}),
    )

    assert [model.provider for model in config.provider_smoke_models()] == [
        "nvidia_nim"
    ]


def test_provider_smoke_collection_params_are_grouped_by_provider(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_DEEPSEEK", raising=False)
    monkeypatch.delenv("FCC_SMOKE_MODEL_NVIDIA_NIM", raising=False)
    config = _smoke_config(
        live=True,
        settings=_settings(
            deepseek_api_key="deepseek-key",
            nvidia_nim_api_key="nim-key",
            ollama_base_url="",
        ),
    )

    params = provider_model_params(config)

    assert [param.id for param in params] == ["nvidia_nim", "deepseek"]
    groups = [
        mark.args[0]
        for param in params
        for mark in param.marks
        if mark.name == "xdist_group"
    ]
    assert groups == ["provider:nvidia_nim", "provider:deepseek"]


def test_provider_smoke_collection_uses_disabled_placeholder_when_not_live() -> None:
    config = _smoke_config(live=False, settings=_settings(ollama_base_url=""))

    params = provider_model_params(config)

    assert [param.values[0] for param in params] == [DISABLED_PROVIDER_MODEL]
    assert provider_xdist_group(DISABLED_PROVIDER_MODEL) == "provider:smoke_disabled"


def test_provider_smoke_includes_local_provider_when_model_mapping_uses_it(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_OLLAMA", raising=False)
    config = _smoke_config()

    assert [model.provider for model in config.provider_smoke_models()] == ["ollama"]


def test_provider_smoke_does_not_include_default_local_urls_when_unmapped(
    monkeypatch,
) -> None:
    monkeypatch.delenv("FCC_SMOKE_MODEL_OLLAMA", raising=False)
    config = _smoke_config(settings=_settings(model="nvidia_nim/test"))

    assert config.provider_smoke_models() == []


def test_nvidia_nim_cli_default_models_are_normalized() -> None:
    refs = nvidia_nim_cli_model_refs({})

    assert tuple(refs) == tuple(
        f"nvidia_nim/{model}" for model in NVIDIA_NIM_CLI_DEFAULT_MODELS
    )
    assert "nvidia_nim/deepseek-ai/deepseek-v4-pro" in refs
    assert "nvidia_nim/deepseek-ai/deepseek-v4-flash" in refs
    assert set(refs.values()) == {"nvidia_nim_cli_default"}


def test_nvidia_nim_cli_models_override_and_append() -> None:
    refs = nvidia_nim_cli_model_refs(
        {
            "FCC_SMOKE_NIM_MODELS": "z-ai/glm-5.1,nvidia_nim/custom/model",
            "FCC_SMOKE_NIM_EXTRA_MODELS": "moonshotai/kimi-k2.6,z-ai/glm-5.1",
        }
    )

    assert tuple(refs) == (
        "nvidia_nim/z-ai/glm-5.1",
        "nvidia_nim/custom/model",
        "nvidia_nim/moonshotai/kimi-k2.6",
    )
    assert refs["nvidia_nim/z-ai/glm-5.1"] == "FCC_SMOKE_NIM_MODELS"
    assert refs["nvidia_nim/moonshotai/kimi-k2.6"] == ("FCC_SMOKE_NIM_EXTRA_MODELS")


def test_nvidia_nim_cli_models_reject_empty_override() -> None:
    try:
        nvidia_nim_cli_model_refs({"FCC_SMOKE_NIM_MODELS": " , "})
    except ValueError as exc:
        assert "FCC_SMOKE_NIM_MODELS" in str(exc)
    else:
        raise AssertionError("expected empty NVIDIA NIM CLI model override to fail")


def test_nvidia_nim_cli_models_reject_wrong_provider_prefix() -> None:
    try:
        nvidia_nim_cli_model_refs({"FCC_SMOKE_NIM_MODELS": "open_router/model"})
    except ValueError as exc:
        assert "nvidia_nim" in str(exc)
    else:
        raise AssertionError("expected wrong provider prefix to fail")


def test_smoke_config_returns_nvidia_nim_cli_provider_models(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_NIM_MODELS", raising=False)
    monkeypatch.delenv("FCC_SMOKE_NIM_EXTRA_MODELS", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="nvidia_nim/z-ai/glm-5.1",
            nvidia_nim_api_key="nim-key",
            ollama_base_url="",
        )
    )

    models = config.nvidia_nim_cli_models()

    assert models[0].provider == "nvidia_nim"
    assert models[0].full_model == "nvidia_nim/z-ai/glm-5.1"
    assert models[0].source == "nvidia_nim_cli_default"


def test_openrouter_free_cli_default_models_are_normalized() -> None:
    refs = openrouter_free_cli_model_refs({})

    assert tuple(refs) == tuple(
        f"open_router/{model}" for model in OPENROUTER_FREE_CLI_DEFAULT_MODELS
    )
    assert "open_router/nvidia/nemotron-3-super-120b-a12b:free" in refs
    assert "open_router/poolside/laguna-m.1:free" in refs
    assert set(refs.values()) == {"openrouter_free_cli_default"}


def test_openrouter_free_cli_models_override_and_append() -> None:
    refs = openrouter_free_cli_model_refs(
        {
            "FCC_SMOKE_OPENROUTER_FREE_MODELS": (
                "openai/gpt-oss-120b:free,open_router/custom/model:free"
            ),
            "FCC_SMOKE_OPENROUTER_FREE_EXTRA_MODELS": (
                "poolside/laguna-m.1:free,openai/gpt-oss-120b:free"
            ),
        }
    )

    assert tuple(refs) == (
        "open_router/openai/gpt-oss-120b:free",
        "open_router/custom/model:free",
        "open_router/poolside/laguna-m.1:free",
    )
    assert refs["open_router/openai/gpt-oss-120b:free"] == (
        "FCC_SMOKE_OPENROUTER_FREE_MODELS"
    )
    assert refs["open_router/poolside/laguna-m.1:free"] == (
        "FCC_SMOKE_OPENROUTER_FREE_EXTRA_MODELS"
    )


def test_openrouter_free_cli_models_reject_empty_override() -> None:
    try:
        openrouter_free_cli_model_refs({"FCC_SMOKE_OPENROUTER_FREE_MODELS": " , "})
    except ValueError as exc:
        assert "FCC_SMOKE_OPENROUTER_FREE_MODELS" in str(exc)
    else:
        raise AssertionError("expected empty OpenRouter free CLI override to fail")


def test_openrouter_free_cli_models_reject_wrong_provider_prefix() -> None:
    try:
        openrouter_free_cli_model_refs(
            {"FCC_SMOKE_OPENROUTER_FREE_MODELS": "nvidia_nim/model"}
        )
    except ValueError as exc:
        assert "open_router" in str(exc)
    else:
        raise AssertionError("expected wrong provider prefix to fail")


def test_smoke_config_returns_openrouter_free_cli_provider_models(monkeypatch) -> None:
    monkeypatch.delenv("FCC_SMOKE_OPENROUTER_FREE_MODELS", raising=False)
    monkeypatch.delenv("FCC_SMOKE_OPENROUTER_FREE_EXTRA_MODELS", raising=False)
    config = _smoke_config(
        settings=_settings(
            model="open_router/openai/gpt-oss-120b:free",
            open_router_api_key="openrouter-key",
            ollama_base_url="",
        )
    )

    models = config.openrouter_free_cli_models()

    assert models[0].provider == "open_router"
    assert models[0].full_model == "open_router/nvidia/nemotron-3-super-120b-a12b:free"
    assert models[0].source == "openrouter_free_cli_default"
