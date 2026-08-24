import pytest

from smoke.lib.config import SmokeConfig
from smoke.lib.local_providers import (
    first_local_provider_model_id,
    local_provider_api_key,
)


@pytest.mark.live
@pytest.mark.smoke_target("lmstudio")
def test_lmstudio_models_endpoint_when_available(smoke_config: SmokeConfig) -> None:
    first_local_provider_model_id(
        "lmstudio",
        smoke_config.settings.lm_studio_base_url,
        timeout_s=smoke_config.timeout_s,
    )


@pytest.mark.live
@pytest.mark.smoke_target("llamacpp")
def test_llamacpp_models_endpoint_when_available(smoke_config: SmokeConfig) -> None:
    first_local_provider_model_id(
        "llamacpp",
        smoke_config.settings.llamacpp_base_url,
        timeout_s=smoke_config.timeout_s,
    )


@pytest.mark.live
@pytest.mark.smoke_target("ollama")
def test_ollama_models_endpoint_when_available(smoke_config: SmokeConfig) -> None:
    first_local_provider_model_id(
        "ollama",
        smoke_config.settings.ollama_base_url,
        timeout_s=smoke_config.timeout_s,
    )


@pytest.mark.live
@pytest.mark.smoke_target("omlx")
def test_omlx_models_endpoint_when_available(smoke_config: SmokeConfig) -> None:
    first_local_provider_model_id(
        "omlx",
        smoke_config.settings.omlx_base_url,
        timeout_s=smoke_config.timeout_s,
        api_key=local_provider_api_key("omlx", smoke_config.settings),
    )
