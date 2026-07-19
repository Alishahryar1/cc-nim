"""Catalog-derived Admin provider fields."""

from typing import Any

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings

_PROVIDER_FIELD_OVERRIDES: dict[str, dict[str, Any]] = {
    "NVIDIA_NIM_API_KEY": {
        "label": "NVIDIA NIM API Key",
        "description": "Used by NVIDIA NIM chat and optional NIM voice transcription.",
    },
    "MISTRAL_API_KEY": {
        "label": "Mistral API Key",
        "description": (
            "Mistral La Plateforme (api.mistral.ai); Experiment plan is free tier with rate limits."
        ),
    },
    "CODESTRAL_API_KEY": {
        "label": "Codestral API Key",
        "description": (
            "Mistral Codestral endpoint (codestral.mistral.ai); distinct from Mistral "
            "La Plateforme ``MISTRAL_API_KEY``. See Mistral docs for coding/FIM domains."
        ),
    },
    "OPENCODE_API_KEY": {
        "label": "OpenCode API Key",
        "description": (
            "OpenCode Zen curated gateway (opencode.ai/zen/v1) and OpenCode Go subscription "
            "gateway (opencode.ai/zen/go/v1); single key from opencode.ai/auth."
        ),
    },
    "AI_GATEWAY_API_KEY": {
        "label": "Vercel AI Gateway API Key",
        "description": (
            "Vercel AI Gateway API key for the OpenAI-compatible endpoint at "
            "ai-gateway.vercel.sh/v1."
        ),
    },
    "AWS_BEARER_TOKEN_BEDROCK": {
        "label": "Amazon Bedrock API Key",
        "description": (
            "Amazon Bedrock bearer API key for the region-specific Mantle "
            "OpenAI-compatible endpoint."
        ),
    },
    "BEDROCK_BASE_URL": {
        "description": (
            "Amazon Bedrock Mantle OpenAI base URL for the same region as the "
            "API key and selected models."
        ),
    },
    "HUGGINGFACE_API_KEY": {
        "label": "Hugging Face API Key",
        "description": (
            "Hugging Face token with Inference Providers permission; also used "
            "for local Whisper model downloads when voice notes need gated models."
        ),
    },
    "COHERE_API_KEY": {
        "label": "Cohere API Key",
        "description": "Cohere API key for the OpenAI-compatible Compatibility API.",
    },
    "GITHUB_MODELS_TOKEN": {
        "label": "GitHub Models Token",
        "description": (
            "GitHub token with Models access for the OpenAI-compatible inference API "
            "at models.github.ai."
        ),
    },
    "ZAI_API_KEY": {
        "label": "Z.ai API Key",
        "description": "Z.ai Coding Plan API key.",
    },
    "FIREWORKS_API_KEY": {
        "label": "Fireworks API Key",
        "description": "Fireworks AI inference API key.",
    },
    "MINIMAX_API_KEY": {
        "label": "MiniMax API Key",
        "description": (
            "MiniMax API key for the OpenAI-compatible Chat Completions API at "
            "free_claude_code.api.minimax.io/v1."
        ),
    },
    "KIMI_CODE_API_KEY": {
        "label": "Kimi Code API Key",
        "description": (
            "Personal Kimi Code subscription key from kimi.com/code/console; "
            "separate from KIMI_API_KEY credits on the Kimi API platform."
        ),
    },
    "CLOUDFLARE_API_TOKEN": {
        "label": "Cloudflare API Token",
        "description": (
            "Cloudflare API token for account-scoped AI REST requests. "
            "Use with CLOUDFLARE_ACCOUNT_ID."
        ),
    },
    "GEMINI_API_KEY": {
        "label": "Gemini API Key",
        "description": (
            "Google AI Studio Gemini API key (Google AI Studio / Gemini API "
            "[OpenAI-compatible](https://ai.google.dev/gemini-api/docs/openai)); "
            "free tier has per-model rate limits and data may be used for improvement "
            "outside the UK/CH/EEA/EU."
        ),
    },
    "GROQ_API_KEY": {
        "label": "Groq API Key",
        "description": (
            "GroqCloud OpenAI-compatible API key ([console.groq.com/keys]("
            "https://console.groq.com/keys)); see Groq "
            "[OpenAI compatibility docs](https://console.groq.com/docs/openai)."
        ),
    },
    "SAMBANOVA_API_KEY": {
        "label": "SambaNova API Key",
        "description": (
            "SambaNova Cloud OpenAI-compatible API key (create at "
            "[cloud.sambanova.ai/apis](https://cloud.sambanova.ai/apis))."
        ),
    },
    "CEREBRAS_API_KEY": {
        "label": "Cerebras API Key",
        "description": (
            "Cerebras Inference API key (create in [Cloud Console](https://cloud.cerebras.ai)); "
            "see [Quickstart](https://inference-docs.cerebras.ai/quickstart) and "
            "[OpenAI compatibility](https://inference-docs.cerebras.ai/resources/openai)."
        ),
    },
    "OLLAMA_API_KEY": {
        "description": (
            "Ollama API key for direct OpenAI-compatible Cloud access at ollama.com/v1."
        ),
    },
}


def provider_field_specs() -> tuple[dict[str, Any], ...]:
    """Return provider fields generated from the provider catalog."""

    return (
        *_credential_field_specs(),
        *_cloudflare_account_field_specs(),
        *_vertex_field_specs(),
        *_base_url_field_specs(),
        *_proxy_field_specs(),
    )


def _credential_field_specs() -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    seen_env_keys: set[str] = set()
    for descriptor in PROVIDER_CATALOG.values():
        if descriptor.credential_env is None:
            continue
        if descriptor.credential_env in seen_env_keys:
            continue
        seen_env_keys.add(descriptor.credential_env)
        spec = {
            "key": descriptor.credential_env,
            "label": f"{descriptor.display_name} API Key",
            "section_id": "providers",
            "field_type": "secret",
            "settings_attr": descriptor.credential_attr,
            "secret": True,
        }
        spec.update(_PROVIDER_FIELD_OVERRIDES.get(descriptor.credential_env, {}))
        specs.append(spec)
    return tuple(specs)


def _base_url_field_specs() -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for descriptor in PROVIDER_CATALOG.values():
        if descriptor.base_url_attr is None:
            continue
        key = _settings_env_key(descriptor.base_url_attr)
        spec = {
            "key": key,
            "label": f"{descriptor.display_name} Base URL",
            "section_id": "providers",
            "settings_attr": descriptor.base_url_attr,
            "default": descriptor.default_base_url or "",
        }
        spec.update(_PROVIDER_FIELD_OVERRIDES.get(key, {}))
        specs.append(spec)
    return tuple(specs)


def _cloudflare_account_field_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "key": "CLOUDFLARE_ACCOUNT_ID",
            "label": "Cloudflare Account ID",
            "section_id": "providers",
            "settings_attr": "cloudflare_account_id",
            "description": (
                "Cloudflare account ID used to build the /accounts/{id}/ai/v1 endpoint."
            ),
        },
    )


def _vertex_field_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "key": "VERTEX_PROJECT_ID",
            "label": "Google Cloud Project ID",
            "section_id": "providers",
            "settings_attr": "vertex_project_id",
            "description": (
                "Google Cloud project used for Vertex AI. Authentication uses "
                "Application Default Credentials (ADC)."
            ),
        },
        {
            "key": "VERTEX_LOCATION",
            "label": "Vertex AI Location",
            "section_id": "providers",
            "settings_attr": "vertex_location",
            "default": "global",
            "description": (
                "Use global for the global Vertex AI endpoint or a region such as "
                "us-central1."
            ),
        },
    )


def _proxy_field_specs() -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for descriptor in PROVIDER_CATALOG.values():
        if descriptor.proxy_attr is None:
            continue
        specs.append(
            {
                "key": _settings_env_key(descriptor.proxy_attr),
                "label": f"{descriptor.display_name} Proxy",
                "section_id": "providers",
                "field_type": "secret",
                "settings_attr": descriptor.proxy_attr,
                "secret": True,
                "advanced": True,
            }
        )
    return tuple(specs)


def _settings_env_key(settings_attr: str) -> str:
    model_field = Settings.model_fields[settings_attr]
    alias = model_field.validation_alias
    return str(alias) if alias is not None else settings_attr
