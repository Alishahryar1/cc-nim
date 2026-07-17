"""Catalog-derived Admin provider fields."""

from typing import Any

from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings

_PROVIDER_FIELD_OVERRIDES: dict[str, dict[str, Any]] = {
    "NVIDIA_NIM_API_KEY": {
        "label": "NVIDIA NIM API Key",
        "description": (
            "Used by NVIDIA NIM chat and optional NIM voice transcription. "
            "Accepts a comma-separated list of keys for multi-key rotation."
        ),
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
    "KIMI_CODING_API_KEY": {
        "label": "Kimi Coding API Key",
        "description": (
            "Kimi For Coding subscription API key. "
            "Used for the OpenAI-compatible Chat Completions API at api.kimi.com/coding."
        ),
    },
    "CHATGPT_OAUTH_ACCESS_TOKEN": {
        "label": "ChatGPT OAuth Access Token",
        "description": (
            "Experimental/unsanctioned: OAuth access token for ChatGPT/Codex. "
            "Leave empty to auto-load from ~/.codex/auth.json after running "
            "'fcc-chatgpt-oauth-login' or 'codex login'. "
            "Use at your own risk; this flow is not an official OpenAI API product."
        ),
    },
    "CHATGPT_OAUTH_BASE_URL": {
        "label": "ChatGPT OAuth Base URL",
        "description": (
            "Experimental/unsanctioned: ChatGPT/Codex backend root. Defaults to "
            "https://chatgpt.com/backend-api. Only change this if you need to "
            "route through a proxy or mirror."
        ),
    },
    "CHATGPT_OAUTH_PROXY": {
        "label": "ChatGPT OAuth Proxy",
        "description": (
            "Experimental/unsanctioned: optional HTTP proxy for ChatGPT/Codex requests."
        ),
    },
    "MINIMAX_API_KEY": {
        "label": "MiniMax API Key",
        "description": (
            "MiniMax API key for the OpenAI-compatible Chat Completions API at "
            "free_claude_code.api.minimax.io/v1."
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
        *_rotation_field_specs(),
        *_chatgpt_oauth_login_field_specs(),
        *_chatgpt_oauth_account_field_specs(),
        *_cloudflare_account_field_specs(),
        *_local_base_url_field_specs(),
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


def _rotation_field_specs() -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    seen_env_keys: set[str] = set()
    for descriptor in PROVIDER_CATALOG.values():
        if descriptor.credential_env is None:
            continue
        if descriptor.credential_env in seen_env_keys:
            continue
        seen_env_keys.add(descriptor.credential_env)
        specs.append(
            {
                "key": f"{descriptor.credential_env}_ROTATION",
                "label": f"{descriptor.display_name} Key Rotation",
                "section_id": "providers",
                "field_type": "select",
                "default": "single",
                "options": ("single", "round_robin", "on_error"),
                "advanced": True,
                "restart_required": True,
                "description": (
                    "Rotation policy across a comma-separated list of keys in "
                    f"{descriptor.credential_env}. single = use the first key only; "
                    "round_robin = spread requests across keys; "
                    "on_error = stick to one key until it fails, then fail over. "
                    "Requires restart."
                ),
            }
        )
    return tuple(specs)


def _local_base_url_field_specs() -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for descriptor in PROVIDER_CATALOG.values():
        if descriptor.base_url_attr is None:
            continue
        specs.append(
            {
                "key": _settings_env_key(descriptor.base_url_attr),
                "label": f"{descriptor.display_name} Base URL",
                "section_id": "providers",
                "settings_attr": descriptor.base_url_attr,
                "default": descriptor.default_base_url or "",
            }
        )
    return tuple(specs)


def _chatgpt_oauth_account_field_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "key": "CHATGPT_OAUTH_ACCOUNT_ID",
            "label": "ChatGPT OAuth Account ID",
            "section_id": "providers",
            "settings_attr": "chatgpt_oauth_account_id",
            "description": (
                "Experimental/unsanctioned: ChatGPT account ID used for the "
                "ChatGPT-Account-ID header. Optional when using a bridge; "
                "auto-extracted from the access token JWT when empty."
            ),
        },
    )


def _chatgpt_oauth_login_field_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "key": "CHATGPT_OAUTH_LOGIN",
            "label": "ChatGPT OAuth Login",
            "section_id": "providers",
            "field_type": "oauth_login",
            "description": (
                "Experimental/unsanctioned: click the button to log in to ChatGPT/Codex "
                "via OAuth. Tokens are saved to ~/.codex/auth.json. "
                "Use at your own risk; this is not an official OpenAI API product."
            ),
        },
        {
            "key": "CHATGPT_OAUTH_IMPORT_CODEX",
            "label": "Import Codex CLI Tokens",
            "section_id": "providers",
            "field_type": "oauth_login",
            "description": (
                "Experimental/unsanctioned: if you have already run 'codex login', "
                "click the button to import the existing tokens from ~/.codex/auth.json."
            ),
        },
    )


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
