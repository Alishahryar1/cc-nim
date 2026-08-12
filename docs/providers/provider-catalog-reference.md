<!-- Classification Diátaxis :
1. Le lecteur cherche une information factuelle précise (clé d'API, URL par défaut, mode d'authentification).
2. Le lecteur connaît déjà le système et souhaite consulter la spécification technique du catalogue.
Type = Reference Guide. -->

# Référence du Catalogue des Providers (Provider Catalog Reference)

Ce document détaille la structure de données `ProviderDescriptor` et répertorie l'ensemble des fournisseurs gérés par la table `PROVIDER_CATALOG` (`src/free_claude_code/config/provider_catalog.py`).

---

## Structure d'un `ProviderDescriptor`

Chaque fournisseur du catalogue est décrit par une instance immuable de `ProviderDescriptor` :

| Champ | Type | Description |
| :--- | :--- | :--- |
| `provider_id` | `str` | Identifiant unique interne (ex. `"open_router"`, `"antigravity"`). |
| `display_name` | `str` | Nom d'affichage lisible pour l'utilisateur. |
| `auth_kind` | `ProviderAuthKind` | Mode d'authentification (`configuration` via clé/env ou `connected_account`). |
| `local` | `bool` | `True` si le fournisseur s'exécute en local (ex. Ollama, LM Studio). |
| `credential_env` | `str \| None` | Nom de la variable d'environnement contenant la clé d'API. |
| `credential_url` | `str \| None` | URL de la console fournisseur pour obtenir la clé d'API. |
| `credential_attr` | `str \| None` | Nom de l'attribut dans les paramètres de configuration. |
| `default_base_url` | `str \| None` | URL de base d'API HTTP/REST par défaut. |
| `proxy_attr` | `str \| None` | Attribut de configuration pour le proxy réseau HTTP. |

---

## Fournisseurs Répertoriés dans `PROVIDER_CATALOG`

Le catalogue recense plus de 20 fournisseurs classés par catégories d'accès :

### 1. Fournisseurs Cloud Commerciaux Majeurs

| Provider ID | Nom d'affichage | Variable d'environnement | Base URL par défaut |
| :--- | :--- | :--- | :--- |
| `nvidia_nim` | NVIDIA NIM | `NVIDIA_NIM_API_KEY` | `https://integrate.api.nvidia.com/v1` |
| `open_router` | OpenRouter | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| `groq` | Groq | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` |
| `xai` | xAI (Grok) | `XAI_API_KEY` | `https://api.x.ai/v1` |
| `gemini` | Gemini (AI Studio) | `GEMINI_API_KEY` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `vertex` | Google Vertex AI | ADC / GCP Credentials | `https://aiplatform.googleapis.com` |
| `deepseek` | DeepSeek | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` |
| `mistral` | Mistral | `MISTRAL_API_KEY` | `https://api.mistral.ai/v1` |
| `mistral_codestral` | Mistral Codestral | `CODESTRAL_API_KEY` | `https://codestral.mistral.ai/v1` |
| `bedrock` | Amazon Bedrock | `AWS_BEARER_TOKEN_BEDROCK` | `https://bedrock-mantle.us-east-1.api.aws/v1` |
| `github_models` | GitHub Models | `GITHUB_MODELS_TOKEN` | `https://models.github.ai/inference` |
| `qwencloud` | QwenCloud | `QWENCLOUD_API_KEY` | `https://token-plan.ap-southeast-1.maas.aliyuncs.com/...` |
| `together` | Together AI | `TOGETHER_API_KEY` | `https://api.together.ai/v1` |

### 2. Comptes Connectés (OAuth / Web)

| Provider ID | Nom d'affichage | Auth Kind | Base URL par défaut |
| :--- | :--- | :--- | :--- |
| `antigravity` | Google Antigravity CLI | `connected_account` | `https://cloudcode-pa.googleapis.com` |
| `openai` | OpenAI / ChatGPT Codex | `connected_account` | `https://chatgpt.com/backend-api/codex` |

### 3. Adaptateurs Generiques Compatible

| Provider ID | Nom d'affichage | Variable d'environnement | Base URL par défaut |
| :--- | :--- | :--- | :--- |
| `openai_compatible` | OpenAI Compatible | `OPENAI_COMPATIBLE_API_KEY` | `https://api.openai.com/v1` |
| `anthropic_compatible` | Anthropic Compatible | `ANTHROPIC_COMPATIBLE_API_KEY` | `https://api.anthropic.com/v1` |

### 4. Proveurs Locaux (`local=True`)

| Provider ID | Nom d'affichage | Crédentiel Statique | Base URL par défaut |
| :--- | :--- | :--- | :--- |
| `lmstudio` | LM Studio | `lm-studio` | `http://localhost:1234/v1` |
| `llamacpp` | llama.cpp | `llamacpp` | `http://localhost:8080/v1` |
| `ollama` | Ollama | `ollama` | `http://localhost:11434` |

---

## Ordre des Identifiants (`SUPPORTED_PROVIDER_IDS`)

L'ordre du dictionnaire `PROVIDER_CATALOG` détermine l'ordre d'affichage dans la CLI et les messages d'erreur. Les requêtes invalides levant un fournisseur inconnu renvoient la liste exhaustive définie par `SUPPORTED_PROVIDER_IDS`.
