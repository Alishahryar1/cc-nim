<!--
Classification Diátaxis :
1. Lecteur consultant des données techniques factuelles et exhaustives (Thinking).
2. Lecteur connaissant l'existence des commandes et cherchant une référence d'API/CLI (Already knows).
Type = Reference.
-->

# Référence des commandes CLI, variables d'environnement et paramètres

Ce document fournit la référence technique complète des exécutables CLI, des options de ligne de commande, des variables d'environnement et du schéma de configuration Pydantic `Settings`.

## Exécutables CLI et options

| Commande | Option / Drapeau | Description |
| :--- | :--- | :--- |
| `fcc-server` | `--version` | Affiche la version courante de `free-claude-code` et quitte. |
| `fcc-claude` | `[arguments...]` | Transmet les arguments au binaire `claude` avec l'environnement proxy `ANTHROPIC_BASE_URL`. |
| `fcc-codex` | `--print-proxy-auth-token` | Imprime le jeton Bearer d'authentification du proxy local et quitte. |
| `fcc-codex` | `[arguments...]` | Lance `codex` en injectant les options `-c model_provider="fcc"` et le catalogue. |
| `fcc-codex-desktop` | `--setup` | Inscrit le provider `fcc` de façon permanente dans `~/.codex/config.toml`. |
| `fcc-codex-desktop` | `--reset` / `--restore` | Restaure la configuration `config.toml` initiale depuis le fichier `.fccbak`. |
| `fcc-pi` | `[passthrough]` | Redirige directement `config`, `install`, `list`, `remove`, `uninstall`, `update`, `--help`. |
| `fcc-pi` | `[session]` | Lance `pi` avec l'extension `-e pi_extension.ts` et le scope `--models free-claude-code/**`. |
| `fcc-desktop` | *(aucun)* | Démarre la barre d'état système tray (Pystray) et supervise le serveur Uvicorn. |

## Variables d'environnement de la couche CLI

| Variable | Type | Description |
| :--- | :--- | :--- |
| `ANTHROPIC_AUTH_TOKEN` | Chaîne | Jeton Bearer de sécurité protégeant l'accès au proxy HTTP local. |
| `CODEX_DESKTOP_PATH` | Chemin | Chemin absolu surchargeant la localisation du binaire `codex-desktop` ou `ChatGPT`. |
| `CODEX_HOME` | Chemin | Répertoire de configuration de Codex CLI (par défaut `~/.codex`). |
| `FCC_PI_BASE_URL` | URL | URL de base du proxy transmise automatiquement à l'extension Pi. |
| `FCC_PI_API_KEY` | Chaîne | Clé d'API éphémère transmise par le lanceur Pi à l'extension. |

## Schéma de configuration Pydantic (`Settings`)

Le tableau suivant détaille les champs clés du schéma `Settings` (`src/free_claude_code/config/settings.py`) chargés depuis les fichiers dotenv et l'environnement :

| Champ Pydantic | Alias d'environnement | Type / Valeur par défaut | Description |
| :--- | :--- | :--- | :--- |
| `host` | `HOST` | `str = "0.0.0.0"` | Adresse d'écoute de l'application FastAPI Uvicorn. |
| `port` | `PORT` | `int = 8082` | Port d'écoute du serveur proxy local. |
| `open_admin_browser` | `FCC_OPEN_BROWSER` | `bool = True` | Ouvre automatiquement le navigateur sur l'UI Admin au démarrage. |
| `anthropic_auth_token` | `ANTHROPIC_AUTH_TOKEN` | `str = ""` | Jeton d'authentification des requêtes clients. |
| `model` | `MODEL` | `str = "nvidia_nim/..."` | Modèle par défaut au format `provider/model_name`. |
| `log_level` | `LOG_LEVEL` | `str = "INFO"` | Niveau de journalisation (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |
| `messaging_platform` | `MESSAGING_PLATFORM` | `str = "discord"` | Plateforme de messagerie (`telegram`, `discord`, `none`). |
| `enable_web_server_tools` | `ENABLE_WEB_SERVER_TOOLS` | `bool = False` | Active les outils serveur `web_search` et `web_fetch`. |

## Exemples d'utilisation

### Démarrer le serveur sur un port spécifique via l'environnement

```bash
PORT=9090 LOG_LEVEL=DEBUG fcc-server
```

Sortie attendue :

```text
INFO: Uvicorn running on http://0.0.0.0:9090
```

### Extraire le jeton d'authentification du proxy Codex

```bash
fcc-codex --print-proxy-auth-token
```

Sortie attendue :

```text
sk-fcc-proxy-auth-token
```
