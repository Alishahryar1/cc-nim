<!--
Classification Diátaxis :
1. Lecteur en phase d'assimilation théorique de l'architecture du domaine CLI (Thinking).
2. Lecteur découvrant le système pour la première fois (Learning first time).
Type = Explanation.
-->

# Architecture du domaine CLI et des lanceurs

Le domaine CLI de Free Claude Code regroupe les points d'entrée exécutables et la logique de supervision permettant d'interconnecter divers agents de codage (Claude Code, Codex CLI, Codex Desktop, Pi) à un serveur proxy local unifié.

## Problématique et rôle de la couche CLI

Les agents de codage modernes s'attendent à des variables d'environnement, des formats de configuration et des API endpoints spécifiques. Le domaine CLI résout l'hétérogénéité des clients en injectant dynamiquement des configurations d'environnement éphémères (`base_url`, jetons d'authentification) avant de déléguer l'exécution aux binaires natifs.

```
┌─────────────────────────────────────────────────────────────┐
│                      Commandes CLI                          │
│   fcc-server  │  fcc-claude  │  fcc-codex  │  fcc-pi  ...  │
└──────┬──────────────┴──────────────┬─────────────┴──────────┘
       │                             │
       ▼                             ▼
┌───────────────┐           ┌──────────────────┐
│ Server        │           │ Lanceurs Client  │
│ Supervisor    │           │ (env éphémère /  │
│ (FastAPI/     │           │  bypass proxy)   │
│  Uvicorn)     │           └────────┬─────────┘
└──────┬────────┘                    │
       │                             │
       ▼                             ▼
 ┌─────────────────────────────────────────┐
 │ Proxy HTTP Local (http://127.0.0.1:8082)│
 └─────────────────────────────────────────┘
```

## Cartographie de l'arborescence (`src/free_claude_code/cli/`)

Le dossier `src/free_claude_code/cli/` est structuré en trois niveaux de responsabilité :

| Fichier / Dossier | Responsabilité principale |
| :--- | :--- |
| `entrypoints.py` | Points d'entrée légers enregistrés dans `pyproject.toml` (`serve`). |
| `commands.py` | Supervision du serveur FastAPI/Uvicorn (`ServerSupervisor`) et gestion du cycle de vie. |
| `desktop_entrypoint.py` | Point d'entrée GUI pour le mode barquette système (`fcc-desktop`). |
| `desktop.py` & `desktop_tray.py` | Contrôleur d'arrière-plan et intégration du système tray (Pystray). |
| `launchers/` | Lanceurs dédiés aux agents clients (`claude.py`, `codex.py`, `codex_desktop.py`, `pi.py`, `common.py`). |
| `managed/` | Gestionnaires de sessions managées pour l'exécution d'agents en arrière-plan (`manager.py`, `session.py`, `diagnostics.py`). |
| `process_registry.py` | Registre global des processus enfants pour un nettoyage propre à l'arrêt. |
| `proxy_auth.py` & `local_http.py` | Extraction des jetons de proxy et contournement du proxy local pour les appels loopback. |

## Exécutables déclarés dans `pyproject.toml`

La configuration du paquet définit six binaires CLI et GUI :

1. **`fcc-server`** : Démarre et supervise le serveur proxy HTTP local Uvicorn.
2. **`fcc-claude`** : Injecte les variables `ANTHROPIC_BASE_URL` et lance le binaire `claude`.
3. **`fcc-codex`** : Injecte les arguments `-c` TOML éphémères et lance `codex`.
4. **`fcc-codex-desktop`** : Injecte la configuration éphémère dans `~/.codex/config.toml` et lance l'application Codex Desktop.
5. **`fcc-pi`** : Charge l'extension TypeScript intégrée `pi_extension.ts` et démarre `pi`.
6. **`fcc-desktop`** : Lance le serveur proxy en tâche de fond avec une icône dans la zone de notification.

## Analyse comparative des mécanismes de lancement

| Lanceur | Ingestion de config | Nettoyage post-exécution |
| :--- | :--- | :--- |
| `fcc-claude` | Variables d'environnement de processus | Automatique à la fermeture du sous-processus |
| `fcc-codex` | Arguments CLI `-c` dynamiques | Aucun état résiduel sur le disque |
| `fcc-codex-desktop` | Injection temporaire dans `config.toml` | Restauration garantie via context manager `ephemeral_codex_config` |
| `fcc-pi` | Argument `-e` pour extension TS | Restauration de l'environnement de processus |
