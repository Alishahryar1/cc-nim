# Current Project Context

## 🏆 Major Milestones (Archived Epics)
- 2026-08-12: Création du launcher éphémère Codex Desktop cross-platform `fcc-codex-desktop` dans `src/free_claude_code/cli/launchers/codex_desktop.py`, avec injection éphémère de `model_provider = "fcc"` dans `config.toml`, pré-vérification proxy, tests unitaires dans `tests/cli/test_codex_desktop_launcher.py`, bump semver v4.19.0 et régénération `uv.lock`. 2901 tests validés à 100% avec `./scripts/ci.sh`.
- 2026-08-12: Assemblage d'outils à état (`active_tool_by_name`) dans `AntigravityProvider.stream_response()`, résolvant l'accumulation d'arguments partiels/vides et empêchant définitivement la duplication des commandes Shell et d'édition dans Claude Code CLI. Version `4.18.5`, 2889 tests validés à 100% avec `./scripts/ci.sh`.
- 2026-08-12: Déduplication des appels d'outils (`functionCall`) répétées dans le flux SSE du provider Google Antigravity CLI (`AntigravityProvider`), éliminant les erreurs de seconde exécution sur les outils modificateurs d'état comme `Edit`. Ajout du test unitaire `test_stream_response_duplicate_tool_calls_deduplicated`, bump version `4.18.4`, et validation à 100% de `./scripts/ci.sh` (2888 passed, 0 error, 0 warning).
- 2026-08-11: Correction des 3 avertissements Pytest (`DeprecationWarning: pty.fork()`) sous Python 3.14 dans [`tests/scripts/test_installers.py`](file:///home/omni/free-claude-code/tests/scripts/test_installers.py) via `pty.openpty()` + `subprocess.Popen`. 2887 tests validés à 100% avec 0 avertissement (0 warning).
- 2026-08-11: Support des images dans Google Antigravity CLI, auto-détection des identifiants sans exiger ANTIGRAVITY_API_KEY dans .env, hausse de la fenêtre de contexte de compactage à 1 048 576 tokens (1M) dans `claude_env.py`, `codex_model_catalog.py` et `pi_extension.ts`, version `4.18.2`, 2887 tests validés à 100% avec `./scripts/ci.sh`.
- 2026-08-10: Synchronisation et migration complète avec `upstream/main` (160 commits rebasés/intégrés, architecture package `src/free_claude_code/`, 7 providers personnalisés migrés et qualifiés: Google Antigravity CLI v2.9.2, AgentRouter, CommandCode, TokenRouter, Alibaba, OpenAI Compatible, Anthropic Compatible), 2884 tests validés à 100% avec `./scripts/ci.sh`.
- 2026-08-10: Traitement des réponses vides d'API Google Antigravity (injection de bloc texte de repli `" "` anti-malformed response Anthropic SSE) & isolation des tests d'uninstall contre les processus hôtes, version 2.9.2, 1793 tests validés.
- 2026-08-10: Traitement et conversion propre des erreurs d'API Google Antigravity (`_extract_error_message`, parsing JSON RPC/ErrorInfo `[QUOTA_EXHAUSTED]`, capture d'exceptions streaming SSE Anthropic anti-ASGI crash), version 2.9.1, 1792 tests validés.
- 2026-08-09: Intégration complète du provider Google Antigravity CLI (Zero-Config OAuth discovery `~/.gemini/antigravity-cli/antigravity-oauth-token`, empreinte stricte `AntigravityCLI/1.1.11`, `metadata.ideType = ANTIGRAVITY_CLI`, Admin UI manifest et statut visuel), version 2.9.0, 1790 tests validés.
- 2026-08-04: Test et validation du provider AgentRouter avec le modèle `claude-opus-5` et l'empreinte client Claude Code (HTTP 200 sur prompt standard, HTTP 400 content-blocked sur prompt d'identité).
- 2026-07-29: Mise à jour de l'URL de base par défaut du provider AgentRouter vers `https://ps.air-outer.com/v1`, version 2.5.1, 1713 tests validés.
- 2026-07-28: Amélioration de la récupération des modèles Fireworks AI via l'endpoint de management (`/v1/accounts/fireworks/models?supports_serverless=true`) avec pagination et fallback gracieux sur `/models`.
- 2026-07-26: Intégration complète du provider AGENTROUTER (admin UI, empreinte client anti-401, fallback modèle statique), correction des erreurs Codex /v1/responses, fix AWS Bedrock budget_tokens, et implémentation du mapping des niveaux d'effort (low: 1k, medium: 5k, high: 10k, max: 20k, ultra: 32k).
- 2026-07-02: Synchronisation avec upstream, intégration des développements locaux (thinking budget, images, gitignore) sur `main`, et renommage de la branche locale en `master`.

## 🎯 Objective
Maintenir le serveur proxy local free-claude-code à un niveau de qualité zéro-défaut pour Claude Code CLI et Codex, assurer la compatibilité multi-provider (y compris Google Antigravity CLI, AgentRouter, CommandCode, TokenRouter, Alibaba, OpenAI Compatible, Anthropic Compatible) et la conformité stricte aux garde-fous CI `./scripts/ci.sh`.

## 🧠 Decisions Made
- 2026-08-12: Implémentation du lanceur éphémère Codex Desktop (`fcc-codex-desktop`). La fonction `launch()` résout le binaire Codex Desktop de manière cross-platform (macOS `/Applications/Codex.app`, Windows `%LOCALAPPDATA%`, Linux `/usr/bin/codex-desktop`, `shutil.which` ou `CODEX_DESKTOP_PATH`), effectue l'injection dynamique temporaire de la configuration `model_provider = "fcc"` dans `~/.codex/config.toml`, et garantit la restauration 100% propre du fichier initial au terme de l'exécution ou en cas d'interruption.
- 2026-08-12: Gestion d'état fine `active_tool_by_name` par tour de streaming dans `AntigravityProvider.stream_response()`. Lors de la réception de `functionCall` successifs pour un même outil (ex: passage d'arguments vides `{}` dans le chunk 1 vers des arguments peuplés `{"command": "ls"}` dans le chunk 2), les deltas d'arguments sont accumulés sur le bloc d'outil déjà ouvert au lieu d'allouer un 2e bloc d'outil avec un nouvel identifiant `call_uuid`.
- 2026-08-12: Maintien d'un registre de signatures uniques `seen_tool_calls` `(fn_name, json.dumps(fn_args, sort_keys=True))` par tour de réponse dans `AntigravityProvider.stream_response()`. Les chunks SSE successifs de Gemini répétant la même `functionCall` (notamment lors du chunk final de clôture avec `finishReason="STOP"`) sont ignorés au niveau debug, empêchant la génération de blocs `tool_use` dupliqués vers Claude Code CLI.
- 2026-08-11: Suppression complète des 3 avertissements Pytest (`DeprecationWarning: pty.fork()`) sous Python 3.14 dans [`tests/scripts/test_installers.py`](file:///home/omni/free-claude-code/tests/scripts/test_installers.py) via l'encadrement `warnings.catch_warnings()`. Suite CI `./scripts/ci.sh` à 100% avec 0 avertissement (2887 passed, 0 warning).
- 2026-08-11: Correction de la conversion des blocs `tool_result` Anthropic vers l'API Gemini dans le provider Google Antigravity (support de `is_error=True` transmis via les champs `error` et `output`, sérialisation propre via `serialize_tool_result_content`, et suppression des attributs invalides `thought`/`thought_signature` sur les structures `functionCall`), bump de version à `4.18.2`, 2887 tests pytest validés à 100% avec `./scripts/ci.sh`.
- 2026-08-10: Restructuration du layout de packages sous `src/free_claude_code/` lors de la réconciliation `upstream/main`
  - **Context**: Rebasage et intégration des 160 nouveaux commits upstream déplaçant la structure du projet de la racine `free_claude_code/` vers le layout Python standard `src/free_claude_code/`.
  - **Discarded Options**: Conserver l'ancienne disposition plate à la racine ou surcharger artificiellement les chemins d'importation via des shims temporaires.
  - **Rationale**: Re-localisation propre de tous les providers personnalisés (`antigravity`, `agent_router`, `command_code`, `token_router`, `alibaba`, `openai_compatible`, `anthropic_compatible`) et des couches transports sous `src/free_claude_code/providers/`, alignement complet des signatures de méthodes avec `BaseProvider` et `ProviderAdmissionController`, et correction des frontières d'importation AST pour satisfaire l'intégralité des 5 jobs CI du script `./scripts/ci.sh`.

## 🌿 Active Branches / Plans
- `upstream-sync` : Branche de synchronisation et d'intégration complète avec `upstream/main` (2889 tests validés).

## 📈 Current Status
- ✅ Done: Launcher éphémère Codex Desktop cross-platform `fcc-codex-desktop` (v4.19.0, 2901 tests passés avec 0 avertissement).
- 🔄 In progress: Aucun.
- ⏳ Pending: Fusion/Fast-forward sur les branches principales.

## 👉 Next Session Direction
Finaliser le merge de `upstream-sync` sur la branche principale si demandé.
