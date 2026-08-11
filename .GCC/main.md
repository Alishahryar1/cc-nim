# Current Project Context

## 🏆 Major Milestones (Archived Epics)
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
- 2026-08-10: Restructuration du layout de packages sous `src/free_claude_code/` lors de la réconciliation `upstream/main`
  - **Context**: Rebasage et intégration des 160 nouveaux commits upstream déplaçant la structure du projet de la racine `free_claude_code/` vers le layout Python standard `src/free_claude_code/`.
  - **Discarded Options**: Conserver l'ancienne disposition plate à la racine ou surcharger artificiellement les chemins d'importation via des shims temporaires.
  - **Rationale**: Re-localisation propre de tous les providers personnalisés (`antigravity`, `agent_router`, `command_code`, `token_router`, `alibaba`, `openai_compatible`, `anthropic_compatible`) et des couches transports sous `src/free_claude_code/providers/`, alignement complet des signatures de méthodes avec `BaseProvider` et `ProviderAdmissionController`, et correction des frontières d'importation AST pour satisfaire l'intégralité des 5 jobs CI du script `./scripts/ci.sh`.

## 🌿 Active Branches / Plans
- `upstream-sync` : Branche de synchronisation et d'intégration complète avec `upstream/main` (2884 tests validés).

## 📈 Current Status
- ✅ Done: Synchronisation avec `upstream/main` achevée, 7 providers personnalisés migrés sous `src/free_claude_code/`, suppression de `from __future__ import annotations`, et validation à 100% du script `./scripts/ci.sh` (2884 tests passés).
- 🔄 In progress: Aucun.
- ⏳ Pending: Fusion/Fast-forward de `upstream-sync` sur les branches principales.

## 👉 Next Session Direction
Finaliser le merge de `upstream-sync` sur la branche principale si demandé.
