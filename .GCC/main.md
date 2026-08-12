# Current Project Context

## 🏆 Major Milestones (Archived Epics)
- 2026-08-12: Réconciliation et fusion des 5 commits amont (Together AI, QwenCloud, xAI Grok, Novita AI, NaraRoute) sur la branche `upstream-sync` et `main`. Bump version `v4.23.0`, `uv.lock` ré-généré et validation 100% CI (2968 tests passés avec `./scripts/ci.sh`).
- 2026-08-12: Correction des URL d'archives d'installation dans [`scripts/install.sh`](file:///home/omni/free-claude-code/scripts/install.sh) et [`scripts/install.ps1`](file:///home/omni/free-claude-code/scripts/install.ps1) pointant directement sur le fork `omni01-Cell/free-claude-code`, garantissant l'installation de 100% des providers et fonctionnalités du fork. Bump version `v4.20.1`, mise à jour de `uv.lock` et validation 100% CI (2908 tests passés avec `./scripts/ci.sh`).
- 2026-08-12: Enrichissement de la documentation Windows [`docs/WINDOWS_GUIDE.md`](file:///home/omni/free-claude-code/docs/WINDOWS_GUIDE.md) avec la totalité des fonctionnalités et providers personnalisés du fork `omni01-Cell/free-claude-code`.
- 2026-08-12: Intégration de Google Antigravity CLI en tant que Connected Account dans l'Admin UI (`v4.20.0`). Implémentation de `AntigravityAuthManager` (`ConnectedAccountPort`), déclaration `ProviderAuthKind.CONNECTED_ACCOUNT` dans `provider_catalog.py`, câblage dans `bootstrap.py` et validation 100% CI.
- 2026-08-12: Lanceur hybride Codex Desktop / ChatGPT GUI (`fcc-codex-desktop`) v4.19.3. Résolution des exécutables officiels OpenAI Linux (`/usr/bin/chatgpt`, `/usr/lib/chatgpt/codex-launcher`, `/usr/lib/chatgpt/ChatGPT`), support des drapeaux `--setup` et `--reset`/`--restore`.
- 2026-08-12: Assemblage d'outils à état (`active_tool_by_name`) et déduplication des appels dans `AntigravityProvider.stream_response()`.

## 🎯 Objective
Maintenir le serveur proxy local free-claude-code à un niveau de qualité zéro-défaut pour Claude Code CLI et Codex, assurer la compatibilité multi-provider (incluant Google Antigravity CLI, AgentRouter, CommandCode, TokenRouter, Alibaba, OpenAI Compatible, Anthropic Compatible ainsi que les nouveaux Together AI, QwenCloud, xAI Grok, Novita AI, NaraRoute) et la conformité stricte aux garde-fous CI `./scripts/ci.sh`.

## 🧠 Decisions Made
- 2026-08-12: **Stratégie de synchronisation upstream** : Les commits amont de `Alishahryar1/free-claude-code` ont été intégrés d'abord sur la branche dédiée `upstream-sync`, les conflits sur `provider_catalog.py`, `settings.py`, `pyproject.toml`, et la suite de tests ont été résolus en conservant tous les providers locaux et amont, puis fusionnés vers `main` (`v4.23.0`).

## 🌿 Active Branches / Plans
- `main` : Branche principale à jour avec `upstream/main` (`v4.23.0`), 100% qualifiée avec 2968 tests passés.
- `upstream-sync` : Branche de synchronisation miroir pushed sur `origin/upstream-sync`.

## 📈 Current Status
- ✅ Done: Synchronisation upstream `v4.23.0` complète, tests (2968 passed), linter (ruff) et type checker (ty) 100% verts, branch `main` et `upstream-sync` poussées sur `origin`.
- 🔄 In progress: Aucun.
- ⏳ Pending: Attente de nouvelles directives utilisateur.

## 👉 Next Session Direction
Le projet est parfaitement synchronisé avec amont et qualifié CI à 100%. Prêt pour l'implémentation de nouvelles fonctionnalités ou la maintenance courante.
