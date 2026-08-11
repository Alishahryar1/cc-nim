# Execution Plan: Upstream Migration to `src/free_claude_code/` Package Layout

## 📋 Target Invariant & Pre-requisites
- **Target Invariant**: Intégrer les 160 commits d'upstream/main (layout `src/free_claude_code/`, TokenRouter, Azure OpenAI, ChatGPT OAuth, Groq reasoning fixes) sans régresser sur nos providers custom (Google Antigravity CLI v2.9.2, AgentRouter, CommandCode, Alibaba, OpenAI/Anthropic Compatible) et maintenir 100% de réussite CI (`./scripts/ci.sh`).
- **Pre-requisites**: Branche de sauvegarde `backup/v2.9.2-pre-sync` créée.

## 🛠️ Step-by-Step Sequence

### Step 1: Mapping et Analyse de la Structure Upstream `src/free_claude_code/`
- [ ] **Action**: Inspecter les répertoires et fichiers dans `upstream/main` pour établir la carte exacte de migration (`api/` -> `src/free_claude_code/...`, etc.).
- [ ] **Verify**: Rapport de mapping valide.

### Step 2: Checkout et Fusion Structurée des Branches
- [ ] **Action**: Effectuer la fusion de `upstream/main` sur `master` et déplacer/résoudre nos fichiers custom vers la nouvelle arborescence `src/free_claude_code/`.
- [ ] **Verify**: `git status` sans conflit non résolu.

### Step 3: Migration des Providers Custom vers `src/free_claude_code/providers/`
- [ ] **Action**: Placer nos providers (`antigravity`, `agent_router`, `command_code`, `token_router`, `alibaba`, `openai_compatible`, `anthropic_compatible`) dans `src/free_claude_code/providers/`.
- [ ] **Verify**: `uv run ty check` sans erreur d'importation.

### Step 4: Migration du Routing, Admin UI & Manifests
- [ ] **Action**: Porter les enregistrements Antigravity/AgentRouter/CommandCode dans `src/free_claude_code/application/routing.py`, `src/free_claude_code/config/admin/`, et `src/free_claude_code/config/provider_catalog.py`.
- [ ] **Verify**: `uv run pytest tests/api/ -v`

### Step 5: Migration des Tests Unitaires et Validation CI finale
- [ ] **Action**: Mettre à jour et exécuter l'ensemble des tests dans `tests/`.
- [ ] **Verify**: `./scripts/ci.sh` (lint, format, ty, pytest).

## ⚠️ Mitigations & Edge Cases
- **Risque**: Conflits d'importation entre `config.provider_catalog` et `free_claude_code.config.provider_catalog`.
- **Mitigation**: Mise à jour systématique de tous les imports vers le package `free_claude_code` et validation stricte avec `uv run ty check`.
