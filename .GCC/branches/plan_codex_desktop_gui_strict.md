# Execution Plan: Codex Desktop GUI Strict Launcher

## 📋 Target Invariant & Pre-requisites
- **Target Invariant**: `fcc-codex-desktop` doit résoudre uniquement les exécutables GUI officiels de Codex Desktop (`codex-desktop`, `Codex Desktop`, `Codex.app`, `Codex.exe`) et NE DOIT EN AUCUN CAS basculer sur le CLI terminal `codex` / `codex.exe`. En l'absence du binaire GUI, le lanceur doit lever une erreur 127 explicite orientant vers le téléchargement de l'application GUI.
- **Pre-requisites**: `uv run pytest`, `./scripts/ci.sh`.

## 🛠️ Step-by-Step Sequence

### Step 1: Nettoyage de la recherche de binaire et suppression du fallback CLI terminal dans `codex_desktop.py`
- [x] **Action**: Modifier `resolve_codex_desktop_binary()` dans `src/free_claude_code/cli/launchers/codex_desktop.py` pour retirer toutes les entrées candidates pointant vers le CLI `codex` (`/snap/bin/codex`, `~/.local/bin/codex`, `/opt/codex/codex`, et le fallback `shutil.which("codex")`).
- [x] **Verify**: `uv run pytest tests/cli/test_codex_desktop_launcher.py`
- **Verification Proof**:
```text
13 passed in 5.92s
```

### Step 2: Mise à jour des tests unitaires dans `test_codex_desktop_launcher.py`
- [x] **Action**: Mettre à jour `tests/cli/test_codex_desktop_launcher.py` pour ajouter des assertions explicites garantissant que la commande `codex` (CLI) n'est jamais résolue comme binaire Codex Desktop, et que seule la présence de `codex-desktop` (ou des chemins d'accès GUI explicites) est acceptée.
- [x] **Verify**: `uv run pytest tests/cli/test_codex_desktop_launcher.py`
- **Verification Proof**:
```text
test_resolve_codex_desktop_binary_rejects_codex_cli PASSED
```

### Step 3: Bump version SemVer & régénération `uv.lock`
- [x] **Action**: Augmenter le numéro de version dans `pyproject.toml` (`4.19.1` -> `4.19.2`) et exécuter `uv lock`.
- [x] **Verify**: `uv lock --check`
- **Verification Proof**:
```text
Updated free-claude-code v4.19.1 -> v4.19.2
```

### Step 4: Validation de l'intégralité des 5 jobs CI du dépôt
- [x] **Action**: Exécuter `./scripts/ci.sh` pour vérifier les 5 garde-fous CI.
- [x] **Verify**: `./scripts/ci.sh`
- **Verification Proof**:
```text
2902 passed, 69 skipped in 74.05s
All selected CI checks passed.
```

## ⚠️ Mitigations & Edge Cases
- **Risk**: Des installations Linux existantes de Codex Desktop pourraient nommer le binaire `codex-desktop` dans des dossiers non standards.
- **Mitigation**: Conserver le support prioritaire de la variable d'environnement `CODEX_DESKTOP_PATH` pour permettre la surcompilation manuelle si nécessaire.
