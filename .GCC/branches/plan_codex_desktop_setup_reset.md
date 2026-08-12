# Execution Plan: Codex Desktop Hybrid Launcher with OpenAI ChatGPT/Codex GUI Binary Resolution

## 📋 Target Invariant & Pre-requisites
- **Target Invariant**: `fcc-codex-desktop` offre un fonctionnement hybride éphémère / persistant, en intégrant la résolution des exécutables officiels OpenAI Linux (`/usr/bin/chatgpt`, `/usr/lib/chatgpt/codex-launcher`, `/usr/lib/chatgpt/ChatGPT`), macOS (`ChatGPT.app`, `Codex.app`) et Windows (`ChatGPT.exe`, `Codex.exe`) :
  1. **Résolution de binaire GUI élargie (`resolve_codex_desktop_binary()`)** :
     - Prise en compte prioritaire des chemins officiels du paquet OpenAI (`/usr/bin/chatgpt`, `/usr/lib/chatgpt/codex-launcher`, `/usr/lib/chatgpt/ChatGPT`, `/opt/chatgpt/chatgpt`, `shutil.which("chatgpt")`), tout en conservant les candidats `codex-desktop` et les surcharges `CODEX_DESKTOP_PATH`.
  2. **Mode Éphémère (Exécution directe avec binaire GUI résolu)** :
     - Si un binaire GUI (`chatgpt` ou `codex-desktop`) est résolu sur le système, injecter temporairement la configuration `model_provider = "fcc"` dans `~/.codex/config.toml`, lancer l'application et restaurer 100% du fichier d'origine dès la fermeture de l'application.
  3. **Mode Persistant Automatique (Fallback si binaire GUI non détecté)** :
     - Si aucun binaire GUI n'est trouvé dans les chemins standards, créer/mettre à jour la configuration permanente `~/.codex/config.toml` vers le proxy FCC, sauvegarder la version d'origine sous `~/.codex/config.toml.fccbak`, puis afficher le message d'instruction en anglais :
       ```text
       [Free Claude Code] Codex Desktop / ChatGPT GUI binary was not found in standard PATH.
       Persistent configuration applied to ~/.codex/config.toml.

       Setup completed! Please launch ChatGPT / Codex Desktop from your application menu or shortcut.

       To restore your original configuration at any time, run:
         fcc-codex-desktop --reset
       ```
  4. **Commande explicite `--setup`** (`fcc-codex-desktop --setup`) :
     - Applique la configuration permanente `model_provider = "fcc"` dans `~/.codex/config.toml`, sauvegarde la version pré-existante et affiche le message d'instruction sans lancer de processus.
  5. **Commande explicite `--reset` / `--restore`** (`fcc-codex-desktop --reset` ou `--restore`) :
     - Restaure le fichier `~/.codex/config.toml` dans son état initial d'origine (ou supprime les clés `fcc` injectées) et supprime `config.toml.fccbak`. Affiche :
       ```text
       [Free Claude Code] Configuration reset successfully!
       Codex Desktop configuration restored to original settings.
       ```
- **Pre-requisites**: `uv run pytest`, `./scripts/ci.sh`.

## 🛠️ Step-by-Step Sequence

### Step 1: Mise à jour de `resolve_codex_desktop_binary()` et implémentation de `--setup` / `--reset` dans `codex_desktop.py`
- [x] **Action**: Inclure les candidats `chatgpt` (`/usr/bin/chatgpt`, `/usr/lib/chatgpt/codex-launcher`, `/usr/lib/chatgpt/ChatGPT`) dans `resolve_codex_desktop_binary()`, et ajouter `setup_persistent_config()` et `reset_persistent_config()` dans `src/free_claude_code/cli/launchers/codex_desktop.py`.
- [x] **Verify**: `uv run pytest tests/cli/test_codex_desktop_launcher.py`
- **Verification Proof**:
```text
16 passed in 6.12s
```

### Step 2: Implémentation des tests unitaires dans `test_codex_desktop_launcher.py`
- [x] **Action**: Ajouter des tests pour la résolution du binaire `chatgpt` (`/usr/bin/chatgpt`), les drapeaux `--setup`, `--reset`, `--restore` et le fallback persistant gracieux.
- [x] **Verify**: `uv run pytest tests/cli/test_codex_desktop_launcher.py`
- **Verification Proof**:
```text
test_resolve_codex_desktop_binary_chatgpt_linux PASSED
test_launch_setup_flag PASSED
test_launch_reset_and_restore_flags PASSED
test_launch_fallback_persistent PASSED
```

### Step 3: Bump SemVer `4.19.3` & `uv lock`
- [x] **Action**: Mettre à jour `version = "4.19.3"` dans `pyproject.toml` et exécuter `uv lock`.
- [x] **Verify**: `uv lock --check`
- **Verification Proof**:
```text
Updated free-claude-code v4.19.2 -> v4.19.3
```

### Step 4: Verification CI globale
- [x] **Action**: Exécuter `./scripts/ci.sh` pour valider les 5 garde-fous CI.
- [x] **Verify**: `./scripts/ci.sh`
- **Verification Proof**:
```text
2902 passed, 69 skipped in 74.05s
All selected CI checks passed.
```

## ⚠️ Mitigations & Edge Cases
- **Risk**: Conflits de noms si un binaire `chatgpt` tiers non-OpenAI existe dans le PATH.
- **Mitigation**: Privilégier les chemins d'accès absolus vérifiés du paquet OpenAI (`/usr/bin/chatgpt` pointant vers `codex-launcher`, `/usr/lib/chatgpt/ChatGPT`).
