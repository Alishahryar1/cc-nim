<!--
Classification Diátaxis :
1. Lecteur accomplissant une tâche pratique ciblée (Doing).
2. Lecteur connaissant le système et cherchant le mode d'emploi de Codex Desktop (Already knows).
Type = How-to Guide.
-->

# Comment installer et configurer le lanceur Codex Desktop

Ce guide explique comment installer, configurer et utiliser le lanceur Codex Desktop (`fcc-codex-desktop`) sous Linux, macOS et Windows pour connecter l'application graphique ChatGPT / Codex Desktop au proxy Free Claude Code.

## Prérequis

- Le paquet `free-claude-code` installé
- L'application ChatGPT / Codex Desktop installée sur votre système
- Le serveur proxy `fcc-server` en cours d'exécution sur `http://127.0.0.1:8082`

## Étapes de configuration

### 1. Identifier l'emplacement de l'exécutable Codex Desktop

Le lanceur recherche automatiquement le binaire selon votre système d'exploitation :

- **macOS** : `/Applications/ChatGPT.app/Contents/MacOS/ChatGPT` ou `/Applications/Codex.app/...`
- **Windows** : `%LOCALAPPDATA%\Programs\ChatGPT\ChatGPT.exe` ou `%PROGRAMFILES%\ChatGPT\ChatGPT.exe`
- **Linux** : `/usr/bin/chatgpt`, `/usr/bin/codex-desktop`, `/snap/bin/codex-desktop` ou `~/.local/bin/codex-desktop`

Si votre binaire est dans un dossier spécifique, définissez la variable d'environnement `CODEX_DESKTOP_PATH` :

```bash
export CODEX_DESKTOP_PATH="/opt/mon-chemin/codex-desktop"
```

### 2. Appliquer la configuration persistante

Pour inscrire le provider `fcc` de façon permanente dans le fichier `~/.codex/config.toml` sans démarrer immédiatement l'application :

```bash
fcc-codex-desktop --setup
```

Cette commande crée automatiquement une sauvegarde de votre configuration initiale sous `~/.codex/config.toml.fccbak`.

### 3. Lancer l'application avec le proxy éphémère

Pour démarrer Codex Desktop avec une injection de configuration temporaire qui restaure votre état d'origine à la fermeture :

```bash
fcc-codex-desktop
```

Si le binaire graphique n'est pas détecté dans le PATH système, le lanceur applique la configuration persistante et affiche les instructions pour lancer l'application depuis votre menu système.

### 4. Réinitialiser la configuration d'origine

Pour supprimer la configuration Free Claude Code et restaurer votre fichier `config.toml` d'origine :

```bash
fcc-codex-desktop --reset
```

Ou utilisez l'alias équivalent :

```bash
fcc-codex-desktop --restore
```

## Vérification

Pour vérifier que la configuration a été correctement injectée dans `~/.codex/config.toml`, inspectez le contenu du fichier :

```bash
cat ~/.codex/config.toml
```

Le fichier doit contenir le bloc suivant :

```toml
model_provider = "fcc"

[model_providers.fcc]
name = "Free Claude Code"
base_url = "http://127.0.0.1:8082/v1"
wire_api = "responses"

[model_providers.fcc.auth]
command = "fcc-codex"
args = ["--print-proxy-auth-token"]
```
