<!--
Classification Diátaxis :
1. Lecteur réalisant une suite d'actions pratiques pas-à-pas (Doing).
2. Lecteur découvrant l'utilisation des lanceurs pour la première fois (Learning first time).
Type = Tutorial.
-->

# Démarrer avec Free Claude Code et ses lanceurs CLI

Dans ce tutoriel, vous allez démarrer le serveur proxy local Free Claude Code et exécuter une session interactive avec les trois lanceurs d'agents principaux : `fcc-claude`, `fcc-codex` et `fcc-pi`. À la fin de ce guide, vous aurez un proxy actif et trois agents connectés à votre fournisseur IA.

## Prérequis

- Python 3.14.0 ou plus récent installé avec `uv`
- Le paquet `free-claude-code` installé (`uv pip install -e .`)
- Au moins un binaire client installé (`claude`, `codex` ou `pi`)
- Un terminal ouvert

## Étape 1 — Démarrer le serveur proxy

Ouvrez votre terminal et lancez le serveur proxy :

```bash
fcc-server
```

Vous devez obtenir un affichage similaire à :

```text
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Uvicorn running on http://0.0.0.0:8082 (Press CTRL+C to quit)
```

Laissez ce terminal ouvert.

## Étape 2 — Lancer une session avec Claude Code

Ouvrez un second terminal et exécutez la commande suivante :

```bash
fcc-claude
```

Le lanceur vérifie l'accessibilité du proxy sur `http://127.0.0.1:8082`, injecte la configuration et démarre l'interface Claude Code.

Tapez votre première question dans l'invite :

```text
> Explique-moi le rôle du fichier pyproject.toml
```

Vous recevez la réponse générée par le fournisseur IA configuré dans le proxy.

## Étape 3 — Lancer une session avec Codex CLI

Dans le même terminal (ou un nouveau terminal), démarrez l'agent Codex :

```bash
fcc-codex
```

Le lanceur génère le catalogue de modèles éphémère à partir du serveur local et initialise Codex CLI avec le provider `fcc`.

Posez une question de test :

```text
> list files in current directory
```

Codex CLI exécute la commande via le proxy Free Claude Code.

## Étape 4 — Lancer une session avec Pi

Démarrez l'agent Pi avec le lanceur dédié :

```bash
fcc-pi
```

Le lanceur charge automatiquement l'extension TypeScript intégrée (`pi_extension.ts`) et enregistre le scope de modèle `free-claude-code/**`.

Interrogez l'agent :

```text
> Salut Pi, confirme ton statut.
```

L'agent Pi répond via le canal proxy établi.

## Ce que vous avez construit

Vous avez démarré le serveur proxy `fcc-server` sur le port 8082 et lancé avec succès trois agents CLI différents (`fcc-claude`, `fcc-codex`, `fcc-pi`) pointant tous de manière transparente vers le même proxy local.

## Prochaines étapes

- Configurer le lanceur de bureau avec `fcc-codex-desktop`
- Personnaliser les fournisseurs et clés API dans le fichier `.env`
- Consulter la référence des variables d'environnement dans `cli-commands-reference.md`
