<!-- Classification Diátaxis :
1. Le lecteur cherche à comprendre les concepts sous-jacents et les choix d'architecture.
2. Le lecteur découvre le fonctionnement interne de Google Antigravity (OAuth PKCE et déduplication SSE).
Type = Explanation. -->

# Architecture de Google Antigravity CLI

## Problématique & Contexte

Le fournisseur Google Antigravity (`src/free_claude_code/providers/antigravity/`) permet d'interagir avec l'infrastucture Google Cloud Code PA (`cloudcode-pa.googleapis.com`). Contrairement aux API traditionnelles basées sur des clés d'API statiques, Google Antigravity impose :
1. Une authentification OAuth 2.0 PKCE liée à un compte Google connecté.
2. Un format de réponse Server-Sent Events (SSE) qui diffuse de manière incrémentale les appels d'outils (*tool calls*), générant des risques de doublons ou de fragmentation lors de la reconstruction du flux Anthropic.

---

## 🔑 Flux d'Authentification OAuth 2.0 PKCE & Cycle de Vie des Jetons

L'authentification s’appuie sur le composant `AntigravityAuth` (`auth.py`).

```
┌─────────────────┐      1. OAuth PKCE Flow      ┌──────────────────┐
│ Free Claude Code│ ────────────────────────────>│ Google OAuth2    │
│ (Local WebServer│ <─────────────────────────── │ (accounts.google)│
└────────┬────────┘     2. Access/Refresh Token  └──────────────────┘
         │
         │ 3. Découverte hiérarchique des jetons
         ▼
┌──────────────────────────────────────────────────────────────────┐
│ Priority 1: ~/.gemini/antigravity-cli/antigravity-oauth-token    │
│ Priority 2: ~/.config/antigravity/oauth_token.json               │
│ Priority 3: ~/.gemini/oauth_creds.json                           │
└──────────────────────────────────────────────────────────────────┘
```

### Empreintes et Emulation CLI
Pour garantir la compatibilité avec l'API upstream, les requêtes injectent les en-têtes d'empreinte officielle (`ANTIGRAVITY_USER_AGENT = "AntigravityCLI/1.1.11"` et `ANTIGRAVITY_CLIENT_NAME = "antigravity-cli"`). Le projet Google Cloud par défaut est configuré sur `rising-fact-p41fc`.

---

## 🌊 Déduplication des Outils en Streaming SSE

L'API Antigravity transmet les blocs de texte et de `tool_use` sous forme de fragments SSE successifs.

### Le Problème des Fragments Récurrents
Lors d'un appel d'outil (ex. écriture de fichier ou exécution de commande), l'API Cloud Code PA renvoie parfois l'identifiant et le nom de l'outil dans plusieurs paquets SSE successifs. Sans filtrage, l'adaptateur générerait des blocs d'outils dupliqués ou mal formés, provoquant une erreur de protocole côté client Claude.

### La Solution : `AnthropicStreamLedger`
L'adaptateur `AntigravityProvider` (`client.py`) utilise le livre de comptes de flux `AnthropicStreamLedger` pour gérer l’état de la réponse SSE :

```
Événement SSE entrant
        │
        ▼
Extrait le bloc (ContentBlock / ToolUseBlock)
        │
        ├─► Identifiant Tool Use déjà enregistré dans le Ledger ?
        │        ├── OUI ──► Met à jour l'argument partiel (JSON Delta)
        │        └── NON ──► Émet `content_block_start` et enregistre l'ID
        │
        ▼
Émission propre des événements SSE Anthropic-compatibles
```

Ce mécanisme garantit une déduplication stricte des identifiants d'outils et une sérialisation JSON valide avant l'envoi au client.

---

## Compromis et Perspectives

| Choix d'Architecture | Avantage | Compromis |
| :--- | :--- | :--- |
| **Authentification PKCE local** | Pas besoin de saisir manuellement des tokens complexes. | Nécessite la gestion d'un serveur HTTP local temporaire. |
| **Recherche hiérarchique de tokens** | Réutilise les jetons d'une installation Antigravity CLI existante. | Nécessite de vérifier l'expiration JWT à chaque démarrage. |
| **Dédoublonnement via Ledger** | Supprime 100% des doublons d'appels d'outils en streaming. | Conserve un état d'accumulation en mémoire pendant la durée du stream. |
