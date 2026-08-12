<!-- Classification Diátaxis :
1. Le lecteur cherche à comprendre les concepts sous-jacents, l'architecture et les choix d'ingénierie (Mode Réflexion / Thinking).
2. Le lecteur découvre l'organisation interne du fournisseur Google Antigravity dans Free Claude Code (Apprentissage initial / Learning for the first time).
Type = Explanation. -->

# Architecture de Google Antigravity CLI

## 1. Problématique & Contexte

Le fournisseur Google Antigravity (`src/free_claude_code/providers/antigravity/`) permet à des agents de codage (Claude Code, OpenAI Codex) d'interagir avec l'infrastucture Google Cloud Code PA (`cloudcode-pa.googleapis.com`). 

Contrairement aux API d'IA traditionnelles basées sur des clés d'API statiques, Google Antigravity impose :
1. Une authentification OAuth 2.0 PKCE dynamique liée à un profil d'utilisation IDE.
2. Une empreinte d'en-tête HTTP stricte imitant le Language Server Google Cloud Code (`antigravity/1.1.11`).
3. Un assainissement strict des schémas JSON Schema Draft-07 (interdisant `$schema`, `const`, `propertyNames`).
4. Un format Server-Sent Events (SSE) nécessitant la déduplication d'outils et le support des blocs de réflexion (*thinking*) multi-tours avec signatures.

---

## 2. 🌳 Diagramme d'Architecture & Flux de Données

```text
[ 👤 USER ]
   │
   │ 1. Instruction / CLI Command (e.g. `fcc-claude`)
   ▼
[ 💻 CLAUDE CODE CLI ] (Client Anthropic Officiel)
   │
   │ 2. Envoie la requête Anthropic HTTP (POST /v1/messages)
   │    Payload JSON Anthropic: { messages, tools, system, max_tokens, thinking }
   ▼
[ ⚙️ FREE-CLAUDE-CODE ] (Serveur Proxy Local FastAPI sur 127.0.0.1:8082)
   │
   ├── A. Authentification & Découverte des Jetons:
   │      Lit le token Google OAuth dans `~/.gemini/antigravity-cli/antigravity-oauth-token`
   │
   ├── B. Conversion Anthropic ➔ Google Gemini:
   │      - En-têtes d'Empreinte: User-Agent "antigravity/1.1.11 (Linux)", Client-Name "ANTIGRAVITY"
   │      - Assainissement d'Outils: `_clean_gemini_schema` (supprime $schema, const, propertyNames, exclusiveMinimum)
   │      - Conversion de l'Historique: Transforme le thinking précédent en `{"thought": true, "text": "..."}`
   │      - Support Multi-Tours: Injection de `thought_signature` et `functionCall` / `functionResponse`
   │
   │ 3. Envoie la requête REST HTTPS directe (POST /v1internal:streamGenerateContent?alt=sse)
   ▼
[ ☁️ GOOGLE CLOUD CODE ASSIST ] (Serveur Cloud Google Antigravity)
   │
   │ 4. Traite la requête sur un modèle Antigravity (ex: `gemini-3.6-flash-high`)
   │ 5. Renvoie le flux SSE Google (JSON streamés)
   ▼
[ ⚙️ FREE-CLAUDE-CODE ] (Serveur Proxy Local)
   │
   ├── C. Analyse du Flux SSE Google:
   │      - Détecte `{"thought": true, "text": "..."}` ➔ Émet les événements de réflexion
   │      - Détecte `{"text": "..."}` ➔ Émet le texte final
   │      - Déduplique les appels d'outils via `AnthropicStreamLedger`
   │
   ├── D. Traduction SSE Google ➔ Anthropic:
   │      - `event: content_block_start` (type: thinking ou text)
   │      - `event: content_block_delta` (thinking_delta ou text_delta)
   │
   │ 6. Ré-émet le flux SSE au format 100% compatible Anthropic
   ▼
[ 💻 CLAUDE CODE CLI ] ➔ [ 👤 Rendu visuel dans le terminal ]
```

---

## 3. 🔑 Comparatif des Flux d'Authentification OAuth (CLI vs IDE)

L'authentification s’appuie sur le composant `AntigravityAuth` (`auth.py`) avec recherche hiérarchique :
1. `~/.gemini/antigravity-cli/antigravity-oauth-token` (Priorité 1)
2. `~/.config/antigravity/oauth_token.json` (Priorité 2)
3. `~/.gemini/oauth_creds.json` (Priorité 3)

| Fonctionnalité | Authentification CLI (`agy`) | Authentification IDE (`antigravity_login.py`) |
| :--- | :--- | :--- |
| **Type de Flux OAuth** | **Device Code Flow** (RFC 8628 / OOB) | **Web Authorization Code Flow** |
| **Saisie Utilisateur** | Saisie manuelle d'un code alphanumérique | Connexion navigateur en 1 clic (`localhost:8085`) |
| **Client ID Google** | OAuth Client CLI Terminal | **Antigravity IDE** OAuth Client (`1071006060591-...`) |
| **Profil Serveur Google** | Profil "CLI Terminal" | Profil **"Antigravity IDE / Full Platform"** |
| **Modèles Disponibles** | **22 modèles** (Chat uniquement) | **48 modèles** (Chat, Autocompletion `tab-`, Images) |
| **Support des Outils** | ❌ Incompatible / Tronqué | ✅ 100% Fonctionnel |

---

## 4. 🛡️ Empreinte HTTP du Language Server (*Fingerprinting*)

Le Language Server Google Antigravity (`language_server_pb`) communique avec les serveurs Google via des en-têtes d'empreinte stricts. Dans `free-claude-code`, cette empreinte est reproduite exactement via `httpx` (`client.py`) :

```python
ANTIGRAVITY_USER_AGENT = "antigravity/1.1.11 (Linux)"
ANTIGRAVITY_CLIENT_NAME = "ANTIGRAVITY"
ANTIGRAVITY_GOOG_API_CLIENT = "gl-python/3.14.0 grpc/1.62.0 gax/2.17.0"

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
    "User-Agent": ANTIGRAVITY_USER_AGENT,
    "X-Goog-Api-Client": ANTIGRAVITY_GOOG_API_CLIENT,
    "Client-Name": ANTIGRAVITY_CLIENT_NAME,
}
```

---

## 5. 🧰 Assainissement des Schémas d'Outils (*Tool Call Sanitation*)

Les outils transmis par Claude Code (`exec_command`, `file_read`, etc.) utilisent la norme JSON Schema Draft-07. Le parseur OpenAPI strict de Google Gemini rejette ces requêtes (erreur HTTP 400).

La fonction `_clean_gemini_schema` nettoie récursivement les schémas avant transmission :
```python
UNSUPPORTED_GEMINI_SCHEMA_KEYS = {
    "$schema", "$id", "$comment", "propertyNames", "const",
    "exclusiveMinimum", "exclusiveMaximum", "patternProperties",
    "unevaluatedProperties", "unevaluatedItems", "contains",
    "minContains", "maxContains",
}
```

---

## 6. 🧠 Rendu de la Réflexion (*Thinking*) & Historique Multi-Tours

### Flux Entrant (Google ➔ Free-Claude-Code)
Dans le flux SSE d'un modèle avec réflexion activée, Google renvoie :
```json
{ "thought": true, "text": "**Analyse de la tâche**\n\nProcessus de réflexion..." }
```
`client.py` intercepte `thought: true` et redirige le texte vers `ledger.emit_thinking_delta()`.

### Historique Multi-Tours (Free-Claude-Code ➔ Google)
Lorsqu'un message précédent de l'assistant contient du texte de réflexion, Google exige un booleen strict (`TYPE_BOOL`) pour la clé `thought` dans le tableau `parts` de la requête :
```json
{
  "role": "model",
  "parts": [{ "thought": true, "text": "**Analyse de la tâche**\n\nProcessus de réflexion..." }]
}
```

### Signatures d'Outils (*Thought Signatures*)
Chaque objet `functionCall` dans l'historique assistant inclut automatiquement `"thought": True` et `"thought_signature": ts or "skip_thought_signature_validator"` pour valider les contraintes de l'API Google.

---

## 7. 🌊 Déduplication des Outils en Streaming SSE

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

---

## 8. 🚨 Défis d'Ingénierie Résolus (Post-Mortem & Solutions)

1. **Erreur HTTP 400 sur Schémas d'Outils (`$schema`, `const`)** : Résolu via la purge récursive `_clean_gemini_schema` avant conversion en `functionDeclarations`.
2. **Erreur HTTP 400 `TYPE_BOOL` sur l'Historique de Réflexion** : Résolu en convertissant la réflexion précédente en objet partiel avec `"thought": True` (type booléen Python).
3. **Erreur `thought_signature` Manquante lors des Appels d'Outils** : Auto-injection systématique de `"thought_signature": "skip_thought_signature_validator"` sur chaque `functionCall`.
4. **Mauvaise Interprétation de `thought: true` dans le Flux SSE** : Interception stricte de `part.get("thought") is True` pour garantir l'émission exclusive dans des blocs `thinking_delta` Anthropic.

---

## 9. Compromis & Perspectives

| Choix d'Architecture | Avantage | Compromis |
| :--- | :--- | :--- |
| **Authentification PKCE IDE local** | Accès complet aux 48 modèles et 100% du support des outils. | Nécessite un serveur de callback local temporaire sur le port 8085. |
| **Recherche hiérarchique de tokens** | Réutilise les jetons d'une installation CLI ou IDE existante. | Nécessite de vérifier l'expiration JWT à chaque démarrage. |
| **Dédoublonnement via Ledger** | Supprime 100% des doublons d'appels d'outils en streaming. | Conserve un état d'accumulation en mémoire pendant la durée du stream. |
