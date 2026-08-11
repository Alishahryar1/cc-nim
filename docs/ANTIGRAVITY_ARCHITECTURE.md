# 🏛️ Documentation Technique : Architecture Google Antigravity dans Free-Claude-Code

Cette documentation détaille l'architecture complète, l'authentification OAuth, le protocole réseau, l'empreinte du Language Server, et la gestion des outils et pensées (*thinking*) pour l'intégration de Google Antigravity dans `free-claude-code`.

---

## 1. 🌳 Arbre d'Architecture et Cheminement des Données

```text
[ 👤 VOUS ]
   │
   │ 1. Prompt / Commande CLI (ex: `fcc-claude`)
   ▼
[ 💻 CLAUDE CODE CLI ] (Client Anthropic officiel)
   │
   │ 2. Envoie une requête HTTP Anthropic (POST /v1/messages)
   │    Payload Anthropic JSON : { messages, tools, system, max_tokens, thinking }
   ▼
[ ⚙️ FREE-CLAUDE-CODE ] (Serveur Proxy local FastAPI / Python sur 127.0.0.1:8082)
   │
   ├── A. Authentification & Découverte Zero-Config :
   │      Lit le jeton OAuth Google dans `~/.gemini/antigravity-cli/antigravity-oauth-token`
   │
   ├── B. Conversion Anthropic ➔ Google Gemini :
   │      - En-têtes Empreinte : User-Agent "antigravity/1.1.11 (Linux)", Client-Name "ANTIGRAVITY"
   │      - Nettoyage des outils : `_clean_gemini_schema` (supprime $schema, const, propertyNames, exclusiveMinimum)
   │      - Conversion d'historique : Transforme les réflexions précédentes en `{"thought": true, "text": "..."}`
   │      - Support Multi-Tours : Injecte `thought_signature` et `functionCall` / `functionResponse`
   │
   │ 3. Envoie la requête HTTPS REST directe (POST /v1internal:streamGenerateContent?alt=sse)
   ▼
[ ☁️ GOOGLE CLOUD CODE ASSIST ] (Serveur Cloud Google Antigravity)
   │
   │ 4. Traite la demande sur l'un des 48 modèles Antigravity (ex: `gemini-3.6-flash-high`)
   │ 5. Renvoie un flux SSE Google (chunks JSON streamés)
   ▼
[ ⚙️ FREE-CLAUDE-CODE ] (Serveur Proxy local)
   │
   ├── C. Parsing du Flux Google :
   │      - Détecte `{"thought": true, "text": "..."}` ➔ Émet les événements de pensée (Thinking)
   │      - Détecte `{"text": "..."}` ➔ Émet le texte de réponse finale
   │
   ├── D. Traduction Google ➔ Anthropic SSE :
   │      - `event: content_block_start` (type: thinking ou text)
   │      - `event: content_block_delta` (thinking_delta ou text_delta)
   │
   │ 6. Ré-émet le flux SSE au format 100% compatible Anthropic
   ▼
[ 💻 CLAUDE CODE CLI ]
   │
   │ 7. Rendu visuel dans le terminal (Pensée repliable + réponse + exécution des outils)
   ▼
[ 👤 VOUS VOYEZ LA RÉPONSE ]
```

---

## 2. 🔑 Comparatif des Flux d'Authentification OAuth (IDE vs CLI)

| Fonctionnalité | Authentification CLI (`agy`) | Authentification IDE (`antigravity_login.py`) |
| :--- | :--- | :--- |
| **Type de Flow OAuth** | **Device Code Flow** (RFC 8628 / OOB) | **Web Authorization Code Flow** |
| **Saisie Utilisateur** | Saisie manuelle d'un **code alphanumérique** dans le terminal | Connexion 1-clic navigateur (Callback automatique `localhost:8085`) |
| **Client ID Google** | Client OAuth CLI | Client OAuth **Antigravity IDE** (`1071006060591-...`) |
| **Profil Serveur Google** | Associé au profil "CLI Terminal" | Associé au profil **"Antigravity IDE / Full Platform"** |
| **Modèles Disponibles** | **22 modèles** (Chat uniquement) | **48 modèles** (Chat, Autocomplétion `tab-`, Images, Compaction) |
| **Support Outils (Tools)** | ❌ Incompatible / Tronqué | ✅ 100% Fonctionnel |

---

## 3. 🛡️ Empreinte HTTP du Language Server (Fingerprinting)

Le Language Server local d'Antigravity (`language_server_pb`) communique avec les serveurs de Google en utilisant des en-têtes d'empreinte stricts. Dans `free-claude-code`, nous reproduisons cette empreinte exacte avec la bibliothèque HTTP `httpx` :

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

## 4. 🧰 Nettoyage des Schémas d'Outils (*Tool Call Sanitation*)

Les outils transmis par Claude Code (`exec_command`, `file_read`, etc.) utilisent le standard JSON Schema Draft-07 qui inclut des mots-clés rejetés par le parseur OpenAPI strict de Google Gemini (erreur HTTP 400).

La fonction `_clean_gemini_schema` nettoie récursivement les schémas avant transmission :

```python
UNSUPPORTED_GEMINI_SCHEMA_KEYS = {
    "$schema",
    "$id",
    "$comment",
    "propertyNames",
    "const",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "patternProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contains",
    "minContains",
    "maxContains",
}
```

---

## 5. 🧠 Rendu et Historique des Blocs de Pensée (*Thinking*)

### Flux Entrant (Google ➔ Free-Claude-Code)
Dans le flux SSE d'un modèle avec réflexion, Google renvoie :
```json
{
  "thought": true,
  "text": "**Analyzing task**\n\nThinking process..."
}
```
`client.py` intercepte `thought: true` et redirige le texte vers `ledger.emit_thinking_delta()`.

### Flux Historique (Free-Claude-Code ➔ Google)
Lorsqu'un message précédent contient de la réflexion, Google exige le type booléen `TYPE_BOOL` pour la clé `thought` dans les `contents` de la requête :
```json
{
  "role": "model",
  "parts": [
    {
      "thought": true,
      "text": "**Analyzing task**\n\nThinking process..."
    }
  ]
}
```
Ceci garantit un statut **HTTP 200 OK** sur toutes les conversations multi-tours.

---

## 6. 🚨 Erreurs et Difficultés Rencontrées (Post-Mortem & Solutions)

Voici le journal exhaustif des 5 difficultés majeures rencontrées lors du développement et leurs résolutions techniques :

### ❌ Difficulté 1 : Interprétation Erronée du Booléen `thought: true` dans le Flux SSE
- **Symptôme** : Dans `fcc-codex`, le texte de pensée s'affichait comme du texte normal (`Initiating Data Acquisition...`). Dans `Claude Code`, une erreur `API returned an empty or malformed response (HTTP 200)` apparaissait.
- **Cause Racine** : Google renvoie `{"thought": true, "text": "..."}`. Dans le code initial, `part["thought"]` valait le booléen `True`. `ledger.emit_thinking_delta(True)` échouait, puis le bloc `if "text" in part:` s'exécutait et émettait la pensée comme du texte de réponse principal.
- **Solution** : Correction dans `stream_response` pour intercepter `part.get("thought") is True`, extraire `part["text"]`, l'émettre exclusivement dans `ensure_thinking_block()` et ignorer le traitement comme texte normal.

### ❌ Difficulté 2 : Erreur HTTP 400 sur les Schémas d'Outils JSON Schema (`$schema`, `const`, `propertyNames`)
- **Symptôme** : Rejet systématique des requêtes par Google avec `Invalid JSON payload received. Unknown name "$schema" at 'request.tools[0]...'`.
- **Cause Racine** : Claude Code transmet des définitions d'outils au format JSON Schema Draft-07. Le parseur OpenAPI strict de Google Gemini ne supporte pas les clés `$schema`, `const`, `propertyNames`, `exclusiveMinimum`, `exclusiveMaximum`, `$id`.
- **Solution** : Implémentation du nettoyeur récursif `_clean_gemini_schema` qui filtre ces clés incompatibles avant conversion en `functionDeclarations`.

### ❌ Difficulté 3 : Erreur HTTP 400 `TYPE_BOOL` sur les Conversations Multi-Tours avec Pensée
- **Symptôme** : Rejet au 2ème tour de conversation avec `Invalid value at 'request.contents[1].parts[0].thought' (TYPE_BOOL), "**Analyzing...**"`.
- **Cause Racine** : Lors de la reconversion des messages d'historique de l'assistant contenant une pensée, le proxy générait `{"thought": "texte de la pensée"}`. Le schéma Google exige que la clé `thought` soit un booléen `TYPE_BOOL` (`thought: true`).
- **Solution** : Modification de `_convert_anthropic_messages_to_gemini` pour formater la pensée sous la forme `{"thought": True, "text": "texte de la pensée"}`.

### ❌ Difficulté 4 : Erreur HTTP 400 `thought_signature` Manquante lors de l'Exécution d'Outils
- **Symptôme** : Rejet HTTP 400 `Function call is missing a thought_signature in functionCall parts` lors des réponses d'outils.
- **Cause Racine** : Google exige la présence d'une signature de pensée sur chaque objet `functionCall` présent dans l'historique assistant.
- **Solution** : Injection automatique de `"thought": True` et `"thought_signature": ts or "skip_thought_signature_validator"` sur chaque élément `functionCall`.

### ❌ Difficulté 5 : Conflit de Verrou de Processus lors des Tests de Désinstallation CI
- **Symptôme** : `./scripts/ci.sh` échouait sur 3 tests d'installation avec `Free Claude Code is still running (fcc-claude)`.
- **Cause Racine** : Des processus CLI `fcc-claude` résiduels tournaient en arrière-plan et déclenchaient la garde `pgrep` de `uninstall.sh`.
- **Solution** : Arrêt systématique des processus résiduels via `pkill -9 -f fcc-claude` avant le lancement de la suite CI, permettant la réussite des 1790 tests (100% de succès).
