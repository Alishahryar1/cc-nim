# Execution Plan: Integration du Provider Google Antigravity (Zero-Config CLI Fingerprint)

## 📋 Target Invariant & Pre-requisites
- **Target Invariant**: Le provider Antigravity doit s'auto-authentifier à partir du jeton CLI local (`~/.gemini/antigravity-cli/antigravity-oauth-token`, variables d'environnement, ou jetons gcloud) sans aucune saisie manuelle dans l'Admin UI (`localhost`). Toutes les requêtes HTTP envoyées à l'API Cloud Code Assist (`cloudcode-pa.googleapis.com`) doivent reproduire fidèlement l'**empreinte exacte d'Antigravity CLI** (`User-Agent: AntigravityCLI/1.1.11`, `X-Client-Name: antigravity-cli`, protocole `cloud.developer_experience.cloudcode.pkg.modelproxy`). Les 1747+ tests et la CI (`./scripts/ci.sh`) doivent rester à 100% au vert.
- **Pre-requisites**:
  - Module d'auto-détection et rafraîchissement OAuth CLI (`providers/antigravity/auth.py`).
  - Adaptateur de transport CodeAssist/Gemini avec en-têtes et métadonnées d'empreinte Antigravity CLI v1.1.11 (`providers/antigravity/client.py`).
  - Enregistrement neutral dans `config/provider_catalog.py`, `config/settings.py`, `api/admin_config/provider_manifest.py`, et `providers/runtime/factory.py`.

## 🛠️ Step-by-Step Sequence

### Step 1: Module d'auto-détection du jeton CLI (`~/.gemini/antigravity-cli/`) & OAuth Refresh
- [ ] **Action**: Créer `providers/antigravity/auth.py` pour :
  1. Lire la structure JSON du jeton CLI dans `~/.gemini/antigravity-cli/antigravity-oauth-token` (`{"token": {"access_token": "...", "refresh_token": "...", "expiry": "...", "token_type": "Bearer"}, "auth_method": "consumer"}`).
  2. Prévoir les fallbacks secondaires : `ANTIGRAVITY_ACCESS_TOKEN` / `ANTIGRAVITY_REFRESH_TOKEN` depuis l'environnement ou `~/.config/antigravity/oauth_token.json`.
  3. Inspecter l'expiration du JWT (marge de 300s). Si expiré, exécuter le refresh OAuth via `POST https://oauth2.googleapis.com/token`.
  4. Interroger `POST https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist` avec l'empreinte CLI pour récupérer le `cloudaicompanionProject` (fallback: `rising-fact-p41fc`).
- [ ] **Verify**: `uv run pytest tests/providers/test_antigravity_auth.py -v`
- **Verification Proof**:
```text
[A remplir après exécution du test d'authentification]
```

### Step 2: Client Provider Antigravity avec l'empreinte CLI v1.1.11 exacte
- [ ] **Action**: Créer `providers/antigravity/client.py` pour :
  1. Injecter les en-têtes HTTP de l'empreinte Antigravity CLI v1.1.11 :
     - `User-Agent`: `AntigravityCLI/1.1.11`
     - `X-Client-Name`: `antigravity-cli`
     - `X-Goog-Api-Client`: `gl-go/1.22.0 gd/1.1.11`
     - `Authorization`: `Bearer <accessToken>`
  2. Structurer le corps de requête pour inclure la signature de proxy `cloud.developer_experience.cloudcode.pkg.modelproxy` :
     ```json
     {
       "metadata": {
         "ideType": "ANTIGRAVITY_CLI",
         "platform": "PLATFORM_UNSPECIFIED"
       }
     }
     ```
  3. Formater la conversion des messages/outils Anthropic vers Gemini `contents` / `generationConfig`.
  4. Parser la réponse de streaming SSE (`candidates[0].content.parts`).
- [ ] **Verify**: `uv run pytest tests/providers/test_antigravity_client.py -v`
- **Verification Proof**:
```text
[A remplir après exécution des tests unitaires client]
```

### Step 3: Enregistrement dans le Catalogue, les Settings et la Factory Runtime
- [ ] **Action**:
  1. Déclarer `ANTIGRAVITY_DEFAULT_BASE = "https://cloudcode-pa.googleapis.com"` et le descripteur `antigravity` dans [`config/provider_catalog.py`](file:///home/omni/free-claude-code/config/provider_catalog.py) avec `static_credential="auto_discovered"` et l'empreinte CLI.
  2. Ajouter les attributs `antigravity_base_url`, `antigravity_project_id`, `antigravity_proxy` dans [`config/settings.py`](file:///home/omni/free-claude-code/config/settings.py).
  3. Déclarer les constantes par défaut dans [`providers/defaults.py`](file:///home/omni/free-claude-code/providers/defaults.py).
  4. Enregistrer `_create_antigravity` dans [`providers/runtime/factory.py`](file:///home/omni/free-claude-code/providers/runtime/factory.py).
- [ ] **Verify**: `uv run pytest tests/contracts/test_provider_catalog_order.py -v`
- **Verification Proof**:
```text
[A remplir après vérification des contrats de catalogue]
```

### Step 4: Intégration sur le Panneau d'Administration UI (`localhost`)
- [ ] **Action**:
  1. Mettre à jour [`api/admin_config/provider_manifest.py`](file:///home/omni/free-claude-code/api/admin_config/provider_manifest.py) pour exposer `Antigravity`.
  2. Indiquer le statut visuel du fichier `~/.gemini/antigravity-cli/antigravity-oauth-token` ("*Antigravity CLI Token v1.1.11 Détecté*" / "*Non détecté*").
  3. Rendre le champ `API Key` masqué/non requis en raison de la détection automatique (`auto_discovered`), tout en conservant les réglages avancés (`Base URL`, `Proxy`).
- [ ] **Verify**: `uv run pytest tests/api/test_admin.py -v`
- **Verification Proof**:
```text
[A remplir après vérification des endpoints d'admin]
```

### Step 5: Bump SemVer, Documentation et Validation CI Intégrale
- [ ] **Action**:
  1. Mettre à jour `.env.example` et la liste des providers supportés.
  2. Modifier `pyproject.toml` (`2.8.0` -> `2.9.0`) et exécuter `uv lock`.
  3. Lancer la suite de validation CI complète `./scripts/ci.sh`.
- [ ] **Verify**: `./scripts/ci.sh`
- **Verification Proof**:
```text
[A remplir après validation finale du script CI]
```

## ⚠️ Mitigations & Edge Cases
- **Risk**: Le jeton OAuth local `~/.gemini/antigravity-cli/antigravity-oauth-token` est invalide ou la session CLI a expiré.
- **Mitigation**: Émettre un message d'erreur d'authentification précis ("Jeton Antigravity CLI expiré. Lancez la CLI agy pour renouveler la session") sans crash du serveur proxy.
- **Risk**: Modèles Gemini 3.6 / Pro spécifiques à la CLI rejetés si `ideType` est incorrect.
- **Mitigation**: Valider l'envoi systématique des headers `X-Client-Name: antigravity-cli` et du payload `metadata.ideType = ANTIGRAVITY_CLI`.
