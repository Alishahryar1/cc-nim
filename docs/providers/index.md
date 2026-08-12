# Domaine Providers & Catalogue d'IA

Le domaine **Providers & Catalogue d'IA** constitue la couche d’abstraction d'accès aux modèles d’intelligence artificielle dans Free Claude Code (FCC). Il permet d’interagir de manière uniforme avec plus de 20 fournisseurs d'IA (cloud commercial, comptes connectés, passerelles open-source et serveurs locaux) tout en garantissant un comportement homogène des flux de messages, du streaming et des politiques de réflexion (*reasoning*).

---

## 🏛️ Architecture du Domaine

Le domaine est structuré autour d'une séparation stricte des responsabilités :

1. **Catalogue Neutre (`src/free_claude_code/config/provider_catalog.py`)** :
   Déclare les métadonnées statiques des fournisseurs (`ProviderDescriptor`), leurs clés de configuration, URLs d'API par défaut et variables d'environnement, sans importer les dépendances d'exécution des adaptateurs.
2. **Adaptateurs Runtime (`src/free_claude_code/providers/`)** :
   Contient les sous-modules spécifiques à chaque provider (ex. `antigravity`, `openai_chat`, `deepseek`, `nvidia_nim`, `open_router`) héritant de `BaseProvider`.
3. **Usine et Directives (`src/free_claude_code/providers/runtime/`)** :
   Instancie dynamiquement les adaptateurs runtime selon la configuration utilisateur, gère le taux de limite (*rate limiting*), le basculement en cas de panne (*failure policy*) et le routage des jetons.

---

## 📑 Documentation Diátaxis du Domaine

Consultez les guides spécialisés suivants selon votre besoin :

| Document | Type Diátaxis | Description |
| :--- | :--- | :--- |
| [Référence du Catalogue des Providers](file:///home/omni/free-claude-code-pr/docs/providers/provider-catalog-reference.md) | **Reference** | Fiche technique exhaustive des 20+ fournisseurs supportés (URLs, variables d'environnement, modes d'authentification). |
| [Architecture Google Antigravity](file:///home/omni/free-claude-code-pr/docs/providers/antigravity-explanation.md) | **Explanation** | Fonctionnement détaillé de Google Antigravity CLI, du flux OAuth2 PKCE et de la déduplication d'outils en streaming SSE. |
| [Ajouter un Provider IA](file:///home/omni/free-claude-code-pr/docs/providers/add-provider-howto.md) | **How-to Guide** | Guide étape par étape pour intégrer un nouveau fournisseur compatible OpenAI ou Anthropic. |
| [Politique de Réflexion (Reasoning)](file:///home/omni/free-claude-code-pr/docs/providers/reasoning-policy-explanation.md) | **Explanation** | Explication du mapping agnostique du budget de réflexion (*thinking budget*) vers les formats API spécifiques des providers. |

---

## 🔒 Principes de Sécurité et Robustesse

- **Isolation des Crédentiels** : Chaque fournisseur consomme ses clés de manière isolée via l'environnement ou la configuration.
- **Rattrapage d'Erreurs Majeures** : Les exceptions HTTP/gRPC sont traduites dans les classes canoniques de `free_claude_code.providers.exceptions` (`AuthenticationError`, `RateLimitError`, `OverloadedError`).
- **Absence de Dépendance aux Noms de Modèles** : La gestion des capacités (réflexion, outils, streaming) repose sur des politiques explicites et non sur le découpage de chaînes de caractères de noms de modèles.
