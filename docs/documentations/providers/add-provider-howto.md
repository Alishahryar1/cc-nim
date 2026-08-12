<!-- Classification Diátaxis :
1. Le lecteur cherche à accomplir une tâche pratique précise (intégrer un nouveau fournisseur IA dans le codebase).
2. Le lecteur possède déjà les compétences de développement Python et connaît l'architecture du projet.
Type = How-to Guide. -->

# Comment ajouter un nouveau provider IA

Ce guide explique étape par étape comment ajouter et enregistrer un nouveau fournisseur d'IA dans Free Claude Code (FCC).

## Prérequis

- Un environnement de développement Python configuré avec `uv`.
- Les accès ou la clé d'API du nouveau fournisseur d'IA.
- Savoir si le fournisseur utilise le format OpenAI Chat Completions ou Anthropic Messages.

---

## Étapes d'Intégration

### 1. Déclarer le Descripteur dans le Catalogue

Ouvrez `src/free_claude_code/config/provider_catalog.py` et ajoutez la constante d'URL de base et l'entrée dans le dictionnaire `PROVIDER_CATALOG` :

```python
NEWPROVIDER_DEFAULT_BASE = "https://api.newprovider.com/v1"

PROVIDER_CATALOG["newprovider"] = ProviderDescriptor(
    provider_id="newprovider",
    display_name="NewProvider AI",
    credential_env="NEWPROVIDER_API_KEY",
    credential_url="https://console.newprovider.com/keys",
    credential_attr="newprovider_api_key",
    default_base_url=NEWPROVIDER_DEFAULT_BASE,
    proxy_attr="newprovider_proxy",
)
```

### 2. Créer le Module d'Adaptateur Runtime

Créez un dossier dans `src/free_claude_code/providers/newprovider/` contenant au minimum `client.py` et `__init__.py`.

Implémentez la classe en héritant de `BaseProvider` :

```python
# src/free_claude_code/providers/newprovider/client.py
from collections.abc import AsyncIterator
from free_claude_code.providers.base import BaseProvider, ProviderConfig
from free_claude_code.core.anthropic.models import MessagesRequest


class NewProvider(BaseProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)

    async def stream_messages(self, request: MessagesRequest) -> AsyncIterator[dict]:
        # Implémentation du stream HTTP/SSE
        yield {}
```

### 3. Enregistrer l'Adaptateur dans la Factory Runtime

Ouvrez `src/free_claude_code/providers/runtime/factory.py` et ajoutez le créateur d'adaptateur pour `"newprovider"` dans la fonction `create_provider()` :

```python
if provider_id == "newprovider":
    from free_claude_code.providers.newprovider.client import NewProvider

    return NewProvider(config)
```

### 4. Mapper les Erreurs Spécifiques

Si le fournisseur retourne des codes d'erreur ou formats HTTP particuliers, complétez `src/free_claude_code/providers/error_mapping.py` afin d'élever les exceptions canoniques (`AuthenticationError`, `RateLimitError`, `InvalidRequestError`).

---

## Vérification et Validation

Exécutez la suite de tests automatisés pour confirmer l'absence de régression et valider l'intégrité du catalogue :

```bash
# Formater et vérifier la qualité du code
uv run ruff format
uv run ruff check

# Vérifier les types
uv run ty check

# Exécuter les tests unitaires du catalogue et des providers
uv run pytest tests/unit/test_provider_catalog.py -v
```

Assurez-vous que le test de contrat d'importation neutre du catalogue passe sans importer les modules lourds de runtime.
