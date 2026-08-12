<!-- Classification Diátaxis :
1. Le lecteur cherche à comprendre l'architecture théorique et l'abstraction de la réflexion (reasoning/thinking).
2. Le lecteur apprend pour la première fois comment Free Claude Code unifie les budgets de réflexion hétérogènes des LLMs.
Type = Explanation. -->

# Politique de Réflexion Agnostique (Reasoning Policy)

## Le Problème de l'Hétérogénéité du Reasoning

Les modèles d'IA récents (Anthropic Claude 3.7 Sonnet, OpenAI o1/o3-mini, DeepSeek-R1, Google Gemini 2.0 Flash Thinking) prennent en charge la réflexion étendue (*thinking / reasoning*). Néanmoins, l'API de chaque fournisseur impose sa propre syntaxe d'activation et de contrôle du budget :

- **Anthropic** : `thinking: {"type": "enabled", "budget_tokens": 2048}`
- **OpenAI / Azure** : `reasoning_effort: "low" | "medium" | "high"`
- **DeepSeek / Groq** : `extra_body: {"thinking": {"enabled": true}}`
- **llama.cpp** : `extra_body: {"thinking_budget_tokens": 2048}`
- **Mistral** : `extra_body: {"chat_template_kwargs": {"thinking": true}}`

Résoudre cette hétérogénéité au niveau des composants UI ou du cœur de l'application engendrerait un couplage fort et des branchements conditionnels complexes basés sur les noms de modèles.

---

## L'Abstraction Neutre : `ReasoningPolicy`

Free Claude Code résout ce problème en introduisant la politique neutre `ReasoningPolicy` dans `src/free_claude_code/core/reasoning.py`.

```
        ┌─────────────────────────────────────────┐
        │ Requête Client (MessagesRequest)        │
        │ + Configuration (ReasoningPreference)   │
        └────────────────────┬────────────────────┘
                             │
                             ▼  resolve_reasoning_policy()
        ┌─────────────────────────────────────────┐
        │            ReasoningPolicy              │
        │  • control: DEFAULT | OFF | ON          │
        │  • effort: MINIMAL..MAX                 │
        │  • budget_tokens: int | None            │
        └────────────────────┬────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────┐
│ NamedEffortReasoning  │         │ LlamaCppReasoning     │
│ (OpenAI/NIM/Groq)     │         │ (llama.cpp)           │
│ reasoning_effort="high│         │ thinking_budget=2048  │
└───────────────────────┘         └───────────────────────┘
```

### Niveaux d'Effort et Jetons Déduits (`ReasoningEffort`)

Le tableau ci-dessous indique la conversion automatique entre les niveaux d'effort nommés et le budget de jetons sous-jacent :

| `ReasoningEffort` | Budget de jetons (`budget_tokens`) |
| :--- | :--- |
| `MINIMAL` | 512 jetons |
| `LOW` | 512 jetons |
| `MEDIUM` | 1 024 jetons |
| `HIGH` | 2 048 jetons |
| `XHIGH` | 4 096 jetons |
| `MAX` | 8 192 jetons |

---

## Encodeurs de Réflexion Côté Provider (`ReasoningEncoder`)

Dans chaque adaptateur de provider (ex. `providers/openai_chat/reasoning.py`), une classe implémentant le protocole `ReasoningEncoder` traduit l'objet `ReasoningPolicy` neutre vers le format JSON filaire attendu par l'API :

1. **`NamedEffortReasoning`** : Traduit les niveaux nommés vers des paramètres de chaîne de caractères (`reasoning_effort: "medium"`).
2. **`ReasoningObject`** : Génère un objet structuré d'options (`reasoning: {"effort": "high", "max_tokens": 2048}`).
3. **`ThinkingObjectReasoning`** : Active ou désactive l'objet top-level (`thinking: {"enabled": True}`).
4. **`ChatTemplateReasoning`** : Injecte le drapeau de modèle de chat (`chat_template_kwargs: {"thinking": True}`).
5. **`LlamaCppReasoning`** : Transmet directement le budget numérique `thinking_budget_tokens`.

---

## Compromis et Alignement Architectural

- **Indépendance des modèles** : Aucun adaptateur ne teste le nom ou la version du modèle (ex. `if "o1" in model_name`) pour décider du format de raisonnement.
- **Résolution Unique** : La décision d'activer la réflexion et son budget est calculée une seule fois à la frontière applicative via `resolve_reasoning_policy()`.
- **Préférence Utilisateur** : Si la configuration impose `ReasoningPreference.OFF`, la réflexion est désactivée globalement quel que soit le modèle sélectionné.
