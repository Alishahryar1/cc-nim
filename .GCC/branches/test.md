# Test Log

## Completed Tests
- 2026-08-04: tests classifier ciblés -> 169 passed (Configuration, manifest, routing, verdict valide/invalide, fallback et épuisement).
- 2026-08-04: tests admin/configuration/routing -> 193 passed.
- 2026-08-04: tests Gemini/classifier ciblés -> 44 passed; thinking scope, `thinking_level`, `thinking_budget`, modèle wire et classifier Gemini 3 validés.
- 2026-08-04: smoke HTTP classifier réel -> HTTP 200 SSE, verdict Gemini `<block>no</block>`, log `api.classifier_fallback.selected` sur le candidat 1.
- 2026-08-04: suite hors uninstall -> 1716 passed, 5 skipped.
- 2026-08-04: `./scripts/ci.sh` -> 1723 exécutés, 3 tests uninstall bloqués par un processus externe `fcc-server/fcc-claude`; suppression type-ignore, Ruff format/check et ty validés.
- 2026-06-23: test_thinking_budget_retry.py -> 1 passed (Validation du mécanisme de retry du thinking budget).
- 2026-06-23: pytest (Toute la suite de tests) -> 1757 passed (Validation globale de non-régression).
- 2026-06-23: ruff check & format -> All checks passed (Conformité statique).
