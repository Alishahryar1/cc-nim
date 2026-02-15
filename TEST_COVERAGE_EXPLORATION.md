# Test Coverage Exploration

**Date:** 2026-02-15  
**Overall coverage:** 86%  
**Tests:** 684 passed, 1 failed (uvicorn import in `test_server_module.py`)

## Summary by Module

### High coverage (≥95%)
| Module | Coverage | Notes |
|--------|----------|------|
| `api/__init__.py` | 100% | |
| `api/dependencies.py` | 100% | |
| `api/models/*` | 100% | anthropic, responses |
| `cli/manager.py` | 99% | Line 131 uncovered |
| `messaging/tree_data.py` | 99% | Lines 288-291 uncovered |
| `messaging/factory.py` | 100% | |
| `messaging/models.py` | 100% | |
| `config/nim.py` | 100% | |
| `config/settings.py` | 100% | |
| `providers/logging_utils.py` | 100% | |
| `providers/model_utils.py` | 100% | |
| `providers/nvidia_nim/utils/sse_builder.py` | 98% | |
| `providers/nvidia_nim/utils/message_converter.py` | 99% | |
| `utils/text.py` | 100% | |

### Moderate coverage (80–94%)
| Module | Coverage | Key gaps |
|--------|----------|----------|
| `api/app.py` | 89% | Lines 37, 132-138, 151-152, 169-171 |
| `api/command_utils.py` | 92% | Lines 35, 39, 108, 134, 138-139 |
| `api/detection.py` | 91% | Lines 62-63, 111, 121-122 |
| `api/optimization_handlers.py` | 81% | Multiple handler branches |
| `api/request_utils.py` | 94% | Lines 75-77, 94-95 |
| `api/routes.py` | 94% | Lines 88-97 |
| `cli/process_registry.py` | 85% | Lines 35, 43, 72-76 |
| `cli/session.py` | 92% | Session lifecycle edge cases |
| `messaging/handler.py` | 86% | Lines 528-554, 827-834 (large blocks) |
| `messaging/limiter.py` | 85% | Rate limit edge cases |
| `messaging/session.py` | 91% | Lines 220-231, 307-310 |
| `messaging/telegram_markdown.py` | 89% | Lines 245-254, 376-377 |
| `messaging/tree_queue.py` | 94% | |
| `messaging/tree_repository.py` | 96% | |
| `providers/nvidia_nim/client.py` | 76% | Lines 141-160, 199-228, 282-292 |
| `providers/rate_limit.py` | 82% | Lines 180-197 |
| `providers/open_router/request.py` | 90% | |

### Low coverage (<80%)
| Module | Coverage | Notes |
|--------|----------|------|
| **`messaging/telegram.py`** | **69%** | Largest gap: 73 statements uncovered. Lines 273-311, 347-396, 455, 465 |
| **`messaging/transcript.py`** | **78%** | 90 statements uncovered. Lines 316-343, 433-482 |
| **`messaging/base.py`** | **77%** | 12 statements. Abstract/interface methods |
| **`providers/open_router/client.py`** | **43%** | 121 statements uncovered. Major gap |
| **`server.py`** | **30%** | 7 of 10 statements. Entry point / uvicorn run |

## Priority areas for improvement

1. **`providers/open_router/client.py` (43%)** – Large provider with many untested paths.
2. **`server.py` (30%)** – Entry point; currently blocked by missing uvicorn in test env.
3. **`messaging/telegram.py` (69%)** – Telegram integration paths.
4. **`messaging/transcript.py` (78%)** – Transcript handling and edge cases.
5. **`messaging/base.py` (77%)** – Base/interface behavior.

## Test infrastructure

- **Framework:** pytest with pytest-asyncio, pytest-cov
- **Config:** `tests/conftest.py` – shared fixtures (provider_config, nim_provider, mock_cli_session, etc.)
- **CI:** `.github/workflows/tests.yml` – runs `pytest -v --tb=short` (no coverage in CI)
- **Coverage report:** Run `pytest --cov=api --cov=messaging --cov=providers --cov=config --cov=cli --cov=utils --cov=server --cov-report=term-missing --cov-report=html` for HTML report in `htmlcov/`

## Notes

- Two Python 2–style `except` clauses were updated to Python 3 syntax (`except (TypeError, ValueError):`) in `messaging/telegram_markdown.py` and `api/request_utils.py` to allow tests to run.
- `test_server_main_invokes_uvicorn_run` fails when uvicorn is not installed (it is in `[standard]` for FastAPI but may not be in the test environment).
