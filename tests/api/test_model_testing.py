import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from tests.api.support import create_test_app
from free_claude_code.api.model_testing import testing_manager

app = create_test_app()

def test_model_testing_routes_success():
    client = TestClient(app, client=("127.0.0.1", 50000))
    
    # Reset state to be clean before test
    testing_manager.is_running = False
    testing_manager.total_models = 0
    testing_manager.tested_count = 0
    testing_manager.results = {}
    if hasattr(testing_manager, "_current_task") and testing_manager._current_task:
        testing_manager._current_task.cancel()
        testing_manager._current_task = None

    # 1. Get models list
    response = client.get("/admin/api/test-models/models")
    assert response.status_code == 200
    data = response.json()
    assert "grouped" in data

    # 2. Run model tests with mocked start_test
    with patch.object(testing_manager, "start_test", return_value=True) as mock_start:
        run_response = client.post(
            "/admin/api/test-models/run",
            json={"models": ["deepseek/deepseek-chat"]}
        )
        assert run_response.status_code == 200
        run_data = run_response.json()
        assert run_data["status"] == "started"
        assert run_data["total"] == 1
        mock_start.assert_called_once_with(["deepseek/deepseek-chat"], "", 8082)

    # Test the already_running scenario (returns 409 conflict when start_test returns False)
    with patch.object(testing_manager, "start_test", return_value=False) as mock_start_running:
        run_response = client.post(
            "/admin/api/test-models/run",
            json={"models": ["deepseek/deepseek-chat"]}
        )
        assert run_response.status_code == 409
        run_data = run_response.json()
        assert run_data["detail"]["status"] == "already_running"

    # 3. Check status
    status_response = client.get("/admin/api/test-models/status")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert "is_running" in status_data
    assert "results" in status_data

    # Cleanup after test to prevent leakage
    testing_manager.is_running = False
    testing_manager.total_models = 0
    testing_manager.tested_count = 0
    testing_manager.results = {}
