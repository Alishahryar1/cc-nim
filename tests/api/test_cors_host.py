from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import create_app


def test_cors_middleware_wildcard():
    # Test with wildcard origins
    with patch("api.app.get_settings") as mock_settings:
        mock_settings.return_value.cors_origins = ["*"]
        mock_settings.return_value.allowed_hosts = ["*"]
        mock_settings.return_value.log_file = "test.log"
        mock_settings.return_value.log_raw_api_payloads = False

        app = create_app(lifespan_enabled=False)
        client = TestClient(app)

        response = client.options(
            "/",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"


def test_cors_middleware_restricted():
    with patch("api.app.get_settings") as mock_settings:
        mock_settings.return_value.cors_origins = ["http://trusted.com"]
        mock_settings.return_value.allowed_hosts = ["*"]
        mock_settings.return_value.log_file = "test.log"
        mock_settings.return_value.log_raw_api_payloads = False

        app = create_app(lifespan_enabled=False)
        client = TestClient(app)

        # Allowed origin
        response = client.options(
            "/",
            headers={
                "Origin": "http://trusted.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert (
            response.headers.get("access-control-allow-origin") == "http://trusted.com"
        )

        # Denied origin (will not get CORS headers, but still 200/400 depending on route, or CORS blocks it)
        response_bad = client.options(
            "/",
            headers={
                "Origin": "http://malicious.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response_bad.status_code == 400


def test_trusted_host_middleware_wildcard():
    with patch("api.app.get_settings") as mock_settings:
        mock_settings.return_value.cors_origins = ["*"]
        mock_settings.return_value.allowed_hosts = ["*"]
        mock_settings.return_value.log_file = "test.log"
        mock_settings.return_value.log_raw_api_payloads = False

        app = create_app(lifespan_enabled=False)
        client = TestClient(app)

        response = client.get("/", headers={"Host": "anyhost.com"})
        assert response.status_code in (200, 404)


def test_trusted_host_middleware_restricted():
    with patch("api.app.get_settings") as mock_settings:
        mock_settings.return_value.cors_origins = ["*"]
        mock_settings.return_value.allowed_hosts = ["trustedhost.com"]
        mock_settings.return_value.log_file = "test.log"
        mock_settings.return_value.log_raw_api_payloads = False

        app = create_app(lifespan_enabled=False)
        client = TestClient(app)

        response = client.get("/", headers={"Host": "trustedhost.com"})
        assert response.status_code in (200, 404)

        response_bad = client.get("/", headers={"Host": "malicioushost.com"})
        assert response_bad.status_code == 400  # Invalid host header
        assert "Invalid host header" in response_bad.text
