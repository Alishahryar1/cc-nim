import pytest
from fastapi.testclient import TestClient

from api.app import create_app


@pytest.fixture
def client():
    app = create_app(lifespan_enabled=False)
    return TestClient(app, base_url="http://127.0.0.1:50000")


def test_security_headers(client):
    response = client.get("/")  # Any route works

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert response.headers["X-Download-Options"] == "noopen"
    assert response.headers["X-Permitted-Cross-Domain-Policies"] == "none"
    assert response.headers["X-DNS-Prefetch-Control"] == "off"
    assert "max-age=63072000" in response.headers["Strict-Transport-Security"]
    assert response.headers["Cross-Origin-Opener-Policy"] == "same-origin"
    assert response.headers["Cross-Origin-Resource-Policy"] == "same-origin"
    assert response.headers["Cross-Origin-Embedder-Policy"] == "require-corp"
    assert "accelerometer=()" in response.headers["Permissions-Policy"]

    assert "Content-Security-Policy" in response.headers
    csp = response.headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "form-action 'self'" in csp
    assert "base-uri 'self'" in csp
