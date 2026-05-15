import pytest
from fastapi.testclient import TestClient

from api.app import create_app


@pytest.fixture
def client():
    app = create_app(lifespan_enabled=False)
    return TestClient(app)


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


@pytest.mark.parametrize("origin", [
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://[::1]:5173",
    "https://localhost",
])
def test_cors_allowed_origins(client, origin):
    response = client.options("/", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "GET"
    })
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin


@pytest.mark.parametrize("origin", [
    "http://evil.com",
    "https://attacker.net",
    "http://localhost.evil.com",
    "http://127.0.0.1.attacker.com",
])
def test_cors_blocked_origins(client, origin):
    response = client.options("/", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "GET"
    })
    # FastAPI CORSMiddleware generally passes through unauthorized options requests,
    # or responds with 400. In either case it should NOT reflect the origin back
    assert response.headers.get("access-control-allow-origin") != origin
