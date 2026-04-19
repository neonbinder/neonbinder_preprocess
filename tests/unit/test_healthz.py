from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthz_returns_ok():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthz_does_not_require_internal_key():
    # /healthz must stay unauthenticated so Cloud Run health probes pass.
    response = client.get("/healthz")
    assert response.status_code == 200
