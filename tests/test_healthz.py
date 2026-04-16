from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_healthz_returns_ok():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_process_requires_internal_key(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_KEY", "test-key")
    response = client.post("/process")
    assert response.status_code == 401


def test_process_accepts_valid_key(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_KEY", "test-key")
    response = client.post("/process", headers={"x-internal-key": "test-key"})
    assert response.status_code == 501
