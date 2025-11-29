# Unit tests for main API module

import pytest
from src.api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_resolve_ticket_endpoint():
    payload = {"ticket": "What is the refund policy?"}
    response = client.post("/resolve-ticket", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "answer" in data
    assert "references" in data
    assert "action_required" in data

def test_resolve_ticket_invalid_payload():
    payload = {"invalid_key": "Test"}
    response = client.post("/resolve-ticket", json=payload)
    assert response.status_code == 422