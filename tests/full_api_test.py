# tests/api/full_api_test.py

# Full API integration tests

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_resolve_ticket_single_ticket():
    ticket = {"ticket_text": "My domain was suspended and I didn’t get any notice."}
    response = client.post("/resolve-ticket", json=ticket)
    assert response.status_code == 200

    data = response.json()
    
    # MCP structure assertions
    assert "answer" in data
    assert "references" in data
    assert "action_required" in data

    # Validate types
    assert isinstance(data["answer"], str)
    assert isinstance(data["references"], list)
    assert all(isinstance(r, str) for r in data["references"])
    assert isinstance(data["action_required"], str)

def test_resolve_ticket_multiple_tickets():
    tickets = [
        {"ticket_text": "My domain was suspended."},
        {"ticket_text": "Reset password link expired."},
        {"ticket_text": "Refund requested for my order."}
    ]

    for ticket in tickets:
        response = client.post("/resolve-ticket", json=ticket)
        assert response.status_code == 200
        data = response.json()
        # Check keys exist
        assert "answer" in data
        assert "references" in data
        assert "action_required" in data

def test_resolve_ticket_invalid_payload():
    invalid_ticket = {"invalid_key": "Test"}
    response = client.post("/resolve-ticket", json=invalid_ticket)
    assert response.status_code == 422  # FastAPI validation error
