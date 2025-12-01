# Unit tests for main API module

import pytest
from src.api.main import app
from fastapi.testclient import TestClient
from unittest.mock import patch

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_resolve_ticket_endpoint():
    payload = {"ticket_text": "What is the refund policy?"}
    response = client.post("/resolve-ticket", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Basic structure assertions
    assert "answer" in data
    assert "references" in data
    assert "action_required" in data

def test_resolve_ticket_empty_ticket():
    payload = {"ticket_text": ""}
    response = client.post("/resolve-ticket", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"].startswith("Error")
    assert data["references"] == []
    assert data["action_required"] == "none"

def test_resolve_ticket_invalid_payload():
    payload = {"invalid_key": "Test"}
    response = client.post("/resolve-ticket", json=payload)
    assert response.status_code == 422

def test_resolve_ticket_llm_exception():
    payload = {"ticket_text": "What is the refund policy?"}
    
    with patch('src.api.main.generate_response') as mock_generate:
        mock_generate.side_effect = Exception("LLM failure")
        response = client.post("/resolve-ticket", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["answer"].startswith("Error")
        assert data["references"] == []
        assert data["action_required"] == "none"