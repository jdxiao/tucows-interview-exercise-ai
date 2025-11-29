# Unit tests for pipeline module

import pytest
from src.llm.pipeline import build_prompt, generate_response

def test_build_prompt_structure():
    docs = [
        {"policy": "Policy A", "section": "1.1", "title": "Title A", "text": "Sample text."}]
    prompt = build_prompt("What is the refund policy?", docs)
    assert "ROLE" in prompt
    assert "CONTEXT" in prompt
    assert "TASK" in prompt
    assert "OUTPUT SCHEMA" in prompt

def test_generate_response_structure():
    response = generate_response("What is the refund policy?", top_k=1)
    assert "answer" in response
    assert "references" in response
    assert "action_required" in response