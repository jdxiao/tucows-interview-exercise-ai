# Unit tests for pipeline module

import pytest
from src.llm.pipeline import build_prompt, generate_response, extract_json
from unittest.mock import patch

def test_build_prompt_structure():
    docs = [
        {"policy": "Policy A", "section": "1.1", "title": "Title A", "text": "Sample text."}]
    prompt = build_prompt("What is the refund policy?", docs)
    assert "ROLE" in prompt
    assert "CONTEXT" in prompt
    assert "TASK" in prompt
    assert "OUTPUT SCHEMA" in prompt
    assert "Sample text." in prompt
    assert "Policy A" in prompt

def test_extract_json():
    llm_response = """
    Here is the information you requested:
    {
        "answer": "You can request a refund within 30 days of purchase.",
        "references": ["Policy: Refund Policy, Section 3.1"],
        "action_required": "process_refund_request"
    }
    """
    extracted = extract_json(llm_response)
    assert extracted["answer"] == "You can request a refund within 30 days of purchase."
    assert extracted["references"] == ["Policy: Refund Policy, Section 3.1"]
    assert extracted["action_required"] == "process_refund_request"

def test_extract_json_no_json():
    llm_response = "I'm sorry, I cannot provide that information."
    extracted = extract_json(llm_response)
    assert extracted["answer"].startswith("Error")
    assert extracted["references"] == []
    assert extracted["action_required"] == "none"

def test_generate_response_structure():
    response = generate_response("What is the refund policy?", top_k=1)
    assert "answer" in response
    assert "references" in response
    assert "action_required" in response

def test_generate_response_with_retriever():
    sample_docs = [
        {"policy": "Policy B", "section": "2.1", "title": "Title B", "text": "Refunds are processed within 5 business days."}
    ]
    mock_llm_response = '{"answer": "Refunds are processed within 5 business days.", "references": ["Policy B"], "action_required": "initiate_refund"}'

    with patch('src.llm.pipeline.retrieve_docs', return_value=sample_docs):
        with patch('src.llm.pipeline.call_llm', return_value=mock_llm_response):
            response = generate_response("How long does a refund take?", top_k=1)
            assert response["answer"] == "Refunds are processed within 5 business days."
            assert response["references"] == ["Policy B"]
            assert response["action_required"] == "initiate_refund"

def test_generate_response_empty_ticket():
    response = generate_response("", top_k=1)
    assert response["answer"].startswith("Error")
    assert response["references"] == []
    assert response["action_required"] == "none"

def test_generate_response_no_retrieved_docs():
    with patch('src.llm.pipeline.retrieve_docs', return_value=[]):
        response = generate_response("How to reactivate my suspended domain?", top_k=1)
        assert response["answer"].startswith("No relevant documents")
        assert response["references"] == []
        assert response["action_required"] == "none"

def test_generate_response_bad_llm_response():
    sample_docs = [
        {"policy": "Policy C", "section": "3.1", "title": "Title C", "text": "Sample text for testing."}
    ]
    bad_llm_response = "This is not a JSON response."

    with patch('src.llm.pipeline.retrieve_docs', return_value=sample_docs):
        with patch('src.llm.pipeline.call_llm', return_value=bad_llm_response):
            response = generate_response("Test query?", top_k=1)
            assert response["answer"].startswith("Error")
            assert response["references"] == []
            assert response["action_required"] == "none"