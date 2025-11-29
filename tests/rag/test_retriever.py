# Unit tests for retriever module

import pytest
from src.rag.retriever import retrieve_docs

def test_retrieve_docs_return_structure():
    query = "Test query for document retrieval"
    results = retrieve_docs(query, top_k=1)
    
    assert isinstance(results, list)
    assert len(results) <= 2
    
    for doc in results:
        assert "policy" in doc
        assert "section" in doc
        assert "title" in doc
        assert "text" in doc
        assert "distance" in doc
        assert isinstance(doc["text"], str)
        assert len(doc["text"]) > 0

def test_retrieve_docs_top_k():
    query = "Another test query"
    top_k = 3
    results = retrieve_docs(query, top_k=top_k)
    
    assert isinstance(results, list)
    assert len(results) <= top_k

def test_retrieve_docs_returns_list():
    results = retrieve_docs("refund", top_k=2)
    assert isinstance(results, list)
    assert all("policy" in result and "text" in result for result in results)