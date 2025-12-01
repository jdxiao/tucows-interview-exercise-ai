# Unit tests for retriever module

import pytest
from src.rag.retriever import retrieve_docs

def test_retrieve_docs_return_structure():
    ticket = "Test query for document retrieval"
    results = retrieve_docs(ticket, top_k=1)
    
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
        assert isinstance(doc["distance"], float)

def test_retrieve_docs_top_k():
    ticket = "Another test query"
    top_k = 3
    results = retrieve_docs(ticket, top_k=top_k)
    
    assert isinstance(results, list)
    assert len(results) <= top_k

def test_retrieve_docs_returns_list():
    ticket = "refund"
    results = retrieve_docs(ticket, top_k=2)
    assert isinstance(results, list)
    assert all("policy" in result and "text" in result for result in results)

def test_retrieve_docs_empty_ticket():
    ticket = ""
    results = retrieve_docs(ticket, top_k=1)
    assert results == []

def test_retrieve_docs_none_ticket():
    ticket = None
    results = retrieve_docs(ticket, top_k=1)
    assert results == []

def test_retrieve_docs_top_k_exceeds_index_size():
    ticket = "Test query"
    top_k = 1000  # Assuming index has fewer than 1000 documents
    results = retrieve_docs(ticket, top_k=top_k)
    from src.index.faiss_index import FAISSIndex
    faiss_index = FAISSIndex()
    max_docs = faiss_index.get_index().ntotal
    assert len(results) <= max_docs

def test_retrieve_docs_distance_valid():
    ticket = "Sample query"
    results = retrieve_docs(ticket, top_k=2)
    for doc in results:
        assert doc["distance"] >= 0.0