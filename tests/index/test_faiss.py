# Unit tests for FAISS integration

import pytest
from src.index.faiss_index import get_index, get_section_map, get_model

def test_faiss_index_creation():
    index = get_index()
    assert index is not None
    assert index.ntotal > 0

def test_section_map_consistency():
    index = get_index()
    section_map = get_section_map()
    assert len(section_map) == index.ntotal

def test_embedding_dimensions():
    model = get_model()
    assert model is not None
    sample_text = "This is a test document."
    embedding = model.encode([sample_text], convert_to_numpy=True)
    assert embedding.shape[0] == 1
    assert embedding.shape[1] == get_index().d