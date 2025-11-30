# Unit tests for FAISS integration

import pytest
from src.index.faiss_index import FAISSIndex
import json

def test_faiss_index_creation(tmp_path):
    # Create valid policy JSON for testing
    policy_data = {
        "policy": "Test Policy",
        "sections": [
            {"section": "1.1", "title": "Section 1", "text": "This is the text of section 1."},
            {"section": "1.2", "title": "Section 2", "text": "This is the text of section 2."}
        ]
    }

    file_path = tmp_path/"test_policy.json"
    with open(file_path, "w") as f:
        json.dump(policy_data, f)
    
    index_instance = FAISSIndex(policy_dir=str(tmp_path))
    index = index_instance.get_index()
    section_map = index_instance.get_section_map()
    model = index_instance.get_model()

    # Basic assertions
    assert index is not None
    assert index.ntotal == 2
    assert isinstance(section_map, dict)
    assert index.ntotal == len(section_map)
    assert len(section_map) == 2
    assert model is not None

def test_empty_directory(tmp_path):
    index_instance = FAISSIndex(policy_dir=str(tmp_path))
    assert index_instance.get_index() is None
    assert index_instance.get_section_map() == {}
    assert index_instance.get_model() is None

def test_missing_text_field(tmp_path):
    policy_data = {
        "policy": "Partial Policy",
        "sections": [
            {"section": "1.1", "title": "Title"} # Missing 'text' field
        ]
    }

    file_path = tmp_path/"partial_policy.json"
    with open(file_path, "w") as f:
        json.dump(policy_data, f)

    index_instance = FAISSIndex(policy_dir=str(tmp_path))
    assert index_instance.get_index() is None
    assert index_instance.get_section_map() == {}
    assert index_instance.get_model() is None

def test_expected_types(tmp_path):
    index_instance = FAISSIndex(policy_dir=str(tmp_path))

    assert isinstance(index_instance.get_section_map(), dict)
    assert index_instance.get_index() is None or hasattr(index_instance.get_index(), 'search')
    model = index_instance.get_model()
    assert model is None or hasattr(model, 'encode')

def test_embedding_dimensions(tmp_path):
    policy_data = {
        "policy": "Dimensional Policy",
        "sections": [
            {"section": "1.1", "title": "Section 1", "text": "Text for section 1."}
        ]
    }

    file_path = tmp_path/"dim_policy.json"
    with open(file_path, "w") as f:
        json.dump(policy_data, f)

    index_instance = FAISSIndex(policy_dir=str(tmp_path))
    model = index_instance.get_model()
    index = index_instance.get_index()

    sample_text = "Sample text for embedding."
    embedding = model.encode([sample_text], convert_to_numpy=True).astype('float32')

    assert embedding.shape[0] == 1
    assert embedding.shape[1] == index.d