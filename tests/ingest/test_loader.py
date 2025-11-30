# Unit tests for document loader

import pytest
from src.ingest.loader import load_policies
from pathlib import Path
import json

# Helper function to create test policy files
def create_policy_file(directory, filename, data):
    file_path = Path(directory) / filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return file_path

# Test loading basic policy files
def test_load_policies_basic(tmp_path):
    data = {
        "policy": "Test Policy",
        "description": "Description",
        "sections": [
            {"section": "1.1", "title": "Section 1", "text": "Text 1"},
            {"section": "1.2", "title": "Section 2", "text": "Text 2"},
        ]
    }
    create_policy_file(tmp_path, "policy.json", data)
    sections = load_policies(str(tmp_path))
    assert isinstance(sections, list)
    assert len(sections) == 2

# Test structure of loaded sections
def test_section_structure_basic(tmp_path):
    data = {
        "policy": "Test Policy",
        "sections": [{"section": "1.1", "title": "Title", "text": "Some text"}]
    }
    create_policy_file(tmp_path, "policy.json", data)
    sections = load_policies(str(tmp_path))
    section = sections[0]
    for key in ["policy", "section", "title", "text"]:
        assert key in section
        assert isinstance(section[key], str)
    assert len(section["text"]) > 0

# Edge case tests
def test_missing_sections_key(tmp_path):
    create_policy_file(tmp_path, "no_sections.json", {"policy": "No Sections"})
    sections = load_policies(str(tmp_path))
    assert sections == []

def test_malformed_json(tmp_path, caplog):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{ invalid json ")
    
    with caplog.at_level("WARNING"):
        sections = load_policies(str(tmp_path))
    assert sections == [] # Malformed file skipped
    assert any("Skipping invalid JSON file" in record.message for record in caplog.records)

def test_missing_text_field(tmp_path):
    data = {"policy": "Partial Policy", "sections": [{"section": "1.1", "title": "No text"}]}
    create_policy_file(tmp_path, "partial.json", data)
    sections = load_policies(str(tmp_path))
    assert sections == []  # Section skipped

def test_missing_policy_field(tmp_path):
    data = {"sections": [{"section": "1.1", "title": "Title", "text": "Some text"}]}
    create_policy_file(tmp_path, "nopolicy.json", data)
    sections = load_policies(str(tmp_path))
    assert sections[0]["policy"] == "Unknown Policy"

def test_empty_directory(tmp_path):
    sections = load_policies(str(tmp_path))
    assert sections == [] # No files to load

# Test loading policies from the raw_docs directory
def test_load_policies():
    sections = load_policies("./data/raw_docs")
    assert isinstance(sections, list)
    assert len(sections) > 0

# Test structure of loaded sections from raw_docs
def test_section_structure():
    sections = load_policies("./data/raw_docs")
    for section in sections:
        assert "policy" in section
        assert "section" in section
        assert "title" in section
        assert "text" in section
        assert isinstance(section["text"], str)
        assert len(section["text"]) > 0