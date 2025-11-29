# Unit tests for document loader

import pytest
from src.ingest.loader import load_policies
from pathlib import Path

def test_load_policies():
    sections = load_policies("./data/raw_docs")
    assert isinstance(sections, list)
    assert len(sections) > 0

def test_section_structure():
    sections = load_policies("./data/raw_docs")
    for section in sections:
        assert "policy" in section
        assert "section" in section
        assert "title" in section
        assert "text" in section
        assert isinstance(section["text"], str)
        assert len(section["text"]) > 0