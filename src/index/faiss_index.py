# src/index/faiss_index.py

# FAISS index module for RAG system
# Implements vector-based document retrieval using FAISS
# Uses sample documents for demonstration

import faiss
from src.ingest.loader import load_policies
from sentence_transformers import SentenceTransformer
import numpy as np

# Load sample policy documents
sections = load_policies("./data/raw_docs")

# Initialize sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for each document section
texts = [section["text"] for section in sections]
embeddings = model.encode(texts, convert_to_numpy=True).astype('float32')

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Map FAISS indices to document sections
section_map = {i: sections[i] for i in range(len(sections))}

# Get functions
def get_index():
    """
    Returns the FAISS index for document retrieval.
    """
    return index

def get_section_map():
    """
    Returns the mapping of FAISS indices to document sections.
    """
    return section_map

def get_model():
    """
    Returns the sentence transformer model used for embeddings.
    """
    return model

# Test usage
if __name__ == "__main__":
    print(f"Index contains {index.ntotal} documents.")