# src/rag/retriever.py

# Retriever module for fetching relevant documents for RAG system
# Utilizes FAISS index of sample documents for demonstration

import numpy as np
from src.index.faiss_index import FAISSIndex
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load FAISS components
faiss_index = FAISSIndex()
index = faiss_index.get_index()
model = faiss_index.get_model()
section_map = faiss_index.get_section_map()

def retrieve_docs(ticket: str, top_k: int = 1):
    """
    Retrieve relevant documents based on the input ticket.
    
    Args:
        ticket (str): The input ticket string.
        top_k (int): Number of top relevant documents to retrieve.
        
    Returns:
        List[dict]: List of relevant documents.
    """
    if not isinstance(ticket, str) or not ticket.strip():
        logger.warning("Empty or invalid ticket provided to retrieve_docs.")
        return []
    
    if index is None or model is None:
        logger.warning("FAISS index or model is not initialized.")
        return []
    
    try:
        ticket_emb = model.encode([ticket], convert_to_numpy=True).astype('float32')
    except Exception as e:
        logger.error(f"Error generating embedding for ticket: {e}")
        return []
    
    # Adjust top_k if it exceeds the number of indexed documents
    top_k = min(top_k, index.ntotal)

    distances, indices = index.search(ticket_emb, top_k)

    # Fetch corresponding documents
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        doc = section_map[idx]
        results.append({
            "policy": doc["policy"],
            "section": doc["section"],
            "title": doc["title"],
            "text": doc["text"],
            "distance": float(dist)
        })
    
    return results

# Test usage
if __name__ == "__main__":
    
    test_tickets = [
        "My domain was suspended",
        "Reset password link expired",
        "Refund requested for my order"
    ]

    for ticket in test_tickets:
        print(f"Ticket: {ticket}")
        docs = retrieve_docs(ticket, top_k=1)
        
        for d in docs:
            print(f"Policy: {d['policy']}")
            print(f"Section: {d['section']} - {d['title']}")
            print(f"Distance: {d['distance']:.4f}")
            print("Text:")
            print(d["text"])
            print("\n---")