# src/rag/retriever.py

# Retriever module for fetching relevant documents for RAG system
# Utilizes sample index for initial testing and development

import numpy as np
from src.index.faiss_index import FAISSIndex

# Load FAISS components
faiss_index = FAISSIndex()
index = faiss_index.get_index()
model = faiss_index.get_model()
section_map = faiss_index.get_section_map()

def retrieve_docs(query: str, top_k: int = 1):
    """
    Retrieve relevant documents based on the input query.
    
    Args:
        query (str): The input query string.
        top_k (int): Number of top relevant documents to retrieve.
        
    Returns:
        List[dict]: List of relevant documents.
    """
    
    
    # Generate embedding for the query
    query_emb = model.encode([query], convert_to_numpy=True).astype('float32')

    # Search FAISS index for nearest neighbors
    distances, indices = index.search(query_emb, top_k)

    # Fetch corresponding document sections
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
    
    test_queries = [
        "My domain was suspended",
        "Reset password link expired",
        "Refund requested for my order"
    ]

    for query in test_queries:
        print(f"Query: {query}")
        docs = retrieve_docs(query, top_k=1)
        
        for d in docs:
            print(f"Policy: {d['policy']}")
            print(f"Section: {d['section']} - {d['title']}")
            print(f"Distance: {d['distance']:.4f}")
            print("Text:")
            print(d["text"])
            print("\n---")