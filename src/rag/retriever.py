# src/rag/retriever.py

# Retriever module for fetching relevant documents for RAG system
# Utilizes sample index for initial testing and development

from src.index.sample_index import get_all_docs

def retrieve_docs(query: str, top_k: int = 1):
    """
    Retrieve relevant documents based on the input query.
    
    Args:
        query (str): The input query string (currently not used in basic implementation).
        top_k (int): Number of top relevant documents to retrieve.
        
    Returns:
        List[dict]: List of relevant documents.
    """
    all_docs = get_all_docs()
    
    # Simple keyword matching for basic pipeline purposes
    relevant_docs = []
    for doc in all_docs:
        if any(keyword.lower() in doc["text"].lower() for keyword in query.split()):
            if doc not in relevant_docs:
                relevant_docs.append(doc)
        if len(relevant_docs) >= top_k:
            break
    
    # Return top_k documents
    return relevant_docs[:top_k]


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
            print(f"{d['policy']}\nText: {d['text']}\n")