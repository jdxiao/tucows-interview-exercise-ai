# src/rag/pipeline.py

# Pipeline module for RAG system
# Integrates retriever and generator components

from src.rag.retriever import retrieve_docs
import subprocess
import json

def build_prompt(query: str, docs: list):
    """
    Build the prompt for the LLM.
    Combines the user query with retrieved documents.
    Generates MCP-formatted prompt.

    Args:
        query (str): The user input query.
        docs (list): List of retrieved documents, each as a dict with 'policy' and 'text'.

    Returns:
        str: The constructed prompt string.
    """

    context = "\n\n".join(
        [f"{doc['policy']}:\nText: {doc['text']}" for doc in docs]
    )

    prompt = f"""
    ROLE:
    You are a knowledge assistant that analyzes customer support tickets 
    and produces structured, actionable responses based on retrieved documentation.

    CONTEXT:
    The following policy documents are provided to assist in answering the query:
    {context}

    TASK: 
    Analyze the user's support ticket.
    Generate an answer using only the provided context.
    If the context does not contain the answer, respond with "Insufficient information to answer the query."

    CONSTRAINTS:
    - Use only the provided context to formulate your response.
    - Provide clear, concise, and actionable information.
    - Format your response in the required output JSON structure.

    OUTPUT FORMAT:
    {{
        "answer": "<short helpful explanation>",
        "references": ["<policy names or sections used>"],
        "action_required": "<a concise, descriptive action like 'escalate_to_abuse_team' or 'none' if no action is needed>"
    }}

    USER TICKET:
    {query}
    """

    return prompt.strip()


def call_llm(prompt: str, model: str = "mistral"):
    """
    Call Ollama model locally to generate a response based on the prompt.

    Args:
        prompt (str): The input prompt for the LLM.
        model (str): The Ollama model to use.
    """
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    return result.stdout.decode().strip()


def generate_response(query: str, top_k: int = 1):
    """
    Minimal RAG pipeline to:
    1. Retrieve relevant documents based on the query.
    2. Build a prompt for the LLM.
    3. Call the LLM with the constructed prompt.
    4. Return the LLM's structured response.

    Args:
        query (str): The user input query.
        top_k (int): Number of top relevant documents to retrieve.
    """

    docs = retrieve_docs(query, top_k=top_k)
    prompt = build_prompt(query, docs)
    response = call_llm(prompt)
    
    try:
        response_json = json.loads(response)
    except json.JSONDecodeError:
        response_json = {
            "answer": "Error: Unable to parse LLM response.",
            "references": [],
            "action_required": "none"
        }

    return response_json

#  Test usage
if __name__ == "__main__":
    test_queries = [
        "My domain was suspended due to missing WHOIS info. How can I reactivate it?",
        "I need to reset my password but the link expired. What should I do?",
        "I requested a refund for my canceled order. When will I receive it?"
    ]

    for query in test_queries:
        print(f"Query: {query}")
        print(generate_response(query))