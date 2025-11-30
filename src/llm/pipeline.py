# src/llm/pipeline.py

# Pipeline module for RAG system
# Integrates retriever and generator components

from src.rag.retriever import retrieve_docs
import subprocess
import json
import regex as re

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
        [f"{doc['policy']} — Section {doc['section']} ({doc['title']}):\n{doc['text']}" for doc in docs]
    )

    prompt = f"""
    ROLE:
    You are a knowledge assistant that analyzes customer support tickets 
    and produces structured, actionable responses based on retrieved documentation.

    CONTEXT:
    The following policy documents are provided to assist in answering the ticket:
    {context}

    TASK: 
    1. Analyze the user ticket.
    2. Analyze the provided policy documents.
    3. Generate a concise answer to the user's ticket based on the documents.
    4. Determine which policy sections were referenced.
    5. Assign an appropriate action required based on the analysis in the format action_required_by_policy.
    6. Output the response strictly in the specified JSON format.

    EXAMPLES:

    Ticket: "My domain was suspended and I didn’t get any notice. How can I reactivate it?"
    Output:
    {{
        "answer": "Your domain may have been suspended due to a violation of policy or missing WHOIS information. Please update your WHOIS details and contact support.",
        "references": ["Policy: Domain Suspension Guidelines, Section 4.2"],
        "action_required": "escalate_to_abuse_team"
    }}

    CONSTRAINTS:
    - Provide answers strictly based on the provided documents.
    - Output must be a single JSON object with keys: answer, references, action_required.
    - Do not include any explanations outside the JSON format.
    - Do not include any formatting or markdown in the output.
    - The output schema is defined below.

    OUTPUT SCHEMA:
    {{
        "answer": "<short helpful explanation>",
        "references": ["<policy name - section title>"],
        "action_required": "<a concise, descriptive action like 'escalate_to_abuse_team' based on the analysis.>"
    }}

    USER TICKET:
    {query}

    FINAL INSTRUCTION:
    Respond with ONLY the JSON. Do not say anything else.
    """

    return prompt.strip()


def call_llm(prompt: str, model: str = "llama3.2:1b"):
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


def extract_json(response: str):
    """
    Extract JSON object from LLM response string.

    Args:
        response (str): The raw response string from the LLM.

    Returns:
        dict: The extracted JSON object.
    """
    matches = re.findall(r'\{(?:[^{}]|(?R))*\}', response, re.DOTALL)

    if not matches:
        return {
            "answer": "Error: Unable to parse LLM response.",
            "references": [],
            "action_required": "none"
        }

    # Take final JSON object found in event of multiple reasoning steps
    final_json = matches[-1]

    try:
        return json.loads(final_json)
    except json.JSONDecodeError:
        return {
            "answer": "Error: Unable to parse LLM response.",
            "references": [],
            "action_required": "none"
        }


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
    
    # Extract JSON from LLM response
    response_json = extract_json(response)

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
        print(f"Response: {generate_response(query, top_k=1)}\n")