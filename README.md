# 🧠 AI Coding Challenge: Knowledge Assistant for Support Team

This project implements a minimal **LLM-powered RAG system** that helps a support team respond to customer tickets efficiently using relevant documentation.

---

## Features

- **RAG Pipeline**
  - Retrieves relevant policy and support documentation using FAISS and sentence embeddings.
  - Builds a vector index and fetches top-k relevant sections for each ticket.
  - Automatically handles document ingestion and indexing.

- **LLM Integration**
  - Generates answers with context via local Ollama model (`llama3.2:1b`).
  - Injects retrieved context into prompts to produce MCP-compliant output.
  - Single line modification allows switching to any other locally installed Ollama model.

- **MCP-Compliant Output**
  - Returns structured JSON responses with keys:
    - `answer`
    - `references`
    - `action_required`

- **FastAPI Endpoint**
  - Single POST endpoint: `/resolve-ticket`
  - Input: `{"ticket_text": "..."}`  
  - Output: structured JSON response following MCP.

- **Comprehensive Unit Tests**
  - Document ingestion
  - FAISS index
  - Retrieval module
  - LLM pipeline
  - API layer
  - Full end-to-end system tests
---

## Requirements

- Python 3.11+
- Local Ollama installation with LLaMa model `llama3.2:1b`

## Implementation Notes

Due to time constraints and resource limitations:

- **LLM**: This project uses a lightweight local Ollama model (`llama3.2:1b`) which affects the quality of the chatbot's responses. However, the pipeline is compatible with any locally installed Ollama model with a single line modification.
- **Dockerization**: Docker was omitted in the main branch to simplify local testing and focus on core functionality. The system runs cleanly with Python 3.11+ and standard dependencies.
  - Docker was implemented in a separate branch (feature/docker). However, while this setup is functional, the pipeline performance inside Docker is poor due to the LLM resource restrictions. It is included as a demonstration of containerization capabilities, not an indication of production-quality work. Further optimization was not pursued due to time constraints.

---

## Project Structure

```
├── README.md
├── src/
│   ├── api/           # FastAPI server and endpoints
│   ├── ingest/        # Document ingestion, chunking, and embedding
│   ├── llm/           # LLM client
│   ├── rag/           # RAG pipeline
│   └── index/         # FAISS vector index creation and search utilities
└── data/
    └──raw_docs/      # Sample documents
└── tests/            # Unit tests for all modules and end-to-end pipeline
```

## Example Usage

### 🎯 Sample Input (Support Ticket):
```
My domain was suspended and I didn’t get any notice. How can I reactivate it?
```
### ✅ Expected Output (MCP-compliant JSON):
```json
{
  "answer": "Your domain may have been suspended due to a violation of policy or missing WHOIS information. Please update your WHOIS details and contact support.",
  "references": ["Policy: Domain Suspension Guidelines, Section 4.2"],
  "action_required": "escalate_to_abuse_team"
}
```

## Instructions

### 1. Create a virtual environment (recommended)

Linux/macOS
```
python3 -m venv venv
source venv/bin/activate
```

Windows
```
venv\Scripts\activate      
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Start the API

```
uvicorn src.api.main:app --reload
```

### 4. Test the endpoint
Open `test_ticket.py`
```
import requests
import json

# Modify the ticket text as needed
TICKET_TEXT = "My domain was suspended. How do I reactivate it?"

url = "http://localhost:8000/resolve-ticket"
payload = {
    "ticket_text": TICKET_TEXT
    }

response = requests.post(url, json=payload)
data = response.json()

# Print the JSON response in a compact format
print(json.dumps(data, separators=(',', ':')))
```

Change the text in the TICKET_TEXT field as desired.
Run the script with the code below:
```
python test_ticket.py
```


## Running Tests

This project includes unit tests using `pytest` covering:
- Document ingestion
- FAISS indexing
- LLM pipeline
- API layer
- Full system integration

To run tests:

### 1. Install test dependencies

```
pip install -r requirements.txt
```

### 2. Run tests

To run the entire test suite:

```
pytest -v
```

To run tests for a specific module:

**Ingestion Tests**
```
pytest -v tests/ingest/test_loader.py
```

**Indexing Tests**
```
pytest -v tests/index/test_faiss.py
```

**Retrieval Tests**
```
pytest -v tests/rag/test_retriever.py
```

**LLM Pipeline Tests**
```
pytest -v tests/llm/test_pipeline.py
```

**API Tests**

Requires the FAISS index and metadata to exist locally. Tests will use the TestClient (no server startup required)
```
pytest -v tests/api/test_main.py
```

**End-to-End System Tests**

These tests verify the full flow: `ticket → FAISS retrieval → prompt building → LLM → JSON parsing → final output`

```
pytest -v tests/full_api_test.py
```