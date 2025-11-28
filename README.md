# 🧠 AI Coding Challenge: Knowledge Assistant for Support Team

This project implements a minimal **LLM-powered RAG system** that helps a support team respond to customer tickets efficiently using relevant documentation.

---

## Project Structure

```
├── README.md
├── src/
│   ├── api/           # FastAPI server and endpoints
│   ├── ingest/        # Document ingestion, chunking, and embedding
│   ├── llm/           # LLM client
│   ├── rag/           # RAG pipeline
│   ├── index/         # FAISS vector index creation and search utilities
│   └── utils/         # Configuration, logging, and helper functions
└── data/
    ├── raw_docs/      # Original documents
    ├── index.faiss    # FAISS vector index
    └── metadata.json  # Mapping of chunk IDs to source documents
```

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

## 🔧 Requirements
### 1.  RAG Pipeline
- Embed sample support docs and policy FAQs (provided or synthetic).
- Use a vector database (e.g., FAISS, Qdrant, etc.) to retrieve context based on the query.

### 2.  LLM Integration
- Use a language model (e.g., OpenAI GPT, LLaMA2 via Ollama, Mistral, etc.)
- Inject context and query into the prompt to generate the final answer.

### 3.  MCP (Model Context Protocol)
- Prompt should have clearly defined role, context, task, and output schema.
- Output must be valid JSON in the following format:
  ```json
  {
    "answer": "...",
    "references": [...],
    "action_required": "..."
  }
  ```
### 4.  API Endpoint
- Expose a single endpoint: POST /resolve-ticket
- Input: { "ticket_text": "..." }
- Output: structured JSON response as shown above

## 📂 Suggested Tech Stack (Use what you're comfortable with)
- Languages: Python or Go
- Embedding Models: Sentence Transformers / OpenAI / HuggingFace
- Vector Store: FAISS, Qdrant, Weaviate, etc.
- LLMs: OpenAI, Ollama, Local LLM, or APIs
- API: FastAPI (Python), Gin/Fiber (Go)
- Docker Compose