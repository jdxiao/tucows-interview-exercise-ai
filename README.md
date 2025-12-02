# 🧠 AI Coding Challenge: Knowledge Assistant for Support Team

**Note:** This branch was created specifically to demonstrate Docker containerization of the API and RAG pipeline.
The main branch contains the fully functional system for local Python environments, which is recommended for testing and review.
The Docker setup works and loads the LLM model, but due to resource constraints, response quality are significantly lower than the main branch and optimization was not pursued due to time constraints. This branch exists to showcase containerization capability, not a production-quality system.

---

## Instructions

### 1. Build and start the service

```
docker-compose up --build
```

- This builds the image and starts the API in a container.
- The API is accessible at: http://localhost:8000/resolve-ticket
Model is loaded from /app/models/TinyLlama-1.1B-32k-f16.gguf inside the container.

### 2. Run Python inside the container

```
docker exec -it <container_name> python
```

Run the following Python commands to test the pipeline (will likely result in an error due to incorrect LLM response format).

```
import requests
import json

TICKET_TEXT = "My domain was suspended. How do I reactivate it?" # Modify as required

url = "http://localhost:8000/resolve-ticket"
payload = {"ticket_text": TICKET_TEXT}

response = requests.post(url, json=payload)
data = response.json()

# Print exact JSON
print(json.dumps(data, separators=(',', ':')))
```

**Note:** for proof of LLM functionality, instead run:

```
from src.llm.pipeline import call_llm
call_llm("your_prompt_here")
```