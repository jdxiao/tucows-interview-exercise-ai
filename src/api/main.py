# src/api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from src.llm.pipeline import generate_response

app = FastAPI(title="RAG Knowledge Assistant")

# Request body model
class TicketRequest(BaseModel):
    ticket_text: str

# Response model
class TicketResponse(BaseModel):
    answer: str
    references: list[str]
    action_required: str

@app.post("/resolve-ticket", response_model=TicketResponse)
def resolve_query(request: TicketRequest):
    """
    Endpoint to handle user queries and return structured responses.

    Args:
        request (TicketRequest): The incoming request containing the user ticket.
    Returns:
        TicketResponse: The structured response from the LLM.
    """
    response = generate_response(request.ticket_text, top_k=1)
    return response

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}