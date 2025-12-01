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
def resolve_ticket(request: TicketRequest):
    """
    Endpoint to handle user queries and return structured responses.

    Args:
        request (TicketRequest): The incoming request containing the user ticket.
    Returns:
        TicketResponse: The structured response from the LLM.
    """
    if not request.ticket_text or not request.ticket_text.strip():
        return TicketResponse(
            answer="Error: Empty ticket provided.",
            references=[],
            action_required="none"
        )
    try:
        response = generate_response(request.ticket_text, top_k=1)
    except Exception as e:
        response = {
            "answer": f"Error processing the ticket: {e}",
            "references": [],
            "action_required": "none"
        }

    if not all(k in response for k in ("answer", "references", "action_required")):
        response = {
            "answer": "Error: Incomplete response from LLM.",
            "references": [],
            "action_required": "none"
        }
    
    return response

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}