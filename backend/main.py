"""
main.py — FastAPI application entry point.

Creates the FastAPI app instance and registers top-level routes.
Run with:
    uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config import settings
from agents.base_agent import run_agent
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AntiGravity Backend",
    description="FastAPI backend powering the AntiGravity AI agent platform.",
    version="0.1.0",
)


@app.on_event("startup")
async def log_environment():
    """Log environment configuration on startup to verify keys are loaded."""
    logger.info("🚀 AntiGravity Backend starting up...")
    logger.info(
        "GROQ_API_KEY: %s",
        "✅ set" if settings.GROQ_API_KEY else "❌ missing",
    )
    logger.info(
        "PINECONE_API_KEY: %s",
        "✅ set" if settings.PINECONE_API_KEY else "❌ missing",
    )
    logger.info(
        "PINECONE_INDEX: %s",
        settings.PINECONE_INDEX or "❌ missing",
    )


# ─── Request / Response schemas ─────────────────────────────────────────

class AgentQueryRequest(BaseModel):
    """Incoming JSON body for the agent query endpoint."""
    query: str


class AgentQueryResponse(BaseModel):
    """Outgoing JSON body returned by the agent query endpoint."""
    query: str
    response: str
    status: str


# ─── Routes ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """
    Health-check endpoint.

    Returns a simple JSON payload confirming the service is running.
    Useful for uptime monitors, container orchestrators, and load balancers.
    """
    return {"status": "ok"}


# -----------------------------------------------------------------------
# POST /api/agent/query
# -----------------------------------------------------------------------
# This endpoint is the primary interface for clients to interact with the
# ReAct agent.  It:
#   1. Receives a JSON body containing a natural-language "query" string.
#   2. Passes the query to the base ReAct agent (agents/base_agent.py),
#      which runs the Thought → Action → Observation loop via Groq/LLaMA.
#   3. Returns the agent's final answer along with the original query and
#      a "status" field ("success" or "error").
#   4. On failure, responds with HTTP 500 and an error detail message.
# -----------------------------------------------------------------------

@app.post("/api/agent/query", response_model=AgentQueryResponse)
async def agent_query(request: AgentQueryRequest):
    """
    Run a user query through the ReAct agent and return the result.

    **Request body** – ``{ "query": "your question here" }``

    **Response body** – ``{ "query": "...", "response": "...", "status": "success" }``

    Raises HTTP 500 if the agent encounters an error during execution.
    """
    try:
        result = run_agent(request.query)
        return AgentQueryResponse(
            query=request.query,
            response=result,
            status="success",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}",
        )
