"""
main.py — FastAPI application entry point.

Creates the FastAPI app instance and registers top-level routes.
Run with:
    uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from agents.base_agent import run_agent
from agents.rag_agent import run_rag_agent
from rag.indexer import index_documents_from_folder
from rag.retriever import retrieve
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


class RAGIndexRequest(BaseModel):
    """Incoming JSON body for indexing local documents into Pinecone."""
    folder_path: str


class RAGIndexResponse(BaseModel):
    """Outgoing JSON body for index trigger results."""
    status: str
    message: str


class RAGQueryRequest(BaseModel):
    """Incoming JSON body for RAG agent query endpoint."""
    query: str


class RAGQueryResponse(BaseModel):
    """Outgoing JSON body returned by the RAG query endpoint."""
    query: str
    response: str
    sources: list[str]
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


# -----------------------------------------------------------------------
# POST /api/rag/index
# -----------------------------------------------------------------------
# Purpose:
#   Trigger batch indexing of local .txt documents into Pinecone.
#
# Request body:
#   { "folder_path": "data/sample_docs" }
#
# Response body:
#   { "status": "indexed", "message": "X documents indexed" }
# -----------------------------------------------------------------------
@app.post("/api/rag/index", response_model=RAGIndexResponse)
async def rag_index(request: RAGIndexRequest):
    """
    Index `.txt` documents from a local folder into the vector database.

    Error handling:
    - Returns HTTP 400 for invalid folder/input errors.
    - Returns HTTP 500 for unexpected ingestion failures.
    """
    try:
        summary = index_documents_from_folder(request.folder_path)
        indexed_docs = summary.get("total_files_indexed", 0)
        return RAGIndexResponse(
            status="indexed",
            message=f"{indexed_docs} documents indexed",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG indexing failed: {str(e)}")


# -----------------------------------------------------------------------
# POST /api/rag/query
# -----------------------------------------------------------------------
# Purpose:
#   Query the RAG-enabled agent which uses retrieval + LLM reasoning.
#
# Request body:
#   { "query": "string" }
#
# Response body:
#   { "query", "response", "sources", "status" }
# -----------------------------------------------------------------------
@app.post("/api/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Run a query through the RAG agent and return grounded response metadata.

    Error handling:
    - Converts agent-level failures into HTTP 500.
    - Also guards against malformed internal return payloads.
    """
    try:
        result = run_rag_agent(request.query)
        return RAGQueryResponse(
            query=result.get("query", request.query),
            response=result.get("response", ""),
            sources=result.get("sources", []),
            status=result.get("status", "error"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


# -----------------------------------------------------------------------
# GET /api/rag/retrieve
# -----------------------------------------------------------------------
# Purpose:
#   Debug endpoint to inspect raw retrieval results from Pinecone without
#   running the full agent. Useful for tuning chunking and relevance quality.
#
# Query params:
#   ?query=string&top_k=5
# -----------------------------------------------------------------------
@app.get("/api/rag/retrieve")
async def rag_retrieve(query: str = Query(...), top_k: int = Query(5, ge=1, le=50)):
    """
    Return raw top-k retrieved chunks for a given query.

    Error handling:
    - Returns HTTP 400 for invalid retrieval arguments.
    - Returns HTTP 500 for Pinecone/embedding/runtime failures.
    """
    try:
        return retrieve(query=query, top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG retrieval failed: {str(e)}")
