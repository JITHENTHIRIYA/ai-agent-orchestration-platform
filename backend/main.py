"""
main.py — FastAPI application entry point.

Creates the FastAPI app instance and registers top-level routes.
Run with:
    uvicorn main:app --reload
"""

import asyncio
import logging
import os
import subprocess
import sys
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from agents.base_agent import run_agent
from agents.rag_agent import run_rag_agent
from agents.mcp_agent import (
    list_available_tools,
    run_mcp_agent,
    _async_call_tool,   # used by the direct-invoke endpoint (async-native)
)
from config import settings
from rag.indexer import index_documents_from_folder
from rag.retriever import retrieve

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AntiGravity Backend",
    description="FastAPI backend powering the AntiGravity AI agent platform.",
    version="0.1.0",
)


# Reference to the MCP server subprocess started at FastAPI startup.
# Kept at module level so the shutdown handler can terminate it cleanly.
_mcp_process: Optional[asyncio.subprocess.Process] = None


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


@app.on_event("startup")
async def start_mcp_server():
    """
    Launch the MCP server as a background subprocess on port 8001.

    Why a subprocess rather than an in-process thread?
    ---------------------------------------------------
    The MCP server runs its own uvicorn/anyio event loop via FastMCP.run().
    Embedding that inside FastAPI's event loop would cause loop-within-loop
    conflicts.  A subprocess gives it a completely isolated Python process
    and event loop, matching production deployments where the two services
    are independent containers.

    How it fits into the startup sequence:
      1. FastAPI starts (uvicorn binds port 8000).
      2. This handler fires → spawns `python mcp/server.py` as a child process.
      3. The child process binds port 8001 and begins accepting SSE connections.
      4. Incoming requests to /api/mcp/* will reach the server once it is ready
         (the endpoints handle ConnectionRefusedError gracefully).

    The process handle is stored in _mcp_process so the shutdown handler can
    terminate it cleanly when uvicorn exits.
    """
    global _mcp_process

    # Resolve the path to server.py relative to this file so the command
    # works regardless of the working directory uvicorn is started from.
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(backend_dir, "mcp", "server.py")

    try:
        # asyncio.create_subprocess_exec is the async-native way to spawn a
        # child process from within an already-running event loop.
        # stdout/stderr are piped so the MCP server's log lines don't
        # interleave with FastAPI's own output in the terminal.
        _mcp_process = await asyncio.create_subprocess_exec(
            sys.executable,
            server_script,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Run from backend/ so the server's relative imports resolve.
            cwd=backend_dir,
        )
        logger.info(
            "MCP Server started on port %d (pid=%d)",
            settings.MCP_SERVER_PORT,
            _mcp_process.pid,
        )
    except Exception as exc:
        # Non-fatal: FastAPI continues running even if MCP startup fails.
        # /api/mcp/* endpoints will return 503 until the server is available.
        logger.error("Failed to start MCP server: %s", exc)


@app.on_event("shutdown")
async def stop_mcp_server():
    """Terminate the MCP subprocess when FastAPI shuts down."""
    if _mcp_process and _mcp_process.returncode is None:
        _mcp_process.terminate()
        try:
            await asyncio.wait_for(_mcp_process.wait(), timeout=5.0)
            logger.info("MCP server subprocess terminated cleanly.")
        except asyncio.TimeoutError:
            _mcp_process.kill()
            logger.warning("MCP server subprocess killed after timeout.")


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


# ═══════════════════════════════════════════════════════════════════════════
# MCP endpoints
#
# These three endpoints expose the MCP server layer to external clients.
# Unlike the /api/rag/* endpoints (which call tool logic directly inside
# FastAPI), the /api/mcp/* endpoints talk to the separate MCP server process
# running on port 8001 — a clean separation of concerns:
#
#   Frontend / client
#       │
#       ▼
#   FastAPI (port 8000)   ← these endpoints live here
#       │  JSON-RPC over SSE
#       ▼
#   MCP Server (port 8001) ← tool logic lives here
#       │
#       ▼
#   Pinecone / Groq / ...
#
# Error handling strategy across all three endpoints:
#   • ConnectionRefusedError / OSError → 503 Service Unavailable
#     (MCP server not running or still starting up)
#   • ValueError (bad input) → 400 Bad Request
#   • Unexpected exceptions → 500 Internal Server Error
# ═══════════════════════════════════════════════════════════════════════════


# ─── MCP schemas ────────────────────────────────────────────────────────

class MCPQueryRequest(BaseModel):
    """Request body for the MCP agent query endpoint."""
    query: str


class MCPQueryResponse(BaseModel):
    """
    Response body for the MCP agent query endpoint.

    Fields mirror the dict returned by run_mcp_agent() so the API surface
    is predictable and typed for frontend consumers.
    """
    query: str
    response: str
    tool_used: Optional[str]     # None when the LLM answered without a tool
    tool_input: Optional[dict]   # None when no tool was called
    status: str                  # "success" | "error"


class MCPToolsResponse(BaseModel):
    """Response body listing all tools registered on the MCP server."""
    tools: list[dict]


class MCPToolInvokeRequest(BaseModel):
    """
    Request body for direct (agent-bypassing) MCP tool invocation.

    tool_name must exactly match a registered MCP tool name.
    inputs must satisfy that tool's JSON Schema (validated server-side).
    """
    tool_name: str
    inputs: dict[str, Any]


class MCPToolInvokeResponse(BaseModel):
    """Response body for a direct MCP tool invocation."""
    tool_name: str
    result: str    # Raw string result returned by the MCP server
    status: str    # "success" | "error"


# ─── MCP routes ─────────────────────────────────────────────────────────

# -----------------------------------------------------------------------
# POST /api/mcp/query
# -----------------------------------------------------------------------
# Runs the full MCP-backed ReAct agent for a user query.
#
# Flow:
#   1. Validate request body (Pydantic).
#   2. run_mcp_agent() builds/reuses the LangChain AgentExecutor:
#        a. Fetches tool list from MCP server via tools/list (cached).
#        b. Creates LangChain Tool stubs wrapping each MCP tool.
#        c. Runs Thought → Action → Observe loop via Groq llama-3.3-70b.
#        d. Each Action triggers an MCP tools/call to port 8001.
#   3. Returns the final answer plus which tool was last called (if any).
#
# If the LLM decides no tool is needed (general-knowledge query), it emits
# "Final Answer" immediately — no MCP round-trip occurs.
# -----------------------------------------------------------------------
@app.post(
    "/api/mcp/query",
    response_model=MCPQueryResponse,
    summary="Run a query through the MCP-backed ReAct agent",
)
async def mcp_query(request: MCPQueryRequest):
    """
    Send a query to the MCP agent which uses Model Context Protocol tools.

    The agent connects to the MCP server (port 8001) to discover and call
    available tools (rag_retriever, doc_summarizer, knowledge_search).

    **Request body** — ``{ "query": "your question" }``

    **Response body** — ``{ query, response, tool_used, tool_input, status }``

    - ``tool_used``: name of the MCP tool called last, or null if the LLM
      answered without any tool.
    - ``tool_input``: argument dict passed to that tool, or null.

    Returns HTTP 503 if the MCP server is not reachable.
    Returns HTTP 500 for unexpected agent failures.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        # run_mcp_agent is synchronous (wraps async MCP calls via _run_async).
        # We run it in the default executor so it doesn't block the event loop.
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_mcp_agent, request.query)

        return MCPQueryResponse(
            query=result.get("query", request.query),
            response=result.get("response", ""),
            tool_used=result.get("tool_used"),
            tool_input=result.get("tool_input"),
            status=result.get("status", "error"),
        )

    except (ConnectionRefusedError, OSError) as exc:
        # MCP server not running or still starting — return a clear 503 so
        # callers can retry rather than treating it as a permanent failure.
        raise HTTPException(
            status_code=503,
            detail=(
                f"MCP server at {settings.MCP_SERVER_HOST}:{settings.MCP_SERVER_PORT} "
                f"is not reachable.  Ensure it has finished starting up.  ({exc})"
            ),
        )
    except Exception as exc:
        logger.exception("MCP agent query failed for query=%r", request.query)
        raise HTTPException(status_code=500, detail=f"MCP agent failed: {exc}")


# -----------------------------------------------------------------------
# GET /api/mcp/tools
# -----------------------------------------------------------------------
# Queries the MCP server's tools/list endpoint and returns the full
# catalogue of registered tools with their names, descriptions, and
# JSON-Schema input specs.
#
# Useful for:
#   • Frontend UIs that want to display available capabilities.
#   • Health / admin dashboards confirming tool registration.
#   • Testing that the MCP server started correctly and all tools loaded.
# -----------------------------------------------------------------------
@app.get(
    "/api/mcp/tools",
    response_model=MCPToolsResponse,
    summary="List all tools registered on the MCP server",
)
async def mcp_list_tools():
    """
    Retrieve the catalogue of tools available on the MCP server.

    Sends a ``tools/list`` JSON-RPC request to the MCP server (port 8001)
    and returns the result.  Each tool entry contains:

    - ``name``        — unique tool identifier (e.g. "rag_retriever")
    - ``description`` — natural-language description shown to the LLM
    - ``inputSchema`` — JSON Schema of the tool's accepted arguments

    Returns HTTP 503 if the MCP server is not reachable.
    Returns HTTP 500 for unexpected errors.
    """
    try:
        # list_available_tools() is synchronous — run in executor to avoid
        # blocking FastAPI's event loop during the SSE handshake.
        loop = asyncio.get_event_loop()
        tools = await loop.run_in_executor(None, list_available_tools)
        return MCPToolsResponse(tools=tools)

    except (ConnectionRefusedError, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"MCP server at {settings.MCP_SERVER_HOST}:{settings.MCP_SERVER_PORT} "
                f"is not reachable.  ({exc})"
            ),
        )
    except Exception as exc:
        logger.exception("Failed to list MCP tools")
        raise HTTPException(status_code=500, detail=f"Failed to list MCP tools: {exc}")


# -----------------------------------------------------------------------
# POST /api/mcp/tool/invoke
# -----------------------------------------------------------------------
# Directly invokes a named MCP tool, bypassing the ReAct agent entirely.
#
# Why bypass the agent?
#   • During development you often want to test a single tool in isolation
#     without the overhead of the full LLM reasoning loop.
#   • Integration tests can assert exact tool outputs without worrying about
#     how the LLM decides to call (or not call) the tool.
#   • Allows fine-grained debugging of tool input/output contracts.
#
# The request goes directly to the MCP server via a tools/call JSON-RPC
# request; the MCP server validates inputs and executes the handler.
#
# Request body:
#   { "tool_name": "rag_retriever", "inputs": { "query": "...", "top_k": 5 } }
#
# Response body:
#   { "tool_name": "rag_retriever", "result": "<raw string>", "status": "success" }
# -----------------------------------------------------------------------
@app.post(
    "/api/mcp/tool/invoke",
    response_model=MCPToolInvokeResponse,
    summary="Directly invoke a specific MCP tool (bypasses the agent)",
)
async def mcp_tool_invoke(request: MCPToolInvokeRequest):
    """
    Invoke a named MCP tool directly, bypassing the ReAct agent.

    Sends a ``tools/call`` JSON-RPC request to the MCP server with the
    provided tool name and input arguments.  The MCP server validates the
    arguments against the tool's JSON Schema, executes the handler, and
    returns the raw result.

    **Request body**::

        {
          "tool_name": "rag_retriever",
          "inputs": { "query": "what is machine learning?", "top_k": 3 }
        }

    **Response body**::

        { "tool_name": "rag_retriever", "result": "<json string>", "status": "success" }

    ``result`` is the raw string content returned by the MCP server.
    For structured tools (rag_retriever, knowledge_search) this will be
    a JSON-encoded dict; for doc_summarizer it will be plain text.

    Returns HTTP 400 if tool_name is empty.
    Returns HTTP 503 if the MCP server is not reachable.
    Returns HTTP 500 for unexpected tool or server errors.
    """
    if not request.tool_name.strip():
        raise HTTPException(status_code=400, detail="tool_name must not be empty.")

    try:
        # _async_call_tool is an async coroutine — we can await it directly
        # inside this async endpoint handler without any executor gymnastics.
        # This is the one place in the codebase where the MCP client is called
        # natively async (the agent uses _run_async because LangChain tools
        # are sync; here FastAPI is already async so we call it directly).
        raw_result = await _async_call_tool(
            tool_name=request.tool_name,
            arguments=request.inputs,
        )

        # Surface MCP-level tool errors as HTTP 500 with the error text.
        # (MCP tool errors arrive as a successful HTTP response with an error
        # payload — we translate them to HTTP 500 for REST API consistency.)
        if raw_result.startswith("[MCP tool error]"):
            raise HTTPException(status_code=500, detail=raw_result)

        return MCPToolInvokeResponse(
            tool_name=request.tool_name,
            result=raw_result,
            status="success",
        )

    except HTTPException:
        raise  # Re-raise HTTP errors unchanged.

    except (ConnectionRefusedError, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"MCP server at {settings.MCP_SERVER_HOST}:{settings.MCP_SERVER_PORT} "
                f"is not reachable.  ({exc})"
            ),
        )
    except Exception as exc:
        logger.exception(
            "Direct MCP tool invocation failed: tool=%r inputs=%r",
            request.tool_name,
            request.inputs,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Tool invocation failed: {exc}",
        )
