"""
rag_tool.py — MCP tool: "rag_retriever"

What this file does
-------------------
Wraps the existing RAG retrieval pipeline as an MCP-compliant tool so that any
LLM agent using the Model Context Protocol can call it by name.

MCP tool contract
-----------------
Every MCP tool must satisfy three things that the framework enforces at runtime:

1. **Name** — a unique, snake_case string that the agent uses when it calls
   `tools/call`.  Here it is "rag_retriever".

2. **Input schema** — a Pydantic model (or equivalent JSON Schema) that
   describes every parameter the tool accepts, including types and defaults.
   The MCP SDK serialises this to JSON Schema and returns it in `tools/list`
   so the LLM knows what arguments to supply.

3. **Output** — whatever the handler returns is serialised to JSON and sent
   back to the agent as the tool result.  Keeping outputs structured (dicts,
   not raw strings) lets the agent parse and reason over them reliably.

Why typed inputs/outputs matter for agents
------------------------------------------
Unlike a human reading free-form text, an LLM agent plans its next action
based on the structure of a tool's response.  A typed, documented output
(e.g. { context: str, sources: list }) lets the agent:
  - Know exactly which fields are available without guessing.
  - Extract the `context` field to inject into its next prompt.
  - Surface `sources` to the user for citations.
"""

from __future__ import annotations

import os
import sys

# Make sure `backend/` is on the path so sibling packages resolve correctly
# when this module is loaded by the MCP server process.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

from rag.retriever import retrieve, retrieve_as_context  # noqa: E402


# ---------------------------------------------------------------------------
# Input schema
#
# Pydantic is used here for two reasons:
#   1. The MCP SDK reads the model's JSON Schema to advertise parameter types
#      in `tools/list`, giving the LLM a machine-readable spec to follow.
#   2. At call time the SDK validates and coerces the agent's arguments against
#      this schema before passing them to the handler — type errors surface
#      early with a clear message rather than crashing deep in retriever code.
# ---------------------------------------------------------------------------
class RagRetrieverInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Natural-language question or search phrase used to look up "
            "semantically similar chunks in the vector database."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Number of most relevant chunks to retrieve.  Higher values give "
            "broader context but may introduce noise.  Must be between 1-20."
        ),
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def rag_retriever(query: str, top_k: int = 5) -> dict:
    """
    MCP tool handler for "rag_retriever".

    Retrieves the top-k most semantically similar document chunks from the
    Pinecone vector database and returns them in two forms:

      - ``context`` — a single, newline-separated string ready to be injected
        directly into an LLM prompt under a "Context:" header.  The agent
        typically uses this when it needs to answer a user question with RAG.

      - ``sources`` — a structured list of dicts so the agent (or the caller
        downstream) can display citations, filter by source, or do further
        reasoning over individual passages.

    MCP tool contract
    -----------------
    - Tool name   : "rag_retriever"
    - Input model : RagRetrieverInput  →  { query: str, top_k: int = 5 }
    - Output      : { context: str, sources: list[dict] }

    Each dict in ``sources`` contains:
      - text     : the raw chunk text
      - score    : cosine-similarity score in [0, 1]; higher = more relevant
      - metadata : full Pinecone metadata payload (source file, chunk_index, …)

    Parameters
    ----------
    query : str
        The search phrase sent to the embedding model and then to Pinecone.
    top_k : int
        Number of nearest-neighbour chunks to fetch.

    Returns
    -------
    dict
        ``{ "context": str, "sources": list[dict] }``
    """
    # retrieve() returns a list of dicts: [{text, score, metadata}, ...]
    raw_results = retrieve(query=query, top_k=top_k)

    # retrieve_as_context() formats those same results as a numbered
    # passage block — the canonical format for RAG prompt injection.
    context_str = retrieve_as_context(query=query) if raw_results else "No relevant context found."

    # Build a clean sources list; drop the raw embedding vector (not useful
    # to the agent and inflates the JSON payload significantly).
    sources = [
        {
            "text": r["text"],
            "score": r["score"],
            "metadata": r["metadata"],
        }
        for r in raw_results
    ]

    return {"context": context_str, "sources": sources}


# ---------------------------------------------------------------------------
# Tool registration
#
# FastMCP.tool() registers the function under the given name and derives the
# JSON-Schema input spec from the function's type annotations automatically.
# The server must import this module so that the decorator fires at import
# time and adds "rag_retriever" to the server's tool registry.
# ---------------------------------------------------------------------------
def register(mcp: FastMCP) -> None:
    """
    Register the rag_retriever tool on the given FastMCP server instance.

    Call this once during MCP server startup:

        from mcp.tools.rag_tool import register
        register(mcp_server)

    After registration, ``tools/list`` will include "rag_retriever" and
    ``tools/call`` with name="rag_retriever" will invoke the handler above.
    """

    @mcp.tool(name="rag_retriever", description=(
        "Retrieve semantically relevant document chunks from the knowledge base "
        "for a given query.  Returns a formatted context string (for prompt "
        "injection) and a structured list of source passages with similarity scores."
    ))
    def _handler(query: str, top_k: int = 5) -> dict:
        return rag_retriever(query=query, top_k=top_k)
