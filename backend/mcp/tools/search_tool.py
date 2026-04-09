"""
search_tool.py — MCP tool: "knowledge_search"

What this file does
-------------------
Provides category-filtered semantic search over the Pinecone knowledge base as
an MCP tool.  Unlike "rag_retriever" (which retrieves and formats context for
prompt injection), this tool is designed for structured lookup — the agent
gets back a list of result dicts it can iterate, filter, or display as-is.

How metadata filtering works in Pinecone
-----------------------------------------
Every document chunk stored in Pinecone can carry arbitrary key-value metadata
alongside its embedding vector.  At query time, Pinecone supports a ``filter``
parameter that applies a pre-filter (before ANN search) to restrict the
candidate set to vectors whose metadata matches the given predicates.

Example Pinecone filter syntax (passed as a dict):

    { "category": { "$eq": "legal" } }
    { "source": { "$in": ["doc_a.pdf", "doc_b.pdf"] } }
    { "score": { "$gte": 0.8 } }

Operators supported: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or.

In this tool:
  - If category == "general" (the default), NO filter is applied and the
    search spans the entire index — behaviour identical to a plain retrieve().
  - If any other category string is supplied, the filter
    ``{ "category": { "$eq": category } }`` is added to the Pinecone query,
    restricting results to chunks whose metadata["category"] matches exactly.

This means that when indexing documents you should store a "category" key in
the metadata dict for this filter to be selective:

    index.upsert(vectors=[{
        "id": "chunk-001",
        "values": embedding,
        "metadata": { "text": "...", "source": "law_review.pdf", "category": "legal" }
    }])

MCP tool contract
-----------------
- Tool name   : "knowledge_search"
- Input model : KnowledgeSearchInput → { query: str, category: str = "general" }
- Output      : { results: list[dict], total_found: int }
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from pinecone import Pinecone  # noqa: E402

from config import settings  # noqa: E402
from rag.embeddings import get_embedding  # noqa: E402


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------
class KnowledgeSearchInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "Natural-language search phrase.  Converted to an embedding and "
            "used for nearest-neighbour search in the vector database."
        ),
    )
    category: str = Field(
        default="general",
        description=(
            "Metadata category to restrict search to.  Use 'general' (default) "
            "to search the entire index.  Any other value applies a Pinecone "
            "metadata filter { category: $eq <value> }, returning only chunks "
            "whose stored metadata contains that category key."
        ),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_index():
    """Return a Pinecone Index handle using app settings."""
    if not settings.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not configured.")
    index_name = settings.PINECONE_INDEX_NAME or settings.PINECONE_INDEX
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME is not configured.")
    client = Pinecone(api_key=settings.PINECONE_API_KEY)
    return client.Index(index_name)


def _build_filter(category: str) -> Dict[str, Any] | None:
    """
    Build a Pinecone metadata filter dict for the given category.

    Pinecone metadata filters use MongoDB-style query operators.  The ``$eq``
    operator matches chunks whose metadata[key] equals the supplied value
    exactly (case-sensitive string comparison).

    Returns None when category is "general" so the caller can omit the filter
    argument entirely — an explicit None filter returns all candidates.

    Parameters
    ----------
    category : str
        The category string from the tool input.

    Returns
    -------
    dict or None
        ``{ "category": { "$eq": category } }`` or ``None``.
    """
    if category.strip().lower() == "general":
        # No filter — search the whole index.
        return None
    # Pinecone filter syntax: { "field": { "operator": value } }
    return {"category": {"$eq": category}}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def knowledge_search(query: str, category: str = "general") -> dict:
    """
    MCP tool handler for "knowledge_search".

    Performs a category-scoped semantic search over the Pinecone index and
    returns a structured list of matching document chunks.

    This tool differs from "rag_retriever" in two important ways:
      1. **Structured output** — results are a list of dicts, not a formatted
         context string.  The agent can iterate, filter, sort, or display them.
      2. **Category filter** — the search can be scoped to a metadata category,
         which is useful when the knowledge base stores documents from multiple
         domains (e.g. "legal", "medical", "technical") and the agent needs to
         restrict the search space based on user intent or routing logic.

    MCP tool contract
    -----------------
    - Tool name : "knowledge_search"
    - Input     : { query: str, category: str = "general" }
    - Output    : { results: list[dict], total_found: int }

    Each dict in ``results`` contains:
      - text      : chunk text stored in Pinecone metadata
      - score     : cosine-similarity score ∈ [0, 1]
      - source    : source file/document identifier from metadata
      - category  : category tag from metadata (may be absent if not indexed)
      - metadata  : full raw metadata payload for the chunk

    Parameters
    ----------
    query : str
        The search phrase to embed and query against.
    category : str
        Metadata category filter.  "general" means no filter.

    Returns
    -------
    dict
        ``{ "results": list[dict], "total_found": int }``

    Raises
    ------
    ValueError
        If Pinecone credentials or index name are missing from config.
    """
    if not query or not query.strip():
        return {"results": [], "total_found": 0}

    query_embedding = get_embedding(query)
    index = _get_index()

    # Build the optional metadata filter.  When None, Pinecone searches all
    # vectors without restriction.  When provided, only vectors whose stored
    # metadata matches the filter predicate are considered as candidates.
    metadata_filter = _build_filter(category)

    query_kwargs: Dict[str, Any] = {
        "vector": query_embedding,
        "top_k": 10,
        "include_metadata": True,
    }
    if metadata_filter is not None:
        # Pinecone evaluates this filter BEFORE the ANN step, so it reduces
        # the search space first and then picks the top_k nearest neighbours
        # from the matching subset — not a post-hoc filter on global results.
        query_kwargs["filter"] = metadata_filter

    response = index.query(**query_kwargs)
    matches = getattr(response, "matches", []) or []

    results: List[Dict[str, Any]] = []
    for match in matches:
        metadata: Dict[str, Any] = getattr(match, "metadata", {}) or {}
        score = float(getattr(match, "score", 0.0) or 0.0)
        results.append(
            {
                "text": metadata.get("text", ""),
                "score": score,
                "source": metadata.get("source", "unknown"),
                # Expose category from metadata so the agent can verify the
                # filter was respected or surface it to the user as a label.
                "category": metadata.get("category", "general"),
                "metadata": metadata,
            }
        )

    return {"results": results, "total_found": len(results)}


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register(mcp: FastMCP) -> None:
    """
    Register the knowledge_search tool on the given FastMCP server instance.

    Call this once during MCP server startup:

        from mcp.tools.search_tool import register
        register(mcp_server)

    After registration, ``tools/list`` will include "knowledge_search" and
    ``tools/call`` with name="knowledge_search" will invoke the handler above.

    The agent uses the description below (from `tools/list`) to decide between
    "knowledge_search" and "rag_retriever":
      - Use "knowledge_search" when you need a structured list of results or
        need to scope the search to a specific document category.
      - Use "rag_retriever" when you need a formatted context string ready for
        direct prompt injection.
    """

    @mcp.tool(name="knowledge_search", description=(
        "Search the knowledge base for documents matching a query, optionally "
        "filtered by a metadata category (e.g. 'legal', 'medical', 'technical'). "
        "Returns a structured list of matching chunks with similarity scores and "
        "source information.  Use 'general' as category to search the full index. "
        "Prefer this over rag_retriever when you need structured results or "
        "category-scoped lookup rather than a formatted context string."
    ))
    def _handler(query: str, category: str = "general") -> dict:
        return knowledge_search(query=query, category=category)
