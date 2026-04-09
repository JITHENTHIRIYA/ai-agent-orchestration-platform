"""
search_tool.py — MCP tool handler: "knowledge_search"

MCP tool contract
-----------------
Name        : knowledge_search
Input       : { query: str, category: str = "general" }
Output      : { results: list[dict], total_found: int }

How metadata filtering works in Pinecone
-----------------------------------------
Every chunk stored in Pinecone carries arbitrary key-value metadata alongside
its embedding vector.  At query time, Pinecone supports a `filter` parameter
that applies a PRE-filter before the ANN (Approximate Nearest Neighbour) search:

  1. Pinecone narrows the candidate set to vectors matching the filter.
  2. Then performs ANN search only within that subset.
  3. Returns up to top_k results from the filtered space.

This differs from a post-hoc filter (fetch everything, then discard) — it is
efficient even on large indexes because the vector search never considers
out-of-category chunks at all.

Filter syntax uses MongoDB-style operators:
  { "category": { "$eq": "legal" } }
  { "source":   { "$in": ["a.pdf", "b.pdf"] } }
  { "score":    { "$gte": 0.8 } }       # numeric metadata field

For this tool, the filter is simple:
  category == "general"  →  no filter (search entire index)
  category == "legal"    →  { "category": { "$eq": "legal" } }

Documents must be indexed with a "category" key in their metadata for this
filter to be selective.  See backend/rag/indexer.py for indexing conventions.

Choosing between knowledge_search and rag_retriever
-----------------------------------------------------
• Use "knowledge_search" when you need a structured list of results, want to
  scope the search to a category, or plan to iterate/filter results yourself.
• Use "rag_retriever" when you need a single formatted context string ready
  for direct injection into an LLM prompt.
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pinecone import Pinecone  # noqa: E402
from config import settings  # noqa: E402
from rag.embeddings import get_embedding  # noqa: E402


DESCRIPTION = (
    "Search the knowledge base for documents matching a query, optionally "
    "filtered by a metadata category (e.g. 'legal', 'medical', 'technical'). "
    "Returns a structured list of matching chunks with similarity scores and "
    "source info.  Use 'general' to search the full index.  "
    "Prefer this over rag_retriever when you need structured results or "
    "category-scoped lookup rather than a prompt-ready context string."
)


def _get_index():
    """Return a Pinecone Index handle using app settings."""
    if not settings.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not configured.")
    index_name = settings.PINECONE_INDEX_NAME or settings.PINECONE_INDEX
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME is not configured.")
    return Pinecone(api_key=settings.PINECONE_API_KEY).Index(index_name)


def _build_filter(category: str) -> dict[str, Any] | None:
    """
    Build a Pinecone metadata filter for the given category string.

    Returns None for "general" so the Pinecone query omits the filter argument
    entirely — an absent filter searches all vectors without restriction.

    For any other value, the $eq operator matches chunks whose stored
    metadata["category"] equals the string exactly (case-sensitive).

    Parameters
    ----------
    category : str

    Returns
    -------
    dict or None
    """
    if category.strip().lower() == "general":
        return None  # No filter — full-index search.
    # Pinecone filter format: { "field": { "$operator": value } }
    return {"category": {"$eq": category}}


def knowledge_search(query: str, category: str = "general") -> dict[str, Any]:
    """
    Perform a category-scoped semantic search over the Pinecone knowledge base.

    MCP contract
    ------------
    Input  : query    (str)         — natural-language search phrase.
             category (str, "general") — metadata category filter; "general"
                                         means no filter (search full index).
    Output : {
               "results":     list[dict],  # matching chunks, structured
               "total_found": int          # count of returned results
             }

    Each dict in results contains:
      text      — chunk text stored in Pinecone metadata
      score     — cosine-similarity ∈ [0, 1]; higher = more relevant
      source    — source file/document identifier from metadata
      category  — category tag from metadata
      metadata  — full raw metadata payload (for downstream filtering)

    How the filter is applied
    -------------------------
    When category != "general", the Pinecone query includes:
      filter={ "category": { "$eq": category } }
    This is evaluated BEFORE the ANN step, so only vectors with matching
    metadata are considered as candidates — not a post-hoc discard.

    Parameters
    ----------
    query : str
        Search phrase to embed and query against.
    category : str
        Metadata category to restrict results to.

    Returns
    -------
    dict
    """
    if not query or not query.strip():
        return {"results": [], "total_found": 0}

    query_embedding = get_embedding(query)
    index = _get_index()

    metadata_filter = _build_filter(category)

    query_kwargs: dict[str, Any] = {
        "vector": query_embedding,
        "top_k": 10,
        "include_metadata": True,
    }
    if metadata_filter is not None:
        query_kwargs["filter"] = metadata_filter

    response = index.query(**query_kwargs)
    matches = getattr(response, "matches", []) or []

    results = [
        {
            "text": (getattr(m, "metadata", {}) or {}).get("text", ""),
            "score": float(getattr(m, "score", 0.0) or 0.0),
            "source": (getattr(m, "metadata", {}) or {}).get("source", "unknown"),
            "category": (getattr(m, "metadata", {}) or {}).get("category", "general"),
            "metadata": getattr(m, "metadata", {}) or {},
        }
        for m in matches
    ]

    return {"results": results, "total_found": len(results)}
