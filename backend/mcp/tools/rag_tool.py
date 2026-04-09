"""
rag_tool.py — MCP tool handler: "rag_retriever"

MCP tool contract
-----------------
Name        : rag_retriever
Input       : { query: str, top_k: int = 5 }
Output      : { context: str, sources: list[dict] }

Why typed inputs/outputs matter for MCP agents
-----------------------------------------------
The MCP SDK derives a JSON Schema from each tool handler's type annotations
and includes it in the `tools/list` response.  The LLM reads this schema to
know exactly which arguments to supply — no guessing, no free-form parsing.

On the output side, returning a structured dict (not a raw string) lets the
agent pull specific fields:
  • `context`  — injected verbatim into the LLM prompt under a "Context:"
                  header for retrieval-augmented generation.
  • `sources`  — surfaced to the user as citations, or passed to another tool
                  (e.g. "doc_summarizer") in a multi-step pipeline.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Add backend/ to sys.path so sibling packages (rag, config) resolve
# when this module is imported by the MCP server process.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rag.retriever import retrieve, retrieve_as_context  # noqa: E402


# ---------------------------------------------------------------------------
# Tool description — shown to the LLM in tools/list so it can decide when to
# call this tool versus "knowledge_search" or "doc_summarizer".
# ---------------------------------------------------------------------------
DESCRIPTION = (
    "Retrieve semantically relevant document chunks from the knowledge base "
    "for a given natural-language query. "
    "Returns a formatted context string ready for prompt injection, plus a "
    "structured list of source passages with similarity scores and metadata. "
    "Use this when you need to ground an answer in indexed documents."
)


def rag_retriever(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Retrieve the top-k most relevant document chunks from Pinecone and return
    them in two forms optimised for different downstream uses.

    MCP contract
    ------------
    Input  : query (str) — natural-language question or search phrase.
             top_k (int, default 5) — number of chunks to retrieve (1–20).
    Output : {
               "context": str,          # numbered passage block for RAG prompts
               "sources": list[dict]    # structured list for citations / chaining
             }

    Each dict in sources contains:
      text     — raw chunk text
      score    — cosine-similarity ∈ [0, 1]; higher = more semantically similar
      metadata — full Pinecone metadata (source file, chunk_index, category, …)

    Parameters
    ----------
    query : str
        Search phrase forwarded to the embedding model and then Pinecone ANN.
    top_k : int
        Number of nearest-neighbour chunks to fetch.  Clamped to [1, 20] by
        the tool registry input validation.

    Returns
    -------
    dict
    """
    raw = retrieve(query=query, top_k=top_k)

    # retrieve_as_context() formats passages as a numbered block:
    #   [1] source=doc.pdf chunk=3 score=0.9231
    #   <chunk text>
    # This is the canonical format for injecting into an LLM system prompt.
    context = retrieve_as_context(query=query) if raw else "No relevant context found."

    sources = [
        {"text": r["text"], "score": r["score"], "metadata": r["metadata"]}
        for r in raw
    ]

    return {"context": context, "sources": sources}
