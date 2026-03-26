"""
retriever.py - Retrieval helpers for Pinecone-backed RAG.

This module converts a user query into an embedding, performs similarity search
in Pinecone, and returns the most relevant chunks for downstream LLM prompting.
"""

from __future__ import annotations

from typing import Any, Dict, List
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from pinecone import Pinecone  # noqa: E402
from config import settings  # noqa: E402
from rag.embeddings import get_embedding  # noqa: E402


def _resolve_index_name() -> str:
    """Resolve Pinecone index name with backwards-compatible fallback."""
    return settings.PINECONE_INDEX_NAME or settings.PINECONE_INDEX


def _get_index():
    """Create and return Pinecone index handle from app settings."""
    if not settings.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing in environment/config.")

    index_name = _resolve_index_name()
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME (or PINECONE_INDEX) is not configured.")

    client = Pinecone(api_key=settings.PINECONE_API_KEY)
    return client.Index(index_name)


def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most similar chunks from Pinecone.

    Parameters
    ----------
    query : str
        The user question or search phrase.
    top_k : int, default=5
        Number of most relevant results to return.
        Higher values provide broader context but can add noise.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries with keys:
          - text: the chunk text stored in metadata
          - score: similarity score from Pinecone
          - metadata: full metadata object attached to that chunk

    Notes on similarity score
    -------------------------
    Pinecone returns a similarity score indicating how close a stored chunk
    embedding is to the query embedding.
    A common intuition is:
      - near 1.0 => highly related / almost identical semantic meaning
      - near 0.0 => weakly related / mostly unrelated
    """
    if not query or not query.strip():
        return []
    if top_k <= 0:
        return []

    query_embedding = get_embedding(query)
    index = _get_index()

    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    matches = getattr(response, "matches", []) or []
    results: List[Dict[str, Any]] = []

    for match in matches:
        metadata = getattr(match, "metadata", {}) or {}
        # Support both object and dict-style responses depending on SDK version.
        score = float(getattr(match, "score", 0.0) or 0.0)
        text = metadata.get("text", "")

        results.append(
            {
                "text": text,
                "score": score,
                "metadata": metadata,
            }
        )

    return results


def retrieve_as_context(query: str) -> str:
    """
    Retrieve relevant chunks and format them as one context string.

    Why format as a single context string?
    --------------------------------------
    LLM prompts are plain text. For RAG, we usually inject retrieved passages
    directly into the prompt under a "Context" section so the model can ground
    its answer in factual snippets from our indexed knowledge base.
    """
    retrieved = retrieve(query=query, top_k=5)
    if not retrieved:
        return "No relevant context found."

    lines: List[str] = []
    for i, item in enumerate(retrieved, start=1):
        source = item["metadata"].get("source", "unknown")
        chunk_index = item["metadata"].get("chunk_index", "n/a")
        lines.append(
            f"[{i}] source={source} chunk={chunk_index} score={item['score']:.4f}\n"
            f"{item['text']}"
        )

    return "\n\n".join(lines)
