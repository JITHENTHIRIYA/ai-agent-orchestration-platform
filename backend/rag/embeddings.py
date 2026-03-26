"""
embeddings.py — Text embedding generation using HuggingFace sentence-transformers.

What is an Embedding?
─────────────────────
An embedding is a numerical representation of text (a list of floats).
It captures the semantic meaning of the text. Words, sentences, or documents
with similar meanings will have embeddings that are mathematically "closer"
to each other in the vector space. This allows us to perform similarity searches
to find relevant information for Retrieval-Augmented Generation (RAG).

Why 384 dimensions?
───────────────────
The dimensionality of an embedding refers to the number of float values in the vector.
We use 384 dimensions because it strikes an excellent balance between precision
and performance. Models producing 384-d vectors are fast to run locally, require
far less memory/storage in the vector database (Pinecone), and still deliver
high-quality search results for standard text retrieval tasks.

Why all-MiniLM-L6-v2?
─────────────────────
We use the HuggingFace `sentence-transformers/all-MiniLM-L6-v2` model because:
1. Lightweight — it has a tiny computation footprint and runs quickly on cheap CPUs.
2. Free — it runs entirely locally within our backend, meaning zero API costs.
3. Good Quality — despite its size, it was trained on 1 billion+ training pairs
   and excels at semantic search and clustering.
"""

from typing import List
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from sentence_transformers import SentenceTransformer  # noqa: E402
from config import settings  # noqa: E402

# ── Initialize Model ────────────────────────────────────────────────────────
# Load the embedding model specified in our configuration (defaults to
# "sentence-transformers/all-MiniLM-L6-v2").
# We instantiate it at the module level so it loads only once at startup
# and stays in memory for fast inference on subsequent calls.
model = SentenceTransformer(settings.EMBEDDING_MODEL)


def get_embedding(text: str) -> List[float]:
    """
    Generate a 384-dimensional vector embedding for a single text string.

    Parameters
    ----------
    text : str
        The input text to embed.

    Returns
    -------
    List[float]
        A mathematically representative 384-dimensional float list.
    """
    # model.encode() returns a NumPy array. Pinecone requires standard Python
    # float lists, so we convert it via .tolist()
    embedding = model.encode(text)
    return embedding.tolist()


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate vector embeddings for an entire batch of text strings.

    Batch processing is significantly faster than looping over get_embedding
    because the underlying transformer applies vectorized PyTorch operations
    to process the texts in parallel.

    Parameters
    ----------
    texts : List[str]
        A list of input text strings to embed.

    Returns
    -------
    List[List[float]]
        A list containing the 384-dimensional float embeddings for each string.
    """
    embeddings = model.encode(texts)
    return embeddings.tolist()
