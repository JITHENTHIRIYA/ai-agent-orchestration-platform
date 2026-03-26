"""
indexer.py - Document indexing pipeline for Pinecone.

Responsibilities
----------------
1) Connect to Pinecone using application settings.
2) Ensure the target index exists (dimension=384, metric=cosine).
3) Split incoming text into retrieval-friendly chunks.
4) Generate embeddings for each chunk.
5) Upsert vectors + metadata into Pinecone.
6) Provide folder-based ingestion for local `.txt` knowledge files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from langchain.text_splitter import RecursiveCharacterTextSplitter  # noqa: E402
from pinecone import Pinecone, ServerlessSpec  # noqa: E402
from config import settings  # noqa: E402
from rag.embeddings import get_embeddings_batch  # noqa: E402


def _resolve_index_name() -> str:
    """Resolve Pinecone index name from settings with backwards compatibility."""
    return settings.PINECONE_INDEX_NAME or settings.PINECONE_INDEX


def _get_pinecone_client() -> Pinecone:
    """Build Pinecone client from configured API key."""
    if not settings.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing in environment/config.")
    return Pinecone(api_key=settings.PINECONE_API_KEY)


def _get_index(client: Pinecone):
    """Return handle to configured Pinecone index."""
    index_name = _resolve_index_name()
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME (or PINECONE_INDEX) is not configured.")
    return client.Index(index_name)


def create_index_if_not_exists() -> str:
    """
    Ensure Pinecone index exists and return its name.

    Index defaults:
    - Dimension: settings.PINECONE_DIMENSION (typically 384)
    - Metric: cosine
    """
    client = _get_pinecone_client()
    index_name = _resolve_index_name()
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME (or PINECONE_INDEX) is not configured.")

    existing_indexes = client.list_indexes()
    existing_names = existing_indexes.names() if hasattr(existing_indexes, "names") else existing_indexes

    if index_name not in existing_names:
        # Serverless defaults are intentionally explicit to avoid environment
        # drift; values can be adjusted later if multi-region needs emerge.
        client.create_index(
            name=index_name,
            dimension=settings.PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return index_name


def _split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text into chunks suitable for vector retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def index_text(
    text: str,
    source_name: str,
    namespace: str = "",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> int:
    """
    Index a single raw text payload in Pinecone.

    Returns
    -------
    int
        Number of chunks/vectors written.
    """
    if not text or not text.strip():
        return 0

    create_index_if_not_exists()
    chunks = _split_text(text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return 0

    embeddings = get_embeddings_batch(chunks)
    index = _get_index(_get_pinecone_client())

    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{source_name}-{i}-{uuid.uuid4().hex[:8]}"
        metadata: Dict[str, Any] = {
            "source": source_name,
            "chunk_index": i,
            "text": chunk,
        }
        vectors.append({"id": vector_id, "values": embedding, "metadata": metadata})

    index.upsert(vectors=vectors, namespace=namespace)
    return len(vectors)


def index_documents_from_folder(
    folder_path: str,
    namespace: str = "",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Dict[str, int]:
    """
    Index all `.txt` files from a local folder into Pinecone.

    Returns a summary dictionary with per-file counts and total vectors.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder path: {folder_path}")

    summary: Dict[str, int] = {"total_vectors": 0, "total_files_indexed": 0}
    for txt_file in sorted(folder.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8")
        written = index_text(
            text=text,
            source_name=txt_file.name,
            namespace=namespace,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        summary[txt_file.name] = written
        summary["total_vectors"] += written
        if written > 0:
            summary["total_files_indexed"] += 1

    return summary


def _smoke_test() -> None:
    """Small runnable verification for chunking and Pinecone connectivity."""
    sample_text = (
        "AntiGravity builds an orchestration platform for AI agents. " * 30
    )
    chunks = _split_text(sample_text, chunk_size=500, chunk_overlap=50)
    print(f"Chunking test -> chunks produced: {len(chunks)}")
    print(f"First chunk preview: {chunks[0][:80]}...")

    try:
        index_name = create_index_if_not_exists()
        print(f"Pinecone connection/index check -> OK (index: {index_name})")
    except Exception as exc:
        print(f"Pinecone connection/index check -> FAILED: {exc}")
        return

    if os.getenv("RUN_INDEXER_SMOKE_UPSERT", "0") == "1":
        upserted = index_text(sample_text, source_name="smoke_test.txt")
        print(f"Optional upsert check -> vectors upserted: {upserted}")
    else:
        print("Skipping upsert check (set RUN_INDEXER_SMOKE_UPSERT=1 to enable).")


if __name__ == "__main__":
    _smoke_test()
