"""Persistent ChromaDB vector store client with swappable embedding providers."""

import os
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions
from ..ingestion.chunker import DocumentChunk
from ..config import get_settings


@lru_cache(maxsize=1)
def get_embedding_function():
    """Factory for embedding functions supporting sentence-transformers, Gemini, or mock."""
    settings = get_settings()
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "sentence-transformers":
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL
        )
    elif provider == "gemini" and settings.GEMINI_API_KEY:
        return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=settings.GEMINI_API_KEY,
            model_name="models/embedding-001",
        )
    else:
        # Default to sentence-transformers
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_MODEL
        )


_shared_vector_store = None


def get_shared_vector_store() -> "ChromaVectorStore":
    """Return shared ChromaVectorStore singleton to prevent memory leaks and redundant DB locks."""
    global _shared_vector_store
    if _shared_vector_store is None:
        _shared_vector_store = ChromaVectorStore()
    return _shared_vector_store


class ChromaVectorStore:
    """Persistent ChromaDB vector store wrapper for document chunks."""

    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "grounded_rag"):
        settings = get_settings()
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = get_embedding_function()
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add or update document chunks in the Chroma collection."""
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "source": str(chunk.source),
                "page": int(chunk.page),
                "chunk_id": str(chunk.chunk_id),
                "chunk_index": int(chunk.chunk_index),
                "section": str(chunk.section),
                "char_count": int(chunk.char_count),
                "token_count": int(chunk.token_count),
            }
            for chunk in chunks
        ]

        # Chroma upsert ensures idempotency
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Dense semantic search returning (DocumentChunk, similarity_score)."""
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(k, count)
        where_clause = metadata_filter if metadata_filter else None

        results = self.collection.query(
            query_texts=[query],
            n_results=actual_k,
            where=where_clause,
        )

        output: List[Tuple[DocumentChunk, float]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return output

        ids = results["ids"][0]
        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            text = documents[i] if i < len(documents) else ""
            dist = distances[i] if i < len(distances) else 1.0

            # Convert cosine distance to cosine similarity: sim = 1.0 - dist
            sim_score = max(0.0, min(1.0, 1.0 - dist))

            chunk = DocumentChunk(
                chunk_id=ids[i],
                text=text,
                source=meta.get("source", "unknown"),
                page=meta.get("page", 1),
                chunk_index=meta.get("chunk_index", 0),
                section=meta.get("section", "General"),
                char_count=meta.get("char_count", len(text)),
                token_count=meta.get("token_count", len(text) // 4),
                metadata=meta,
            )
            output.append((chunk, sim_score))

        return output

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Return list of distinct source documents with chunk counts."""
        count = self.collection.count()
        if count == 0:
            return []

        # Retrieve all items metadata
        all_data = self.collection.get(include=["metadatas"])
        docs_summary: Dict[str, Dict[str, Any]] = {}

        if all_data and all_data.get("metadatas"):
            for meta in all_data["metadatas"]:
                src = meta.get("source", "unknown")
                page = meta.get("page", 1)
                if src not in docs_summary:
                    docs_summary[src] = {
                        "filename": src,
                        "chunk_count": 0,
                        "pages": set(),
                    }
                docs_summary[src]["chunk_count"] += 1
                docs_summary[src]["pages"].add(page)

        return [
            {
                "filename": data["filename"],
                "chunk_count": data["chunk_count"],
                "total_pages": len(data["pages"]),
            }
            for data in docs_summary.values()
        ]
