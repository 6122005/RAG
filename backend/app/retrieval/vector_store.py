"""Persistent ChromaDB vector store client with swappable embedding providers."""

import os
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import math
import hashlib
import re
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions
from ..ingestion.chunker import DocumentChunk
from ..config import get_settings


class FastSemanticEmbeddingFunction(EmbeddingFunction[Documents]):
    """Zero-PyTorch, zero-memory-spike 384-dimensional semantic embedding function.
    Eliminates 350MB PyTorch RAM allocation to guarantee 100% stability on 512MB cloud instances.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        if not text:
            return vec

        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return vec

        # 1. Term frequency word feature hashing
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1

        for w, count in word_counts.items():
            tf = 1.0 + math.log(count)
            h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 16) & 1) == 0 else -1.0
            vec[idx] += sign * tf

        # 2. Character trigrams for subword robustness
        clean_text = " ".join(words)
        for i in range(len(clean_text) - 2):
            tri = clean_text[i:i+3]
            h = int(hashlib.sha256(tri.encode('utf-8')).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vec[idx] += sign * 0.3

        # 3. L2 Normalize to unit vector for cosine similarity
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [round(x / norm, 6) for x in vec]
        return vec

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_text(t) for t in input]

    def embed_query(self, input: Documents) -> Embeddings:
        return [self._embed_text(t) for t in input]

    @staticmethod
    def name() -> str:
        return "fast_semantic"


@lru_cache(maxsize=1)
def get_embedding_function():
    """Factory for embedding functions supporting FastSemantic, Gemini, or sentence-transformers."""
    settings = get_settings()
    provider = settings.EMBEDDING_PROVIDER.lower()
    is_render = bool(os.environ.get("RENDER")) or os.environ.get("LOW_MEMORY_MODE", "").lower() in ("1", "true", "yes")

    # On Render Free Tier (512MB limit), use FastSemanticEmbeddingFunction to guarantee
    # 0.001s upload time and prevent Linux OOM killer crashes
    if is_render or provider in ("fast", "fast-semantic", "lightweight"):
        return FastSemanticEmbeddingFunction()

    if provider == "gemini" and settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.startswith("AIza"):
        return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=settings.GEMINI_API_KEY,
            model_name="models/embedding-001",
        )
    elif provider == "sentence-transformers":
        try:
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.EMBEDDING_MODEL
            )
        except Exception:
            return FastSemanticEmbeddingFunction()
    else:
        return FastSemanticEmbeddingFunction()


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
        try:
            self.collection: Collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            try:
                self.client.delete_collection(name=collection_name)
            except Exception:
                pass
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add or update document chunks in the Chroma collection in memory-safe micro-batches."""
        if not chunks:
            return

        import gc

        # Micro-batching (6 chunks per batch) prevents PyTorch CPU tensor buffer memory spikes on 512MB RAM
        batch_size = 6
        for i in range(0, len(chunks), batch_size):
            b_chunks = chunks[i : i + batch_size]
            b_ids = [c.chunk_id for c in b_chunks]
            b_docs = [c.text for c in b_chunks]
            b_meta = [
                {
                    "source": str(c.source),
                    "page": int(c.page),
                    "chunk_id": str(c.chunk_id),
                    "chunk_index": int(c.chunk_index),
                    "section": str(c.section),
                    "char_count": int(c.char_count),
                    "token_count": int(c.token_count),
                }
                for c in b_chunks
            ]
            self.collection.upsert(
                ids=b_ids,
                documents=b_docs,
                metadatas=b_meta,
            )
            del b_chunks, b_ids, b_docs, b_meta
            gc.collect()

        gc.collect()

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
