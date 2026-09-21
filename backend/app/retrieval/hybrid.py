"""Hybrid retrieval merging dense vector search and BM25 sparse keyword search via RRF."""

from typing import Any, Dict, List, Optional, Tuple
from .vector_store import ChromaVectorStore
from .bm25_retriever import BM25Retriever
from ..ingestion.chunker import DocumentChunk


class HybridRetriever:
    """Combines dense vector search and sparse BM25 retrieval using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_retriever: BM25Retriever,
        rrf_constant: int = 60,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_constant = rrf_constant

    def retrieve(
        self,
        query: str,
        k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve top-k candidates from dense and sparse search, merged and deduplicated."""
        # 1. Fetch dense vector results
        dense_results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            metadata_filter=metadata_filter,
        )

        # 2. Fetch sparse BM25 results
        sparse_results = self.bm25_retriever.search(
            query=query,
            k=k,
            metadata_filter=metadata_filter,
        )

        # 3. Apply Reciprocal Rank Fusion (RRF)
        # RRF score = sum(1.0 / (rrf_constant + rank))
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        # Add ranks from dense search
        for rank, (chunk, _score) in enumerate(dense_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
                1.0 / (self.rrf_constant + rank)
            )

        # Add ranks from sparse BM25 search
        for rank, (chunk, _score) in enumerate(sparse_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (
                1.0 / (self.rrf_constant + rank)
            )

        # If no results from both, return empty
        if not rrf_scores:
            return []

        # Normalize RRF scores to 0-1 range for clean downstream observability
        max_rrf = max(rrf_scores.values())
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        merged: List[Tuple[DocumentChunk, float]] = []
        for cid in sorted_chunk_ids[:k]:
            normalized_score = float(rrf_scores[cid] / max_rrf) if max_rrf > 0 else 0.0
            merged.append((chunk_map[cid], round(normalized_score, 4)))

        return merged
