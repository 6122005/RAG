"""Cross-encoder reranking module for semantic passage re-scoring."""

import math
from typing import Any, Dict, List, Optional, Tuple
from sentence_transformers import CrossEncoder
from ..ingestion.chunker import DocumentChunk
from ..config import get_settings


def sigmoid(x: float) -> float:
    """Map raw cross-encoder logit to [0, 1] probability."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class CrossEncoderReranker:
    """Cross-encoder reranker providing deep query-passage semantic scoring."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy load cross-encoder model on first inference."""
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[DocumentChunk, float]],
        top_n: int = 4,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-encoder, returning top_n with pre and post scores."""
        if not candidates:
            return []

        # Prepare pairs for cross-encoder
        pairs = [[query, chunk.text] for chunk, _pre_score in candidates]

        try:
            raw_scores = self.model.predict(pairs)
        except Exception:
            # Fallback if cross-encoder fails
            raw_scores = [score for _chunk, score in candidates]

        scored_results: List[Dict[str, Any]] = []
        for i, (chunk, pre_score) in enumerate(candidates):
            raw = float(raw_scores[i])
            # Normalize logit to [0, 1] for intuitive thresholding
            post_score = round(sigmoid(raw), 4)

            scored_results.append({
                "chunk": chunk,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page": chunk.page,
                "section": chunk.section,
                "text": chunk.text,
                "pre_rerank_score": round(float(pre_score), 4),
                "rerank_score": post_score,
                "raw_rerank_score": round(raw, 4),
            })

        # Sort descending by post-rerank score
        scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_results[:top_n]
