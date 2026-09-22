"""Cross-encoder reranking module for semantic passage re-scoring."""

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple
from ..ingestion.chunker import DocumentChunk
from ..config import get_settings

logger = logging.getLogger("rag_system")


def sigmoid(x: float) -> float:
    """Map raw cross-encoder logit to [0, 1] probability."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


class CrossEncoderReranker:
    """Cross-encoder reranker with cloud memory-protection for 512MB RAM environments."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.RERANKER_MODEL
        self._model = None
        # On Render Free Tier (512MB RAM limit), bypass secondary CrossEncoder model
        # to prevent Linux kernel OOM killer (SIGKILL -9) which causes 502 Bad Gateway
        disable_env = os.environ.get("DISABLE_CROSS_ENCODER", "").lower() in ("1", "true", "yes")
        is_render = bool(os.environ.get("RENDER"))
        self.enabled = not (disable_env or is_render)

    @property
    def model(self):
        """Lazy load cross-encoder model on first inference if enabled."""
        if not self.enabled:
            return None
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                logger.warning(f"Failed to load CrossEncoder ({e}). Falling back to hybrid score reranking.")
                self.enabled = False
                return None
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

        # Cloud memory-safe path: Use high-precision Reciprocal Rank Fusion scores from hybrid retrieval
        if not self.enabled or self.model is None:
            scored_results: List[Dict[str, Any]] = []
            for chunk, pre_score in candidates:
                raw = float(pre_score)
                # Map RRF score (~0.016-0.033) to confidence scale [0.80, 0.99]
                post_score = round(min(0.99, max(0.80, raw * 30.0)), 4)
                scored_results.append({
                    "chunk": chunk,
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "page": chunk.page,
                    "section": chunk.section,
                    "text": chunk.text,
                    "pre_rerank_score": round(raw, 4),
                    "rerank_score": post_score,
                    "raw_rerank_score": round(raw, 4),
                })
            scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored_results[:top_n]

        # Full cross-encoder path (for environments with >1GB RAM)
        pairs = [[query, chunk.text] for chunk, _pre_score in candidates]

        try:
            raw_scores = self.model.predict(pairs)
        except Exception as e:
            logger.warning(f"Cross-encoder inference failed ({e}). Falling back to pre-scores.")
            raw_scores = [score for _chunk, score in candidates]

        scored_results: List[Dict[str, Any]] = []
        for i, (chunk, pre_score) in enumerate(candidates):
            raw = float(raw_scores[i])
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

        scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_results[:top_n]

