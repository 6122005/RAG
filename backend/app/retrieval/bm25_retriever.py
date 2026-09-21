"""BM25 keyword search retriever with persistent corpus storage."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from rank_bm25 import BM25Okapi
from ..ingestion.chunker import DocumentChunk
from ..config import get_settings


def tokenize(text: str) -> List[str]:
    """Simple alphanumeric lowercase tokenizer."""
    return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())


class BM25Retriever:
    """Keyword-based BM25 retriever with serialization and filtering support."""

    def __init__(self, storage_dir: Optional[str] = None):
        settings = get_settings()
        base_dir = storage_dir or settings.CHROMA_PERSIST_DIR
        self.index_file = Path(base_dir) / "bm25_corpus.json"
        self.index_file.parent.mkdir(parents=True, exist_ok=True)

        self.chunks: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

        self._load_index()

    def _load_index(self) -> None:
        """Load stored chunks from disk and initialize BM25 index."""
        if not self.index_file.exists():
            return

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = [DocumentChunk(**item) for item in data]
            self.tokenized_corpus = [tokenize(c.text) for c in self.chunks]
            if self.tokenized_corpus:
                self.bm25 = BM25Okapi(self.tokenized_corpus)
        except Exception:
            self.chunks = []
            self.tokenized_corpus = []
            self.bm25 = None

    def _save_index(self) -> None:
        """Persist chunk corpus to disk."""
        data = [chunk.model_dump() for chunk in self.chunks]
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_chunks(self, new_chunks: List[DocumentChunk]) -> None:
        """Add new chunks to the BM25 corpus and reindex."""
        if not new_chunks:
            return

        # Deduplicate existing chunks by chunk_id
        chunk_dict = {c.chunk_id: c for c in self.chunks}
        for chunk in new_chunks:
            chunk_dict[chunk.chunk_id] = chunk

        self.chunks = list(chunk_dict.values())
        self.tokenized_corpus = [tokenize(c.text) for c in self.chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        self._save_index()

    def search(
        self,
        query: str,
        k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Perform BM25 keyword search, apply filters, and return normalized scores."""
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

        candidates: List[Tuple[DocumentChunk, float]] = []
        for i, chunk in enumerate(self.chunks):
            raw_score = scores[i]
            if raw_score <= 0:
                continue

            # Apply metadata filter if provided
            if metadata_filter:
                match = True
                for key, val in metadata_filter.items():
                    if chunk.metadata.get(key) != val and getattr(chunk, key, None) != val:
                        match = False
                        break
                if not match:
                    continue

            normalized_score = float(raw_score / max_score)
            candidates.append((chunk, normalized_score))

        # Sort descending by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]
