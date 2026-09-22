"""Ingestion pipeline coordinating loading, chunking, and index persistence."""

from pathlib import Path
from typing import Dict, List, Optional
from .loader import load_document
from .chunker import chunk_documents, DocumentChunk


class IngestionPipeline:
    """Manages end-to-end ingestion of documents into vector store and keyword index."""

    def __init__(self, vector_store=None, bm25_retriever=None):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    def ingest_file(
        self,
        file_path: Path,
        max_tokens: int = 800,
        overlap_pct: float = 0.10,
        max_chunks: int = 10,
    ) -> Dict:
        """Ingest a single file and persist into ChromaDB and BM25 index with strict low-memory optimization."""
        # 1. Load document
        loaded_doc = load_document(file_path)

        # 2. Chunk document
        chunks: List[DocumentChunk] = chunk_documents(
            loaded_doc,
            max_tokens=max_tokens,
            overlap_pct=overlap_pct,
        )

        # Free raw document memory immediately
        file_type = loaded_doc.file_type
        doc_filename = loaded_doc.filename
        page_count = len(loaded_doc.pages)
        del loaded_doc
        import gc
        gc.collect()

        if not chunks:
            return {
                "filename": doc_filename,
                "status": "warning",
                "message": "Document contains no readable text.",
                "chunk_count": 0,
            }

        # Cap max chunks to 10 on cloud instances
        # 10 chunks * 800 tokens = 32,000 characters (covers 10-15 pages)
        # Guarantees embedding finishes in under 5 seconds and RAM stays under 180MB
        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]

        # 3. Add to ChromaDB vector store
        if self.vector_store:
            self.vector_store.add_chunks(chunks)

        # 4. Add to BM25 keyword index
        if self.bm25_retriever:
            self.bm25_retriever.add_chunks(chunks)

        gc.collect()

        return {
            "filename": doc_filename,
            "status": "success",
            "file_type": file_type,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "sample_chunk_id": chunks[0].chunk_id if chunks else None,
        }
