"""LangChain / LangGraph tool implementations for document search and inspection."""

from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from ..retrieval.vector_store import ChromaVectorStore
from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.hybrid import HybridRetriever


_hybrid_retriever: Optional[HybridRetriever] = None


def get_shared_hybrid_retriever() -> HybridRetriever:
    """Singleton getter for shared hybrid retriever."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        vs = ChromaVectorStore()
        bm25 = BM25Retriever()
        _hybrid_retriever = HybridRetriever(vs, bm25)
    return _hybrid_retriever


@tool
def search_documents(
    query: str,
    top_k: int = 10,
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search ingested document corpus using hybrid dense and sparse search.

    Args:
        query: The search query text.
        top_k: Number of candidates to retrieve.
        source_filter: Optional source filename to restrict search.
    """
    retriever = get_shared_hybrid_retriever()
    filter_dict = {"source": source_filter} if source_filter else None
    results = retriever.retrieve(query=query, k=top_k, metadata_filter=filter_dict)

    return [
        {
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "page": chunk.page,
            "section": chunk.section,
            "text": chunk.text,
            "score": score,
        }
        for chunk, score in results
    ]


@tool
def get_document_metadata(filename: str) -> Dict[str, Any]:
    """Retrieve metadata information regarding an ingested document.

    Args:
        filename: Name of the document file (e.g. handbook.pdf)
    """
    vs = ChromaVectorStore()
    docs = vs.get_all_documents()
    for d in docs:
        if d["filename"] == filename:
            return d
    return {"filename": filename, "status": "not_found", "chunk_count": 0}
