"""FastAPI route handlers for document ingestion, query execution, and health."""

import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from .auth import verify_api_key
from .rate_limiter import limiter
from ..config import get_settings
from ..logger import log_query_execution
from ..generation.models import (
    QueryRequest,
    QueryResponse,
    IngestResponse,
    DocumentMetadata,
    GroundedAnswer,
)
from ..ingestion.pipeline import IngestionPipeline
from ..retrieval.vector_store import ChromaVectorStore, get_shared_vector_store
from ..retrieval.bm25_retriever import BM25Retriever
from ..graph.workflow import execute_rag_pipeline, get_shared_hybrid_retriever

router = APIRouter()


@router.api_route("/", methods=["GET", "HEAD"], tags=["System"])
async def root_endpoint() -> Dict[str, str]:
    """Root endpoint for ping and uptime checkers."""
    return {"status": "ok", "message": "Grounded RAG API is running"}


@router.api_route("/health", methods=["GET", "HEAD"], tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Lightweight zero-allocation health check verifying service status and active configurations."""
    settings = get_settings()
    model_name = (
        settings.OPENROUTER_MODEL
        if settings.LLM_PROVIDER == "openrouter"
        else (settings.GEMINI_MODEL if settings.LLM_PROVIDER == "gemini" else "default")
    )
    is_cloud_fast = bool(os.environ.get("RENDER")) or settings.EMBEDDING_PROVIDER.lower() in ("fast", "fast-semantic", "lightweight")
    active_emb = "fast-semantic" if is_cloud_fast else settings.EMBEDDING_PROVIDER
    return {
        "status": "healthy",
        "version": "v1.2-fast-cloud",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": model_name,
        "has_openrouter_key": bool(settings.OPENROUTER_API_KEY),
        "has_gemini_key": bool(settings.GEMINI_API_KEY),
        "embedding_provider": active_emb,
        "embedding_model": "384d-fast-semantic" if is_cloud_fast else settings.EMBEDDING_MODEL,
        "reranker_model": settings.RERANKER_MODEL,
    }



@router.post(
    "/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Ingestion"],
)
@limiter.limit("30/minute")
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
) -> IngestResponse:
    """Upload and process PDF, TXT, or Markdown documents into the vector store and BM25 index."""
    filename = file.filename or "uploaded_file.txt"
    ext = Path(filename).suffix.lower()

    if ext not in [".pdf", ".txt", ".md", ".markdown"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported extensions: .pdf, .txt, .md",
        )

    settings = get_settings()
    doc_store = Path(settings.DOCUMENT_STORE_DIR)
    doc_store.mkdir(parents=True, exist_ok=True)
    temp_path = doc_store / filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        vs = get_shared_vector_store()
        bm25 = BM25Retriever()
        pipeline = IngestionPipeline(vector_store=vs, bm25_retriever=bm25)

        result = pipeline.ingest_file(temp_path)

        # Refresh shared retriever instance
        hybrid = get_shared_hybrid_retriever()
        hybrid.vector_store = vs
        hybrid.bm25_retriever = bm25

        import gc
        gc.collect()

        return IngestResponse(
            filename=filename,
            status=result.get("status", "success"),
            chunk_count=result.get("chunk_count", 0),
            page_count=result.get("page_count", 1),
            file_type=result.get("file_type", ext.lstrip(".")),
            message=result.get("message", "Document successfully ingested and indexed."),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting document '{filename}': {str(e)}",
        )


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Query"],
)
@limiter.limit("60/minute")
async def execute_query(
    request: Request,
    body: QueryRequest,
) -> QueryResponse:
    """Execute grounded RAG query through LangGraph state machine with citations and guardrails."""
    # Ensure active conversation ID
    conversation_id = body.conversation_id or str(uuid.uuid4())

    pipeline_result = execute_rag_pipeline(
        query=body.query,
        conversation_id=conversation_id,
        metadata_filter=body.metadata_filter,
        threshold_override=body.threshold_override,
    )

    final_answer: GroundedAnswer = pipeline_result["final_answer"]
    retrieved_chunks = pipeline_result["retrieved_chunks"]
    pre_scores = pipeline_result["pre_rerank_scores"]
    post_scores = pipeline_result["post_rerank_scores"]
    latency_ms = pipeline_result["latency_ms"]
    validation_exhausted = pipeline_result["validation_exhausted"]

    model_used = pipeline_result.get("model_used") or getattr(final_answer, "model_used", "unknown")

    # Log structured JSON telemetry
    chunk_ids = [c.get("chunk_id", "") for c in retrieved_chunks]
    log_query_execution(
        query=body.query,
        conversation_id=conversation_id,
        retrieved_chunk_ids=chunk_ids,
        pre_rerank_scores=pre_scores,
        post_rerank_scores=post_scores,
        can_answer=final_answer.can_answer,
        confidence=final_answer.confidence,
        latency_ms=latency_ms,
        validation_exhausted=validation_exhausted,
        model_used=model_used,
    )

    return QueryResponse(
        answer=final_answer.answer,
        citations=final_answer.citations,
        confidence=final_answer.confidence,
        can_answer=final_answer.can_answer,
        conversation_id=conversation_id,
        retrieved_chunks=retrieved_chunks,
        latency_ms=latency_ms,
        validation_exhausted=validation_exhausted,
        model_used=model_used,
    )


@router.get(
    "/documents",
    response_model=List[DocumentMetadata],
    tags=["Documents"],
)
async def list_documents() -> List[DocumentMetadata]:
    """List all ingested documents along with chunk and page counts."""
    vs = get_shared_vector_store()
    docs = vs.get_all_documents()
    return [
        DocumentMetadata(
            filename=d["filename"],
            chunk_count=d["chunk_count"],
            total_pages=d["total_pages"],
        )
        for d in docs
    ]
