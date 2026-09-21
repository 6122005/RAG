"""LangGraph state definition for the grounded RAG workflow."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from ..generation.models import GroundedAnswer


class RAGState(TypedDict):
    """Execution state schema tracked across LangGraph nodes."""

    query: str
    conversation_id: str
    metadata_filter: Optional[Dict[str, Any]]
    threshold_override: Optional[float]
    
    # Retrieval & Reranking artifacts
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    pre_rerank_scores: List[float]
    post_rerank_scores: List[float]
    
    # Pre-generation confidence status
    is_confidence_sufficient: bool
    refusal_reason: Optional[str]
    
    # Multi-turn history (for reference resolution)
    conversation_history: List[Dict[str, str]]
    
    # Generation & Validation artifacts
    raw_answer: Optional[GroundedAnswer]
    final_answer: Optional[GroundedAnswer]
    retry_count: int
    validation_issues: List[str]
    validation_exhausted: bool
    
    # Telemetry and node trace
    trace_logs: List[str]
    model_used: Optional[str]
