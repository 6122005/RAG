"""Pydantic schemas for grounded generation, citations, requests, and responses."""

import uuid
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    """Source provenance citation verifying a claim."""

    source: str = Field(description="Filename of the source document (e.g. handbook.pdf)")
    page: int = Field(default=1, description="Page or section number of the citation")
    chunk_id: str = Field(description="Deterministic ID of the chunk containing the claim")
    quoted_snippet: str = Field(
        description="Direct quoted snippet from the chunk verifying the claim (max 25 words)"
    )

    @field_validator("quoted_snippet")
    @classmethod
    def validate_snippet_length(cls, v: str) -> str:
        words = v.strip().split()
        if len(words) > 35:  # generous bound allowing up to ~35 words
            return " ".join(words[:25]) + "..."
        return v.strip()


class GroundedAnswer(BaseModel):
    """Structured output schema enforced on LLM generation."""

    answer: str = Field(
        description="Factual response grounded strictly in retrieved context. "
        "If insufficient information exists, clearly explain what is missing."
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of exact citations supporting the factual claims made in the answer.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level in the answer based solely on retrieved document support."
    )
    can_answer: bool = Field(
        description="True if the retrieved content contains sufficient information to answer the question, False otherwise."
    )
    model_used: Optional[str] = Field(
        default=None,
        description="The actual LLM provider and model identifier that generated this answer.",
    )


class QueryRequest(BaseModel):
    """Incoming user query request."""

    query: str = Field(min_length=1, max_length=2000, description="Natural language question")
    conversation_id: Optional[str] = Field(
        default=None,
        description="UUID representing multi-turn conversation session. Generated if not provided.",
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filter (e.g. {'source': 'policy.pdf'})",
    )
    threshold_override: Optional[float] = Field(
        default=None,
        description="Optional override for reranking score threshold",
    )


class QueryResponse(BaseModel):
    """Full API response for a query execution."""

    answer: str
    citations: List[Citation]
    confidence: Literal["high", "medium", "low"]
    can_answer: bool
    conversation_id: str
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
    validation_exhausted: bool = False
    refusal_reason: Optional[str] = None
    model_used: Optional[str] = None


class IngestResponse(BaseModel):
    """Response returned upon document ingestion."""

    filename: str
    status: str
    chunk_count: int
    page_count: int = 1
    file_type: str = "txt"
    message: str = "Document successfully ingested and indexed."


class DocumentMetadata(BaseModel):
    """Metadata summary of an ingested document."""

    filename: str
    chunk_count: int
    total_pages: int
