"""Generation package: models, prompts, LLM factories, and guardrails."""

from .models import Citation, GroundedAnswer, QueryRequest, QueryResponse, IngestResponse
from .prompts import SYSTEM_GROUNDED_RAG_PROMPT, USER_QUERY_PROMPT
from .llm import get_llm
from .guardrails import (
    check_retrieval_confidence,
    validate_citations,
    sanitize_input_query,
    ValidationResult,
)

__all__ = [
    "Citation",
    "GroundedAnswer",
    "QueryRequest",
    "QueryResponse",
    "IngestResponse",
    "SYSTEM_GROUNDED_RAG_PROMPT",
    "USER_QUERY_PROMPT",
    "get_llm",
    "check_retrieval_confidence",
    "validate_citations",
    "sanitize_input_query",
    "ValidationResult",
]
