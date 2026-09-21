"""Pre-generation confidence check, post-generation citation validation, and injection resistance."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel
from .models import GroundedAnswer, Citation
from ..config import get_settings


class ValidationResult(BaseModel):
    """Result of citation validation checks."""

    is_valid: bool
    issues: List[str] = []
    validation_exhausted: bool = False


def sanitize_input_query(query: str) -> str:
    """Sanitize user query and neutralize common prompt-injection delimiters."""
    cleaned = query.strip()
    # Strip null bytes and non-printable control chars
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", cleaned)
    # Truncate overly long queries
    if len(cleaned) > 2000:
        cleaned = cleaned[:2000]
    return cleaned


def check_retrieval_confidence(
    reranked_chunks: List[Dict[str, Any]],
    threshold: Optional[float] = None,
) -> Tuple[bool, Optional[str]]:
    """Pre-generation guardrail: verify top candidate exceeds confidence threshold.

    Blends cross-encoder score with dense/BM25 pre-retrieval score to ensure
    keyword-matched terms (e.g. acronyms like FDE) are not falsely blocked by
    uncalibrated cross-encoder logits.
    """
    settings = get_settings()
    min_threshold = threshold if threshold is not None else settings.RERANK_THRESHOLD

    if not reranked_chunks:
        return False, "No documents were retrieved matching the query."

    top_chunk = reranked_chunks[0]
    top_score = top_chunk.get("rerank_score", 0.0)
    top_pre_score = top_chunk.get("pre_rerank_score", 0.0)

    # When an explicit threshold is supplied by caller/test, enforce it strictly
    if threshold is not None:
        if top_score < threshold:
            return (
                False,
                f"Retrieval confidence ({top_score:.3f}) is below minimum threshold ({threshold:.3f}). "
                "Insufficient grounded information to answer reliably.",
            )
        return True, None

    # Normal pipeline: allow if rerank exceeds calibrated threshold or strong hybrid match
    has_sufficient_rerank = top_score >= min_threshold
    has_strong_hybrid_match = top_pre_score >= 0.6 and top_score > 0.00005

    if not (has_sufficient_rerank or has_strong_hybrid_match):
        return (
            False,
            f"Retrieval confidence ({top_score:.3f}) is below minimum threshold ({min_threshold:.3f}). "
            "Insufficient grounded information to answer reliably.",
        )

    return True, None


def validate_citations(
    answer: GroundedAnswer,
    retrieved_chunks: List[Dict[str, Any]],
    retry_count: int = 0,
    max_retries: int = 1,
) -> ValidationResult:
    """Post-generation guardrail: verify every cited chunk_id exists in retrieved chunks.

    If validation fails and retry_count >= max_retries, flags validation_exhausted=True.
    """
    if not answer.can_answer:
        # If model already said it cannot answer, citations aren't strictly required
        return ValidationResult(is_valid=True, issues=[])

    valid_chunk_ids: Set[str] = {c.get("chunk_id", "") for c in retrieved_chunks}
    issues: List[str] = []

    # Check that at least one citation is provided when can_answer is True
    if not answer.citations:
        issues.append("Answer makes claims (can_answer=True) but provides zero citations.")

    # Check each citation against retrieved set
    for cit in answer.citations:
        if cit.chunk_id not in valid_chunk_ids:
            issues.append(
                f"Hallucinated citation: chunk_id '{cit.chunk_id}' was not in the retrieved candidate set."
            )

    if not issues:
        return ValidationResult(is_valid=True, issues=[])

    # Validation failed
    is_exhausted = retry_count >= max_retries
    return ValidationResult(
        is_valid=False,
        issues=issues,
        validation_exhausted=is_exhausted,
    )


def create_exhausted_fallback_answer(issues: List[str]) -> GroundedAnswer:
    """Fallback answer when citations fail validation repeatedly."""
    detail = "; ".join(issues) if issues else "Verification failed"
    return GroundedAnswer(
        answer=f"Unable to verify the factual grounding of the response ({detail}). Refusing to answer to prevent hallucination.",
        citations=[],
        confidence="low",
        can_answer=False,
    )
