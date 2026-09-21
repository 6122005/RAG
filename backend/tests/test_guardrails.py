"""Unit tests for guardrails, citation validation, prompt injection resistance, and fallback."""

import pytest
from app.generation.models import GroundedAnswer, Citation
from app.generation.guardrails import (
    validate_citations,
    check_retrieval_confidence,
    create_exhausted_fallback_answer,
    sanitize_input_query,
)


@pytest.fixture
def mock_retrieved_chunks():
    return [
        {"chunk_id": "policy_p1_c0", "source": "policy.pdf", "text": "Refunds are granted within 14 days."},
        {"chunk_id": "policy_p1_c1", "source": "policy.pdf", "text": "Annual contracts require 60 days notice."},
    ]


def test_sanitize_input_query():
    """Test query sanitization and control character removal."""
    dirty = "What is the SLA?\x00\x08"
    clean = sanitize_input_query(dirty)
    assert clean == "What is the SLA?"
    assert "\x00" not in clean


def test_validate_citations_valid(mock_retrieved_chunks):
    """Test valid citation passes validation cleanly."""
    answer = GroundedAnswer(
        answer="Customers can get a refund within 14 days.",
        citations=[
            Citation(
                source="policy.pdf",
                page=1,
                chunk_id="policy_p1_c0",
                quoted_snippet="Refunds are granted within 14 days.",
            )
        ],
        confidence="high",
        can_answer=True,
    )

    result = validate_citations(answer, mock_retrieved_chunks, retry_count=0)
    assert result.is_valid
    assert len(result.issues) == 0
    assert not result.validation_exhausted


def test_validate_citations_rejects_hallucinated_chunk_id(mock_retrieved_chunks):
    """Test that citation validator catches fabricated chunk IDs."""
    answer = GroundedAnswer(
        answer="Customers can get a refund within 14 days.",
        citations=[
            Citation(
                source="policy.pdf",
                page=1,
                chunk_id="fabricated_nonexistent_id",
                quoted_snippet="Refunds are granted within 14 days.",
            )
        ],
        confidence="high",
        can_answer=True,
    )

    # Initial attempt (retry_count=0)
    result = validate_citations(answer, mock_retrieved_chunks, retry_count=0, max_retries=1)
    assert not result.is_valid
    assert not result.validation_exhausted
    assert any("Hallucinated citation" in issue for issue in result.issues)

    # Second attempt (retry_count=1 >= max_retries) -> exhaustion
    result_exhausted = validate_citations(answer, mock_retrieved_chunks, retry_count=1, max_retries=1)
    assert not result_exhausted.is_valid
    assert result_exhausted.validation_exhausted


def test_create_exhausted_fallback_answer():
    """Test that exhausted validation creates clean refusal answer."""
    fallback = create_exhausted_fallback_answer(["Hallucinated chunk_id"])
    assert not fallback.can_answer
    assert fallback.confidence == "low"
    assert "Refusing to answer" in fallback.answer
    assert len(fallback.citations) == 0


def test_validation_bypassed_when_can_answer_false():
    """If model already declared can_answer=False, citations should not trigger failure."""
    answer = GroundedAnswer(
        answer="Information is missing.",
        citations=[],
        confidence="low",
        can_answer=False,
    )
    result = validate_citations(answer, [], retry_count=0)
    assert result.is_valid
