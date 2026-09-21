"""Unit tests for CrossEncoder reranker scoring, ordering, and threshold cutoff."""

import pytest
from unittest.mock import MagicMock, patch
from app.ingestion.chunker import DocumentChunk
from app.retrieval.reranker import CrossEncoderReranker, sigmoid
from app.generation.guardrails import check_retrieval_confidence


@pytest.fixture
def sample_candidates():
    """Create a mock candidate set with high and low relevance text."""
    c1 = DocumentChunk(
        chunk_id="doc_p1_c0",
        text="The weather on Mars consists of carbon dioxide storms and subzero temperatures.",
        source="doc.txt",
        page=1,
        chunk_index=0,
        section="Space",
        char_count=80,
        token_count=20,
    )
    c2 = DocumentChunk(
        chunk_id="doc_p1_c1",
        text="Acme Cloud Platform Professional Tier costs $49 per user per month billed annually.",
        source="doc.txt",
        page=1,
        chunk_index=1,
        section="Pricing",
        char_count=85,
        token_count=21,
    )
    # Suppose c1 had higher initial vector similarity (e.g. spurious keyword overlap), c2 had lower
    return [(c1, 0.85), (c2, 0.60)]


def test_sigmoid_mapping():
    """Test logit to probability mapping."""
    assert sigmoid(0.0) == 0.5
    assert sigmoid(10.0) > 0.99
    assert sigmoid(-10.0) < 0.01


def test_reranker_reorders_candidates(sample_candidates):
    """Confirm reranker reorders candidates based on cross-encoder prediction."""
    reranker = CrossEncoderReranker()
    
    # Mock model predict to return high score for candidate 2 (pricing) and low for candidate 1 (Mars)
    # Query: "How much does the professional tier cost?"
    mock_model = MagicMock()
    mock_model.predict.return_value = [-3.0, 4.5]  # c1 -> low logit, c2 -> high logit
    reranker._model = mock_model

    query = "How much does the professional tier cost?"
    results = reranker.rerank(query=query, candidates=sample_candidates, top_n=2)

    assert len(results) == 2
    # Candidate 2 (pricing) should now be rank 1 despite lower pre-score
    assert results[0]["chunk_id"] == "doc_p1_c1"
    assert results[0]["rerank_score"] > results[1]["rerank_score"]
    assert results[0]["pre_rerank_score"] == 0.60
    assert results[1]["chunk_id"] == "doc_p1_c0"


def test_threshold_confidence_guardrail(sample_candidates):
    """Confirm confidence guardrail triggers refusal when top rerank score is below threshold."""
    reranker = CrossEncoderReranker()
    mock_model = MagicMock()
    mock_model.predict.return_value = [-5.0, -6.0]  # both very low logits
    reranker._model = mock_model

    results = reranker.rerank("Query on unrelated topic", sample_candidates, top_n=2)
    
    # High threshold
    is_sufficient, reason = check_retrieval_confidence(results, threshold=0.50)
    assert not is_sufficient
    assert "below minimum threshold" in reason

    # With empty results
    is_empty_sufficient, empty_reason = check_retrieval_confidence([], threshold=0.10)
    assert not is_empty_sufficient
    assert "No documents were retrieved" in empty_reason
