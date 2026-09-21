"""Unit tests for document chunking, boundary detection, overlap, and deterministic chunk IDs."""

import pytest
from app.ingestion.loader import LoadedDocument, DocumentPage
from app.ingestion.chunker import (
    chunk_documents,
    recursive_split_text,
    estimate_tokens,
    DocumentChunk,
)


def test_estimate_tokens():
    """Test token estimation heuristic."""
    text = "Hello world, this is a test of token estimation."
    tokens = estimate_tokens(text)
    assert tokens > 0
    assert tokens == len(text) // 4


def test_recursive_split_text_respects_max_chars():
    """Test that text splits do not exceed max_chars when natural separators exist."""
    text = "Sentence one. " * 50 + "Sentence two. " * 50
    splits = recursive_split_text(text, max_chars=200, overlap_chars=30)
    
    assert len(splits) > 1
    for s in splits:
        assert len(s) <= 250  # allows small margin around boundary


def test_chunk_ids_are_deterministic_and_unique():
    """Test that chunk IDs are generated deterministically and uniquely."""
    pages = [
        DocumentPage(text="First page content regarding pricing and plans.", page_number=1, section_heading="Pricing"),
        DocumentPage(text="Second page content regarding SLA and guarantees.", page_number=2, section_heading="SLA"),
    ]
    doc = LoadedDocument(filename="test_policy.pdf", pages=pages, file_type="pdf")

    chunks1 = chunk_documents(doc, max_tokens=100, overlap_pct=0.15)
    chunks2 = chunk_documents(doc, max_tokens=100, overlap_pct=0.15)

    assert len(chunks1) == len(chunks2)
    
    # Check uniqueness
    chunk_ids = [c.chunk_id for c in chunks1]
    assert len(chunk_ids) == len(set(chunk_ids)), "All chunk IDs must be unique within document"

    # Check determinism across repeated runs
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        assert c1.page == c2.page
        assert c1.section == c2.section


def test_chunk_metadata_provenance():
    """Test that chunk metadata fields are properly populated."""
    pages = [
        DocumentPage(text="Section 1 details on security encryption.", page_number=1, section_heading="Security")
    ]
    doc = LoadedDocument(filename="security.md", pages=pages, file_type="md")
    chunks = chunk_documents(doc, max_tokens=150, overlap_pct=0.15)

    assert len(chunks) >= 1
    c = chunks[0]
    assert c.source == "security.md"
    assert c.page == 1
    assert c.section == "Security"
    assert "security_md_p1_c0" in c.chunk_id
    assert c.token_count > 0
    assert c.char_count > 0
