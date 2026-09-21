"""Unit tests for OpenRouter client, defensive JSON parsing, and fallback behaviors."""

import pytest
from app.generation.llm import parse_defensive_json, OpenRouterGroundedLLM
from app.generation.models import GroundedAnswer


def test_defensive_json_strict():
    raw = '{"answer": "Test answer", "citations": [], "confidence": "high", "can_answer": true}'
    parsed = parse_defensive_json(raw)
    assert parsed is not None
    assert parsed["answer"] == "Test answer"
    assert parsed["can_answer"] is True


def test_defensive_json_with_markdown_fences():
    raw = """```json
{
  "answer": "Answer inside markdown fence",
  "citations": [],
  "confidence": "high",
  "can_answer": true
}
```"""
    parsed = parse_defensive_json(raw)
    assert parsed is not None
    assert parsed["answer"] == "Answer inside markdown fence"


def test_defensive_json_with_generic_fences():
    raw = """```
{
  "answer": "Answer inside generic fence",
  "citations": [],
  "confidence": "high",
  "can_answer": true
}
```"""
    parsed = parse_defensive_json(raw)
    assert parsed is not None
    assert parsed["answer"] == "Answer inside generic fence"


def test_defensive_json_with_leading_and_trailing_text():
    raw = """Sure, here is the JSON output you requested:
{
  "answer": "Human-like answer with commentary around it",
  "citations": [
    {
      "source": "handbook.pdf",
      "page": 2,
      "chunk_id": "handbook_p2_c1",
      "quoted_snippet": "exact quote here"
    }
  ],
  "confidence": "high",
  "can_answer": true
}
Hope this answers your question! Let me know if you need more details."""
    parsed = parse_defensive_json(raw)
    assert parsed is not None
    assert parsed["answer"] == "Human-like answer with commentary around it"
    assert len(parsed["citations"]) == 1
    assert parsed["citations"][0]["chunk_id"] == "handbook_p2_c1"


def test_defensive_json_invalid_returns_none():
    raw = "This is just plain text without any json object at all."
    parsed = parse_defensive_json(raw)
    assert parsed is None


def test_openrouter_llm_instantiation():
    llm = OpenRouterGroundedLLM(
        api_key="mock-test-key",
        primary_model="meta-llama/llama-3.3-70b-instruct:free",
        fallback_model="google/gemini-2.0-flash-exp:free",
    )
    assert llm.primary_model == "meta-llama/llama-3.3-70b-instruct:free"
    assert llm.fallback_model == "google/gemini-2.0-flash-exp:free"
