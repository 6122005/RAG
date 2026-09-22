"""LangGraph state machine implementing explicit nodes, guardrails, and checkpointer memory."""

import time
from typing import Any, Dict, List, Literal, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import RAGState
from .tools import get_shared_hybrid_retriever
from ..retrieval.reranker import CrossEncoderReranker
from ..generation.models import GroundedAnswer, Citation
from ..generation.prompts import SYSTEM_GROUNDED_RAG_PROMPT, USER_QUERY_PROMPT
from ..generation.llm import get_llm
from ..generation.guardrails import (
    check_retrieval_confidence,
    validate_citations,
    create_exhausted_fallback_answer,
    sanitize_input_query,
)
from ..config import get_settings


# Global shared components
_reranker: Optional[CrossEncoderReranker] = None
_checkpointer = MemorySaver()


def get_shared_reranker() -> CrossEncoderReranker:
    """Singleton getter for CrossEncoder reranker."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


# ==============================================================================
# Graph Node Definitions
# ==============================================================================

def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """Node 1: Hybrid dense + sparse search for top candidates."""
    query = sanitize_input_query(state["query"])
    metadata_filter = state.get("metadata_filter")
    settings = get_settings()

    retriever = get_shared_hybrid_retriever()
    candidates = retriever.retrieve(
        query=query,
        k=settings.TOP_K_RETRIEVAL,
        metadata_filter=metadata_filter,
    )

    retrieved_chunks = []
    pre_scores = []
    for chunk, score in candidates:
        retrieved_chunks.append({
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "page": chunk.page,
            "section": chunk.section,
            "text": chunk.text,
            "char_count": chunk.char_count,
            "token_count": chunk.token_count,
            "pre_score": score,
        })
        pre_scores.append(score)

    return {
        "retrieved_chunks": retrieved_chunks,
        "pre_rerank_scores": pre_scores,
        "trace_logs": state.get("trace_logs", []) + [f"retrieve_node: found {len(candidates)} candidates"],
    }


def rerank_node(state: RAGState) -> Dict[str, Any]:
    """Node 2: Cross-encoder re-scoring of candidates to cut down to top-n."""
    query = state["query"]
    retrieved = state.get("retrieved_chunks", [])
    settings = get_settings()

    if not retrieved:
        return {
            "reranked_chunks": [],
            "post_rerank_scores": [],
            "trace_logs": state.get("trace_logs", []) + ["rerank_node: no chunks to rerank"],
        }

    # Reconstruct chunk tuples for reranker
    from ..ingestion.chunker import DocumentChunk
    candidate_tuples = [
        (
            DocumentChunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                source=item["source"],
                page=item["page"],
                chunk_index=0,
                section=item.get("section", "General"),
                char_count=item.get("char_count", len(item["text"])),
                token_count=item.get("token_count", len(item["text"]) // 4),
            ),
            item.get("pre_score", 0.5),
        )
        for item in retrieved
    ]

    reranker = get_shared_reranker()
    top_n = settings.TOP_N_RERANK
    reranked = reranker.rerank(query=query, candidates=candidate_tuples, top_n=top_n)

    post_scores = [item["rerank_score"] for item in reranked]

    return {
        "reranked_chunks": reranked,
        "post_rerank_scores": post_scores,
        "trace_logs": state.get("trace_logs", []) + [
            f"rerank_node: scored {len(reranked)} chunks (top score: {post_scores[0] if post_scores else 0.0})"
        ],
    }


def confidence_check_node(state: RAGState) -> Dict[str, Any]:
    """Node 3: Guardrail check on top reranked score."""
    reranked = state.get("reranked_chunks", [])
    override = state.get("threshold_override")

    is_sufficient, reason = check_retrieval_confidence(reranked, threshold=override)

    return {
        "is_confidence_sufficient": is_sufficient,
        "refusal_reason": reason,
        "trace_logs": state.get("trace_logs", []) + [
            f"confidence_check_node: sufficient={is_sufficient}, reason={reason}"
        ],
    }


def refusal_node(state: RAGState) -> Dict[str, Any]:
    """Node 4a: Terminal node when retrieval confidence is inadequate."""
    reason = state.get("refusal_reason") or "Insufficient grounded evidence."
    answer = GroundedAnswer(
        answer=f"I do not have sufficient information in the provided documents to answer your question. ({reason})",
        citations=[],
        confidence="low",
        can_answer=False,
        model_used="deterministic/refusal-guardrail",
    )
    return {
        "final_answer": answer,
        "model_used": "deterministic/refusal-guardrail",
        "trace_logs": state.get("trace_logs", []) + ["refusal_node: executed refusal"],
    }


def format_context_for_prompt(reranked_chunks: List[Dict[str, Any]]) -> str:
    """Format candidate chunks with explicit citation identifiers."""
    parts = []
    for item in reranked_chunks:
        cid = item.get("chunk_id", "unknown")
        src = item.get("source", "unknown")
        page = item.get("page", 1)
        text = item.get("text", "")
        parts.append(f"[CHUNK ID: {cid} | SOURCE: {src} | PAGE: {page}]\nText: {text}")
    return "\n\n".join(parts) if parts else "[NO RETRIEVED CONTEXT]"


def format_history_for_prompt(history: List[Dict[str, str]], max_turns: int = 4) -> str:
    """Format recent turns for conversational pronoun/entity resolution."""
    if not history:
        return "[No prior turns]"
    recent = history[-max_turns:]
    formatted = []
    for turn in recent:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "")
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


def generate_node(state: RAGState) -> Dict[str, Any]:
    """Node 4b: Grounded structured generation via LLM."""
    query = state["query"]
    reranked = state.get("reranked_chunks", [])
    history = state.get("conversation_history", [])
    retry_count = state.get("retry_count", 0)
    validation_issues = state.get("validation_issues", [])

    context_str = format_context_for_prompt(reranked)
    history_str = format_history_for_prompt(history)

    # Build prompt
    prompt_text = USER_QUERY_PROMPT.format(
        query=query,
        history=history_str,
        context=context_str,
    )

    # If this is a retry due to citation validation failure, prepend correction directive
    if retry_count > 0 and validation_issues:
        hint = (
            "ATTENTION: Your previous attempt was REJECTED because of the following citation errors:\n"
            + "\n".join(f"- {issue}" for issue in validation_issues)
            + "\nYou MUST fix this. Cite ONLY valid chunk_ids from the context above.\n\n"
        )
        prompt_text = hint + prompt_text

    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_GROUNDED_RAG_PROMPT),
        HumanMessage(content=prompt_text),
    ]

    try:
        response: GroundedAnswer = llm.invoke(messages)
    except Exception as e:
        # If cloud LLM fails (e.g. invalid API key, quota limit), gracefully fallback to MockGroundedLLM
        try:
            from ..generation.llm import MockGroundedLLM
            mock_llm = MockGroundedLLM()
            response = mock_llm.with_structured_output(GroundedAnswer).invoke(messages)
        except Exception as inner_e:
            response = GroundedAnswer(
                answer=f"An error occurred during structured generation: {str(e)}",
                citations=[],
                confidence="low",
                can_answer=False,
                model_used="error/generation-failed",
            )

    model_name = getattr(response, "model_used", None) or "unknown"

    return {
        "raw_answer": response,
        "model_used": model_name,
        "trace_logs": state.get("trace_logs", []) + [
            f"generate_node (attempt {retry_count + 1}): model={model_name}, can_answer={response.can_answer}, citations={len(response.citations)}"
        ],
    }


def validate_node(state: RAGState) -> Dict[str, Any]:
    """Node 5: Verify citations against retrieved chunks with retry tracking."""
    raw_answer = state.get("raw_answer")
    reranked = state.get("reranked_chunks", [])
    retry_count = state.get("retry_count", 0)

    if not raw_answer:
        return {
            "validation_issues": ["No answer was produced to validate."],
            "validation_exhausted": True,
        }

    val_res = validate_citations(
        answer=raw_answer,
        retrieved_chunks=reranked,
        retry_count=retry_count,
        max_retries=1,
    )

    new_logs = state.get("trace_logs", []) + [
        f"validate_node: is_valid={val_res.is_valid}, issues={val_res.issues}, exhausted={val_res.validation_exhausted}"
    ]

    return {
        "validation_issues": val_res.issues,
        "validation_exhausted": val_res.validation_exhausted,
        "retry_count": retry_count + 1,
        "trace_logs": new_logs,
    }


def exhausted_fallback_node(state: RAGState) -> Dict[str, Any]:
    """Node 6a: Fallback when validation fails repeatedly (prevents hallucination)."""
    issues = state.get("validation_issues", [])
    fallback_answer = create_exhausted_fallback_answer(issues)
    fallback_answer.model_used = "deterministic/exhausted-fallback"
    return {
        "final_answer": fallback_answer,
        "model_used": "deterministic/exhausted-fallback",
        "validation_exhausted": True,
        "trace_logs": state.get("trace_logs", []) + [
            "exhausted_fallback_node: citation validation failed twice; executed safe fallback"
        ],
    }


def format_output_node(state: RAGState) -> Dict[str, Any]:
    """Node 6b: Successfully validated final answer formatting."""
    answer = state.get("raw_answer")
    model_name = state.get("model_used") or (getattr(answer, "model_used", None) if answer else "unknown")
    if answer and hasattr(answer, "model_used"):
        answer.model_used = model_name
    return {
        "final_answer": answer,
        "model_used": model_name,
        "trace_logs": state.get("trace_logs", []) + [f"format_output_node: finalized validated answer (model: {model_name})"],
    }


# ==============================================================================
# Conditional Edge Routing Functions
# ==============================================================================

def route_after_confidence_check(state: RAGState) -> Literal["generate", "refusal"]:
    """Route to generate if confidence sufficient, else refusal."""
    return "generate" if state.get("is_confidence_sufficient", False) else "refusal"


def route_after_validation(
    state: RAGState,
) -> Literal["format_output", "generate", "exhausted_fallback"]:
    """Route based on citation validation result and retry exhaustion."""
    issues = state.get("validation_issues", [])
    exhausted = state.get("validation_exhausted", False)

    if not issues:
        return "format_output"
    elif exhausted:
        return "exhausted_fallback"
    else:
        return "generate"


# ==============================================================================
# Build and Compile Graph
# ==============================================================================

def build_rag_graph():
    """Assemble and compile the explicit LangGraph state machine with checkpointing."""
    builder = StateGraph(RAGState)

    # Add nodes
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("confidence_check", confidence_check_node)
    builder.add_node("refusal", refusal_node)
    builder.add_node("generate", generate_node)
    builder.add_node("validate", validate_node)
    builder.add_node("exhausted_fallback", exhausted_fallback_node)
    builder.add_node("format_output", format_output_node)

    # Add edges
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "confidence_check")

    builder.add_conditional_edges(
        "confidence_check",
        route_after_confidence_check,
        {
            "generate": "generate",
            "refusal": "refusal",
        },
    )

    builder.add_edge("generate", "validate")

    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "format_output": "format_output",
            "generate": "generate",
            "exhausted_fallback": "exhausted_fallback",
        },
    )

    builder.add_edge("refusal", END)
    builder.add_edge("format_output", END)
    builder.add_edge("exhausted_fallback", END)

    # Compile with checkpointing
    return builder.compile(checkpointer=_checkpointer)


# Compiled singleton graph
rag_graph = build_rag_graph()


def execute_rag_pipeline(
    query: str,
    conversation_id: str,
    metadata_filter: Optional[Dict[str, Any]] = None,
    threshold_override: Optional[float] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Execute the full RAG pipeline through the LangGraph state machine."""
    start_time = time.perf_counter()

    initial_state: RAGState = {
        "query": query,
        "conversation_id": conversation_id,
        "metadata_filter": metadata_filter,
        "threshold_override": threshold_override,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "pre_rerank_scores": [],
        "post_rerank_scores": [],
        "is_confidence_sufficient": False,
        "refusal_reason": None,
        "conversation_history": conversation_history or [],
        "raw_answer": None,
        "final_answer": None,
        "retry_count": 0,
        "validation_issues": [],
        "validation_exhausted": False,
        "trace_logs": [],
        "model_used": None,
    }

    config = {"configurable": {"thread_id": conversation_id}}
    result_state = rag_graph.invoke(initial_state, config=config)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    final_ans = result_state.get("final_answer")
    model_used = result_state.get("model_used") or (getattr(final_ans, "model_used", None) if final_ans else "unknown")

    return {
        "final_answer": final_ans,
        "retrieved_chunks": result_state.get("reranked_chunks", []),
        "pre_rerank_scores": result_state.get("pre_rerank_scores", []),
        "post_rerank_scores": result_state.get("post_rerank_scores", []),
        "latency_ms": round(latency_ms, 2),
        "validation_exhausted": result_state.get("validation_exhausted", False),
        "trace_logs": result_state.get("trace_logs", []),
        "model_used": model_used,
    }
