"""Comprehensive evaluation harness scoring groundedness, citation accuracy, and refusal precision."""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from tabulate import tabulate

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_documents
from app.retrieval.vector_store import ChromaVectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.ingestion.pipeline import IngestionPipeline
from app.graph.workflow import execute_rag_pipeline, get_shared_hybrid_retriever
from app.generation.models import GroundedAnswer


def ensure_documents_ingested() -> None:
    """Ensure sample knowledge base is ingested prior to evaluation."""
    sample_path = Path(__file__).parent / "sample_docs" / "acme_knowledge_base.md"
    if not sample_path.exists():
        print(f"Sample doc not found at {sample_path}")
        return

    vs = ChromaVectorStore()
    bm25 = BM25Retriever()
    existing_docs = vs.get_all_documents()
    already_indexed = any(d["filename"] == sample_path.name for d in existing_docs)

    if not already_indexed:
        print(f"Indexing sample document '{sample_path.name}' into vector store & BM25...")
        pipeline = IngestionPipeline(vector_store=vs, bm25_retriever=bm25)
        res = pipeline.ingest_file(sample_path)
        print(f"Ingestion complete: {res.get('chunk_count', 0)} chunks indexed.\n")

    # Refresh shared retriever
    shared = get_shared_hybrid_retriever()
    shared.vector_store = vs
    shared.bm25_retriever = bm25


def evaluate_dataset(dataset_path: Path) -> Dict[str, Any]:
    """Run all evaluation questions through the full pipeline and score metrics."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    total_questions = len(qa_pairs)
    answerable_count = sum(1 for q in qa_pairs if q["is_answerable"])
    unanswerable_count = total_questions - answerable_count

    correct_citations = 0
    correct_answers = 0
    correct_refusals = 0
    total_latency = 0.0

    table_rows = []

    print(f"\n================================================================================")
    print(f"   STARTING PRODUCTION GROUNDED RAG EVALUATION BENCHMARK ({total_questions} Questions)")
    print(f"================================================================================\n")

    for idx, item in enumerate(qa_pairs, start=1):
        q_id = item["id"]
        question = item["question"]
        is_answerable = item["is_answerable"]
        expected_source = item["expected_source"]
        keywords = item.get("keywords", [])

        # Execute query through full LangGraph pipeline
        conv_id = f"eval-session-{q_id}"
        t0 = time.perf_counter()
        result = execute_rag_pipeline(
            query=question,
            conversation_id=conv_id,
        )
        latency = (time.perf_counter() - t0) * 1000.0
        total_latency += latency

        final_answer: GroundedAnswer = result["final_answer"]
        can_answer = final_answer.can_answer
        confidence = final_answer.confidence
        cited_sources = [c.source for c in final_answer.citations]

        # 1. Evaluate Citation Accuracy
        citation_pass = False
        if is_answerable:
            if expected_source in cited_sources and len(final_answer.citations) > 0:
                citation_pass = True
                correct_citations += 1
        else:
            # For unanswerable, passing citation means no fabricated citations
            if len(final_answer.citations) == 0:
                citation_pass = True
                correct_citations += 1

        # 2. Evaluate Answer Correctness
        answer_pass = False
        if is_answerable:
            if can_answer:
                # Check keyword presence in answer text or snippets
                ans_lower = final_answer.answer.lower()
                matched_kw = sum(1 for kw in keywords if kw.lower() in ans_lower)
                if matched_kw >= max(1, len(keywords) // 2):
                    answer_pass = True
                    correct_answers += 1
                else:
                    # Partial credit if at least one snippet contains keywords
                    snippets_text = " ".join(c.quoted_snippet.lower() for c in final_answer.citations)
                    if any(kw.lower() in snippets_text for kw in keywords):
                        answer_pass = True
                        correct_answers += 1
        else:
            # Must correctly refuse
            if not can_answer:
                answer_pass = True
                correct_answers += 1

        # 3. Evaluate Refusal Accuracy
        if not is_answerable:
            if not can_answer:
                correct_refusals += 1

        status_str = "PASS" if (citation_pass and answer_pass) else "FAIL"
        short_q = (question[:42] + "..") if len(question) > 42 else question
        top_score = result["post_rerank_scores"][0] if result["post_rerank_scores"] else 0.0

        table_rows.append([
            q_id,
            short_q,
            "Yes" if is_answerable else "No",
            "Yes" if can_answer else "No",
            confidence.upper(),
            f"{top_score:.3f}",
            "PASS" if citation_pass else "FAIL",
            "PASS" if answer_pass else "FAIL",
            f"{latency:.0f}ms",
            status_str,
        ])

    # Summary calculations
    citation_accuracy = (correct_citations / total_questions) * 100.0
    overall_accuracy = (correct_answers / total_questions) * 100.0
    refusal_accuracy = (correct_refusals / max(1, unanswerable_count)) * 100.0
    avg_latency = total_latency / max(1, total_questions)

    headers = [
        "ID",
        "Question",
        "Target Ans",
        "Got Ans",
        "Conf",
        "Rerank",
        "Citation",
        "Correct",
        "Latency",
        "Status",
    ]
    print(tabulate(table_rows, headers=headers, tablefmt="github"))

    print("\n" + "=" * 80)
    print("                     EVALUATION REPORT SUMMARY")
    print("=" * 80)
    print(f" Total Evaluated Questions       : {total_questions}")
    print(f" Answerable Questions (in-scope) : {answerable_count}")
    print(f" Out-of-Scope (refusal targets)  : {unanswerable_count}")
    print(f"--------------------------------------------------------------------------------")
    print(f" Citation Accuracy               : {citation_accuracy:.1f}% ({correct_citations}/{total_questions})")
    print(f" Overall Answer Correctness      : {overall_accuracy:.1f}% ({correct_answers}/{total_questions})")
    print(f" Out-of-Scope Refusal Accuracy   : {refusal_accuracy:.1f}% ({correct_refusals}/{unanswerable_count})")
    print(f" Average Latency per Query       : {avg_latency:.1f}ms")
    print("=" * 80 + "\n")

    return {
        "total_questions": total_questions,
        "citation_accuracy": citation_accuracy,
        "overall_accuracy": overall_accuracy,
        "refusal_accuracy": refusal_accuracy,
        "avg_latency_ms": avg_latency,
    }


def main():
    """Main CLI entrypoint."""
    ensure_documents_ingested()
    dataset_file = Path(__file__).parent / "eval_dataset.json"
    results = evaluate_dataset(dataset_file)

    if results["citation_accuracy"] >= 70.0 and results["overall_accuracy"] >= 70.0:
        print("SUCCESS: Evaluation benchmarks passed quality criteria.")
        sys.exit(0)
    else:
        print("WARNING: Benchmarks did not meet 70% threshold.")
        sys.exit(1)


if __name__ == "__main__":
    main()
