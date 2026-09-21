"""Structured JSON logging for queries, retrieval metrics, and latency."""

import json
import logging
import sys
import time
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Custom formatter outputting logs as structured JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "structured_data") and isinstance(record.structured_data, dict):
            log_obj.update(record.structured_data)
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, default=str)


def setup_logger(name: str = "rag_system") -> logging.Logger:
    """Configure and return a structured JSON logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger("rag_system")


def log_query_execution(
    query: str,
    conversation_id: str,
    retrieved_chunk_ids: list,
    pre_rerank_scores: list,
    post_rerank_scores: list,
    can_answer: bool,
    confidence: str,
    latency_ms: float,
    validation_exhausted: bool = False,
    model_used: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Log structured telemetry data for a query execution."""
    data = {
        "event": "query_execution",
        "query": query,
        "conversation_id": conversation_id,
        "model_used": model_used or "unknown",
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "pre_rerank_scores": [round(float(s), 4) for s in pre_rerank_scores],
        "post_rerank_scores": [round(float(s), 4) for s in post_rerank_scores],
        "can_answer": can_answer,
        "confidence": confidence,
        "latency_ms": round(latency_ms, 2),
        "validation_exhausted": validation_exhausted,
    }
    if extra:
        data.update(extra)

    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        fn="",
        lno=0,
        msg=f"Processed query in {round(latency_ms, 2)}ms (model={model_used or 'unknown'}, can_answer={can_answer}, confidence={confidence})",
        args=(),
        exc_info=None,
    )
    record.structured_data = data  # type: ignore[attr-defined]
    logger.handle(record)
