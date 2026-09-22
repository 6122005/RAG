# Production-Grade Grounded RAG System
### LangChain • LangGraph • FastAPI • ChromaDB • Cross-Encoder • React.js • Docker

A reliability-first Retrieval-Augmented Generation (RAG) system engineered for high-stakes enterprise applications where hallucinated answers are unacceptable. It enforces strict grounding exclusively in retrieved context, provides exact source citations for every factual claim, detects out-of-scope questions and deterministically refuses them, maintains multi-turn conversation memory, and includes a repeatable scoring evaluation benchmark.

---

## 🌐 Live Deployments & Demo

| Component | Platform | Live URL | Status |
| :--- | :--- | :--- | :--- |
| **Frontend Web App** | Vercel | [https://rag-orcin-sigma.vercel.app](https://rag-orcin-sigma.vercel.app) | [![Vercel](https://img.shields.io/badge/Vercel-Live-brightgreen?logo=vercel)](https://rag-orcin-sigma.vercel.app) |
| **Backend API** | Render | [https://rag-5djx.onrender.com](https://rag-5djx.onrender.com) | [![Render](https://img.shields.io/badge/Render-Live-brightgreen?logo=render)](https://rag-5djx.onrender.com) |
| **Interactive API Docs** | Swagger / OpenAPI | [https://rag-5djx.onrender.com/docs](https://rag-5djx.onrender.com/docs) | [![Swagger](https://img.shields.io/badge/Swagger-Docs-blue?logo=swagger)](https://rag-5djx.onrender.com/docs) |
| **Health Check Endpoint** | Uptime & Telemetry | [https://rag-5djx.onrender.com/health](https://rag-5djx.onrender.com/health) | [![Health](https://img.shields.io/badge/Health-200%20OK-success)](https://rag-5djx.onrender.com/health) |

> [!TIP]
> **Demo API Key**: Production endpoints are secured with constant-time API authentication. For testing via curl or Postman, include: `X-API-Key: rag-secret-key-prod-2026`.

---

## Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │              React.js Frontend               │
                      │   - Expandable Citation Cards & Quotes       │
                      │   - Color-Coded Confidence Badges            │
                      │   - Distinct Refusal State UI                │
                      │   - Document Management & Ingestion Dropzone │
                      │   - Multi-Turn Session Persistence           │
                      └──────────────────────┬───────────────────────┘
                                             │ HTTP + X-API-Key
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │             FastAPI Gateway                  │
                      │   - Constant-time X-API-Key authentication   │
                      │   - Token-bucket Rate Limiting (slowapi)     │
                      │   - Query input sanitization & log telemetry │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                   ┌─────────────────────────────────────────────────────┐
                   │               LangGraph State Machine               │
                   │        (Checkpointer MemorySaver by thread_id)      │
                   │                                                     │
                   │   ┌───────────────┐        ┌──────────────────┐     │
                   │   │ retrieve_node ├───────►│   rerank_node    │     │
                   │   │ (Dense+BM25)  │        │  (Cross-Encoder) │     │
                   │   └───────────────┘        └────────┬─────────┘     │
                   │                                     │               │
                   │                         [Score >= Threshold?]       │
                   │                                  /     \            │
                   │                                Yes      No          │
                   │                                /         \          │
                   │                               ▼           ▼         │
                   │                     ┌───────────┐  ┌────────────┐   │
                   │                     │ generate  │  │  refusal   │   │
                   │                     │ (+History)│  │ (Low conf) │   │
                   │                     └─────┬─────┘  └─────┬──────┘   │
                   │                           │              │          │
                   │                   [Valid Citations?]     │          │
                   │                      /         \         │          │
                   │                    Yes     No (Retry < 1)│          │
                   │                    /             \       │          │
                   │                   ▼               ▼      │          │
                   │             ┌───────────┐   ┌─────────┐  │          │
                   │             │format_out │◄──┤ retry   │  │          │
                   │             └─────┬─────┘   └────┬────┘  │          │
                   │                   │       Retry Failed   │          │
                   │                   │              ▼       │          │
                   │                   │         ┌─────────┐  │          │
                   │                   │         │exhausted│  │          │
                   │                   │         │fallback │  │          │
                   │                   │         └────┬────┘  │          │
                   │                   │              │       │          │
                   │                   └──────────►◄──┴───────┘          │
                   │                               │                     │
                   │                              END                    │
                   └─────────────────────────────────────────────────────┘
```

---

## Key Features

1. **Hybrid Retrieval (Dense + Sparse)**:
   Combines ChromaDB vector search (`sentence-transformers/all-MiniLM-L6-v2`) with BM25 keyword search using Reciprocal Rank Fusion (RRF) to capture both semantic meaning and exact keyword matches (e.g. acronyms, error codes, specific product tiers).
2. **Cross-Encoder Reranking**:
   Top-10 hybrid candidates are re-scored using `cross-encoder/ms-marco-MiniLM-L-6-v2`. Cross-encoders evaluate full query-passage attention, filtering down to the top 3-4 truly relevant chunks.
3. **Pydantic Structured Output & Provenance Citations**:
   The LLM output is strictly constrained to a Pydantic schema (`GroundedAnswer`):
   ```json
   {
     "answer": "...",
     "citations": [
       {"source": "acme_knowledge_base.md", "page": 1, "chunk_id": "acme_p1_c0", "quoted_snippet": "..."}
     ],
     "confidence": "high | medium | low",
     "can_answer": true | false
   }
   ```
4. **Dual Guardrails with Exhaustion Fallback**:
   - **Pre-generation check**: If the top rerank score is below `RERANK_THRESHOLD` (e.g., 0.10), generation is bypassed, returning an immediate refusal.
   - **Post-generation validator**: Verifies that every cited `chunk_id` actually exists in the retrieved set. Rejects fabricated citations, permits one self-correction retry, and if validation fails twice, transitions to an `exhausted_fallback` refusal with structured telemetry logging.
5. **Multi-Turn Conversation Memory**:
   LangGraph `MemorySaver` checkpointer keyed by `conversation_id`. Prior turns are supplied to the generation node strictly for pronoun/reference resolution without compromising document grounding.
6. **Constant-Time Authentication & Rate Limiting**:
   Requires `X-API-Key` on `/query` and `/ingest`, verified with `secrets.compare_digest` to prevent timing attacks. Rate-limited via `slowapi`.
7. **Production Dockerization**:
   Multi-container orchestration with Docker Compose and persistent named volume for ChromaDB vectors.
8. **Repeatable Evaluation Benchmark (`eval.py`)**:
   Scores 18 realistic questions (factual single-chunk, multi-chunk synthesis, and out-of-scope queries) and outputs citation accuracy, answer correctness, and refusal accuracy.

---

## Quick Start: One-Command Docker Setup

The fastest way to launch the full system is with Docker Compose:

```bash
# 1. Clone or navigate to the repository
cd d:/RAG

# 2. (Optional) Provide your Gemini API Key in .env
echo "GEMINI_API_KEY=your_actual_gemini_key_here" > backend/.env

# 3. Launch backend, ChromaDB volume, and frontend
docker compose up --build
```

- **Frontend Chat UI**: [http://localhost:3000](http://localhost:3000)
- **FastAPI Backend Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)

---

## Local Development Setup

If running locally without Docker:

### 1. Backend Setup

```bash
cd backend

# Create virtual environment and activate
uv venv .venv
.venv\Scripts\activate      # Windows (or source .venv/bin/activate on Linux/macOS)

# Install requirements
uv pip install -r requirements.txt

# Configure environment
copy .env.example .env

# Run FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup

```bash
cd frontend

# Install npm dependencies
npm install

# Start Vite dev server
npm run dev
```

Visit [http://localhost:5173](http://localhost:5173) in your browser.

---

## Running Automated Tests

The test suite covers chunking boundaries, cross-encoder reranker reordering, guardrail citation validation, and FastAPI route authentication using `TestClient`:

```bash
cd backend
.venv\Scripts\pytest tests/ -v
```

All 16 tests verify:
- `test_chunker.py`: Chunk size limits, overlap integrity, and deterministic `chunk_id` generation.
- `test_reranker.py`: Candidate reordering based on cross-encoder logits and threshold cutoff.
- `test_guardrails.py`: Rejection of hallucinated chunk IDs, input sanitization, and double-failure fallback.
- `test_api.py`: `X-API-Key` 401 unauthorized rejection, `/health`, `/ingest` file upload, and `/query` JSON shape.

---

## Running the Evaluation Benchmark

Run the standalone evaluation harness:

```bash
cd backend
.venv\Scripts\python eval.py
```

The script automatically ingests `sample_docs/acme_knowledge_base.md` into ChromaDB and evaluates 18 questions from `eval_dataset.json`.

### Benchmark Results

```
================================================================================
   STARTING PRODUCTION GROUNDED RAG EVALUATION BENCHMARK (18 Questions)
================================================================================
| ID   | Question                                     | Target Ans   | Got Ans   | Conf   |   Rerank | Citation   | Correct   | Latency   | Status   |
|------|----------------------------------------------|--------------|-----------|--------|----------|------------|-----------|-----------|----------|
| q01  | What is the monthly price of the Acme Prof.. | Yes          | Yes       | HIGH   |    1     | PASS       | PASS      | 885ms     | PASS     |
| q02  | How much storage is included in the Acme D.. | Yes          | Yes       | HIGH   |    0.999 | PASS       | PASS      | 577ms     | PASS     |
| q03  | What is the guaranteed support response ti.. | Yes          | Yes       | HIGH   |    0.973 | PASS       | PASS      | 487ms     | PASS     |
...
| q16  | What is the weather forecast for Mars colo.. | No           | No        | LOW    |    0     | PASS       | PASS      | 512ms     | PASS     |
| q17  | How do you configure quantum entanglement .. | No           | No        | LOW    |    0     | PASS       | PASS      | 516ms     | PASS     |
| q18  | What are the winning lottery numbers for t.. | No           | No        | LOW    |    0     | PASS       | PASS      | 548ms     | PASS     |

================================================================================
                     EVALUATION REPORT SUMMARY
================================================================================
 Total Evaluated Questions       : 18
 Answerable Questions (in-scope) : 15
 Out-of-Scope (refusal targets)  : 3
--------------------------------------------------------------------------------
 Citation Accuracy               : 100.0% (18/18)
 Overall Answer Correctness      : 100.0% (18/18)
 Out-of-Scope Refusal Accuracy   : 100.0% (3/3)
 Average Latency per Query       : ~550ms
================================================================================
SUCCESS: Evaluation benchmarks passed quality criteria.
```

---

## Design Decisions (Why These Choices Were Made)

### 1. Hybrid Retrieval (Dense Vector + BM25 via Reciprocal Rank Fusion)
* **The Problem**: Pure dense vector search excels at high-level semantic intent but struggles with exact domain terminology, model numbers, alphanumeric codes, and policy names. Pure BM25 keyword search catches exact words but fails when users phrase questions using synonyms.
* **The Decision**: Combine dense ChromaDB vector search with BM25 Okapi sparse search, fused via Reciprocal Rank Fusion (RRF: $\sum \frac{1}{60 + r}$). RRF is scale-invariant, eliminating the need to calibrate arbitrary score distributions between cosine distance and BM25 scores.

### 2. Cross-Encoder Reranking
* **The Problem**: Bi-encoders (vector similarity) encode query and passage into isolated embedding vectors. While fast for top-$k$ retrieval over millions of records, bi-encoders miss subtle cross-attentions between the query and passage text.
* **The Decision**: Retrieve top-$k$ ($k=10$) with hybrid search, then apply a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to joint `(query, passage)` pairs to select the top-$n$ ($n=3\text{--}4$). Cross-encoders perform full transformer cross-attention, accurately surfacing the most contextually relevant chunks while pruning distracting noise.

### 3. Pydantic Structured Output & Citation Verification Loop
* **The Problem**: Unstructured text generation with "please cite your sources" frequently produces fabricated or vague citations ("as mentioned in Section 1").
* **The Decision**: Enforce schema output via LangChain's `with_structured_output(GroundedAnswer)`. Every factual sentence maps to a `Citation` with exact `source`, `page`, `chunk_id`, and `quoted_snippet`. A post-generation validator verifies that every cited `chunk_id` belongs to the retrieved set. If an LLM hallucinates an ID, it is caught immediately and fed back for self-correction.

### 4. Multi-Turn Conversation Memory via LangGraph Checkpointing
* **The Problem**: RAG systems that maintain chat history often suffer from "context drift" or conversational hallucination: the LLM starts relying on claims it made in prior turns instead of grounding each turn in retrieved documents.
* **The Decision**: Use LangGraph's `MemorySaver` keyed by `conversation_id`. The prompt explicitly instructs the LLM that prior turns exist **only** for entity/pronoun resolution (e.g., "what about the Enterprise tier?"), while every factual claim about the domain must trace back to the freshly retrieved context chunks for that turn.

### 5. Deterministic Guardrails & Exhaustion Fallback
* **The Problem**: When retrieval returns low-confidence passages, ungrounded LLMs tend to answer from general pre-trained knowledge. Furthermore, retry loops that fail repeatedly can throw unhandled exceptions or loop indefinitely.
* **The Decision**:
  - *Pre-generation*: If top rerank score $< 0.10$, skip generation completely and return `can_answer: false`.
  - *Post-generation*: Allow exactly one retry if citation validation fails. If the second attempt still fails, invoke `exhausted_fallback_node`, setting `can_answer: false`, `confidence: "low"`, and logging `validation_exhausted: true` in structured JSON telemetry.

### 6. Constant-Time Authentication & Containerization
* **The Problem**: Standard string comparisons (`==`) in API key validation leak timing information through early exits. In addition, local vector DB deployments can lose state if volumes are not isolated.
* **The Decision**: Authenticate requests with `secrets.compare_digest` in FastAPI dependency injection, and containerize both frontend and backend using Docker Compose with a persistent named volume for ChromaDB data.

---

## Manual Verification Checklist (What to Test First)

1. **Verify Public Health Check**:
   Open [http://localhost:8000/health](http://localhost:8000/health) and confirm `"status": "healthy"`, indexed chunk counts, and active providers.
2. **Verify Document Ingestion**:
   In the React frontend sidebar, click **Upload Document** and select `backend/sample_docs/acme_knowledge_base.md`. Confirm the sidebar updates with the document name, chunk count, and page count.
3. **Test In-Scope Factual Query with Citations**:
   Ask: *"What is the monthly price of the Acme Professional Tier when billed annually?"*
   - Verify the answer states **$49 per user per month**.
   - Verify the **High Confidence** green badge appears.
   - Click the expandable **Citation Card** to inspect the source filename, page number, chunk ID, and quoted excerpt.
4. **Test Multi-Turn Follow-Up Query**:
   Immediately ask: *"What about the Enterprise Tier?"*
   - Verify the system resolves the pronoun reference to pricing and answers with **$499 per month plus $25 per user**, citing the correct section.
5. **Test Out-of-Scope Grounded Refusal**:
   Ask: *"What is the weather forecast for Mars colony base tomorrow?"*
   - Verify the **Refusal Guardrail Triggered** banner appears in red/amber.
   - Confirm `can_answer: false` and the model explains that the knowledge base lacks information on this topic, with zero fabricated citations.
6. **Test API Authentication Security**:
   Send a curl request without the API key header:
   ```bash
   curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"query\":\"test\"}"
   ```
   Confirm the server returns `HTTP 401 Unauthorized` with `{"detail": "Missing required 'X-API-Key' header."}`.
