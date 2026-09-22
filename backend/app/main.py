"""FastAPI application entrypoint with middleware, CORS, rate limiting, and routing."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from .api.routes import router
from .api.rate_limiter import limiter
from .config import get_settings
from .logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle initialization with strict memory bounding."""
    settings = get_settings()
    logger.info(
        f"Starting Grounded RAG Service on {settings.HOST}:{settings.PORT} "
        f"[LLM Provider: {settings.LLM_PROVIDER}, Embeddings: {settings.EMBEDDING_PROVIDER}]"
    )
    yield
    import gc
    gc.collect()
    logger.info("Shutting down Grounded RAG Service.")


app = FastAPI(
    title="Production-Grade Grounded RAG API",
    description=(
        "A reliability-first Retrieval-Augmented Generation API featuring hybrid dense+sparse retrieval, "
        "cross-encoder reranking, LangGraph state machine orchestration, strict citation guardrails, "
        "and multi-turn conversation memory."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Attach rate limiter state
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Clean JSON response for rate limit violations."""
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please throttle your requests."},
    )


# CORS Middleware setup supporting Vercel and local origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8000",
        "https://rag-orcin-sigma.vercel.app",
        "https://rag-5djx.onrender.com",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Register API routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
