import os, logging
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from slowapi.errors import RateLimitExceeded
from middleware.security import limiter
from routers import auth, questions, sessions, scoring, generator, anthropic, recordings, attempts, analytics

# --- Structured logging ---------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("pte.backend")

app = FastAPI(title="PTE API", version="1.1", docs_url="/docs")
app.state.limiter = limiter

# --- Uniform error envelope -------------------------------------------------
# frontend api.ts reads `err.detail` as a string → detail is ALWAYS a string.
# Additional machine-readable fields ride alongside in `code`/`errors`/`path`.
def _error(status: int, code: str, message: str, **extra) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"detail": message, "code": code, **extra},
    )

@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    logger.warning("validation error on %s: %s", request.url.path, exc.errors()[:3])
    return _error(
        422, "validation_error", "Invalid request",
        errors=jsonable_encoder(exc.errors())[:5],
    )

@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    logger.exception("unhandled error on %s", request.url.path)
    return _error(500, "internal_error", "Internal server error", path=request.url.path)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning("rate limited on %s", request.url.path)
    return _error(429, "rate_limited", "Too many requests - slow down and try again shortly.")

frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        frontend_url,
        "http://127.0.0.1:3000", "http://localhost:3000",
        "http://127.0.0.1:3001", "http://localhost:3001",
        "http://127.0.0.1:3030", "http://localhost:3030",
        "https://englis-edu-study.com", "https://www.englis-edu-study.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)

@app.middleware("http")
async def request_logging(request: Request, call_next):
    import time
    start = time.perf_counter()
    response = await call_next(request)
    logger.info(
        "%s %s -> %s (%.0fms)",
        request.method, request.url.path, response.status_code,
        (time.perf_counter() - start) * 1000,
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(questions.router, prefix="/questions", tags=["questions"])
app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
app.include_router(attempts.router, prefix="/attempts", tags=["attempts"])
app.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
app.include_router(scoring.router, prefix="/score", tags=["scoring"])
app.include_router(generator.router, prefix="/generator", tags=["generator"])
app.include_router(anthropic.router, prefix="/anthropic", tags=["anthropic"])
app.include_router(recordings.router, prefix="/recordings", tags=["recordings"])

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    """Readiness: confirms the process can reach Supabase (auth + DB are the
    only external dependencies). If Supabase is unreachable the load balancer
    should stop routing traffic here."""
    from middleware.security import get_supabase
    try:
        get_supabase()
        return {"status": "ready"}
    except Exception as e:
        logger.error("readiness probe failed: %s", e)
        from fastapi.responses import JSONResponse
        return JSONResponse({"status": "not_ready"}, status_code=503)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)