# FastAPI app entrypoint
# L20 compliant: strict types, structlog
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fairhire.api.routes import router

log = structlog.get_logger(__name__)

app = FastAPI(
    title="FairHire Auditor",
    version="0.1.0",
    description="AI Hiring Bias Detection API",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    from fairhire.core.memory import Memory

    try:
        memory = Memory()
        memory.client.ping()
        log.debug("health_check", status="healthy")
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        log.warning("health_check_degraded", error=str(e))
        return {"status": "degraded", "redis": f"error: {e}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
