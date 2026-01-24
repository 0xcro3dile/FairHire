# FastAPI app entrypoint
# L20 compliant: strict types, structlog
from typing import Any, Callable, Protocol, TypeVar, cast

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fairhire.api.routes import router

_THandler = TypeVar("_THandler", bound=Callable[..., object])


class _TypedFastAPI(Protocol):
    def get(self, path: str, **kwargs: Any) -> Callable[[_THandler], _THandler]: ...
    def add_middleware(self, middleware_class: type[Any], **kwargs: Any) -> None: ...
    def include_router(self, router: object) -> None: ...


log = structlog.get_logger(__name__)

app = FastAPI(
    title="FairHire Auditor",
    version="0.1.0",
    description="AI Hiring Bias Detection API",
)
typed_app = cast(_TypedFastAPI, app)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@typed_app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    from fairhire.core.memory import Memory

    try:
        memory = Memory()
        redis_ok = memory.ping_redis()
        postgres_status = memory.archive_status()
        status = "healthy" if redis_ok and postgres_status != "error" else "degraded"
        log.debug("health_check", status=status, postgres=postgres_status)
        return {
            "status": status,
            "redis": "connected" if redis_ok else "error",
            "postgres": postgres_status,
        }
    except Exception as e:
        log.warning("health_check_degraded", error=str(e))
        return {"status": "degraded", "redis": f"error: {e}", "postgres": "error"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
