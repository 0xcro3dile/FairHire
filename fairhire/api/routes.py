# FastAPI routes for bias audit
# L20 compliant: strict types, structlog, proper error handling
import os
import tempfile
import uuid
from typing import Any, Callable, Protocol, TYPE_CHECKING, TypeVar, cast

import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile

# Type-checking stubs keep strict mypy green when runtime deps are missing.
if TYPE_CHECKING:
    class BaseModel:
        def __init__(self, **data: Any) -> None:
            ...

    def Field(*args: Any, **kwargs: Any) -> Any: ...
else:
    from pydantic import BaseModel, Field

from fairhire.core.memory import Memory
from fairhire.core.orchestrator import Orchestrator

_THandler = TypeVar("_THandler", bound=Callable[..., object])


class _TypedRouter(Protocol):
    def post(self, path: str, **kwargs: Any) -> Callable[[_THandler], _THandler]: ...
    def get(self, path: str, **kwargs: Any) -> Callable[[_THandler], _THandler]: ...


router = APIRouter()
typed_router = cast(_TypedRouter, router)
memory = Memory()
log = structlog.get_logger(__name__)

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB limit


class AuditRequest(BaseModel):
    """Request schema for audit endpoint."""

    protected_attrs: list[str] = Field(default=["gender"], min_length=1)
    privileged_groups: list[dict[str, Any]] = Field(default=[{"gender": 1}], min_length=1)
    unprivileged_groups: list[dict[str, Any]] = Field(default=[{"gender": 0}], min_length=1)
    label_col: str = Field(default="hired", min_length=1)


class AuditResponse(BaseModel):
    """Response schema for audit endpoint."""

    audit_id: str
    status: str
    report: str


class AuditListResponse(BaseModel):
    """Response schema for list audits endpoint."""

    audits: list[str]
    total: int
    limit: int
    offset: int


@typed_router.post("/audit", response_model=AuditResponse)
async def run_audit(
    file: UploadFile = File(...),
    request: AuditRequest | None = None,
) -> AuditResponse:
    if request is None:
        request = AuditRequest()

    # Validate file type
    if file.content_type != "text/csv":
        log.warning("invalid_file_type", content_type=file.content_type)
        raise HTTPException(400, "Only CSV files allowed")

    # Validate file size
    contents = await file.read(MAX_FILE_SIZE + 1)
    if len(contents) > MAX_FILE_SIZE:
        log.warning("file_too_large", size=len(contents))
        raise HTTPException(413, "File too large (max 100MB)")

    audit_id = str(uuid.uuid4())
    log.info("audit_start", audit_id=audit_id, file=file.filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = Orchestrator().run_audit(
            tmp_path,
            request.protected_attrs,
            request.privileged_groups,
            request.unprivileged_groups,
            request.label_col,
        )
        # Filter out non-serializable fields
        serializable_result = {
            k: v for k, v in result.items() if k not in ("df", "model_predict_fn") and v is not None
        }
        memory.store_audit(audit_id, serializable_result)
        log.info("audit_complete", audit_id=audit_id, status=result.get("status"))
        return AuditResponse(
            audit_id=audit_id,
            status=result.get("status", "unknown"),
            report=result.get("report", ""),
        )
    except Exception as e:
        log.error("audit_failed", audit_id=audit_id, error=str(e))
        raise HTTPException(500, f"Audit failed: {e}") from e
    finally:
        try:
            os.unlink(tmp_path)
        except OSError as e:
            log.error("cleanup_failed", path=tmp_path, error=str(e))


@typed_router.get("/audit/{audit_id}")
def get_audit(audit_id: str) -> dict[str, Any]:
    result = memory.recall_audit(audit_id)
    if not result:
        log.warning("audit_not_found", audit_id=audit_id)
        raise HTTPException(404, "Audit not found")
    return result


@typed_router.get("/audits", response_model=AuditListResponse)
def list_audits(limit: int = 50, offset: int = 0) -> AuditListResponse:
    if limit > 100 or limit < 1:
        raise HTTPException(400, "Limit must be between 1 and 100")
    if offset < 0:
        raise HTTPException(400, "Offset must be non-negative")

    all_keys = list(memory.list_keys("fairhire:audit:*"))
    paginated = all_keys[offset : offset + limit]
    return AuditListResponse(
        audits=[k.replace("fairhire:audit:", "") for k in paginated],
        total=len(all_keys),
        limit=limit,
        offset=offset,
    )
