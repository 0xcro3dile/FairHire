# FastAPI routes for bias audit
import uuid, tempfile, os, logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from fairhire.core.orchestrator import Orchestrator
from fairhire.core.memory import Memory

router = APIRouter()
memory = Memory()
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB limit

class AuditRequest(BaseModel):
  protected_attrs: list[str] = ["gender"]
  privileged_groups: list[dict] = [{"gender": 1}]
  unprivileged_groups: list[dict] = [{"gender": 0}]
  label_col: str = "hired"

@router.post("/audit")
async def run_audit(file: UploadFile = File(...), request: AuditRequest = None):
  if request is None: request = AuditRequest()
  
  # Validate file type
  if file.content_type != "text/csv":
    raise HTTPException(400, "Only CSV files allowed")
  
  # Validate file size
  contents = await file.read(MAX_FILE_SIZE + 1)
  if len(contents) > MAX_FILE_SIZE:
    raise HTTPException(413, "File too large (max 100MB)")
  
  audit_id = str(uuid.uuid4())
  with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    tmp.write(contents)
    tmp_path = tmp.name
  try:
    result = Orchestrator().run_audit(tmp_path, request.protected_attrs, request.privileged_groups, request.unprivileged_groups, request.label_col)
    memory.store_audit(audit_id, result)
    return {"audit_id": audit_id, "status": result["status"], "report": result["report"]}
  finally:
    try:
      os.unlink(tmp_path)
    except OSError as e:
      logger.error(f"Failed to cleanup {tmp_path}: {e}")

@router.get("/audit/{audit_id}")
def get_audit(audit_id: str):
  result = memory.recall_audit(audit_id)
  if not result: raise HTTPException(404, "Audit not found")
  return result

@router.get("/audits")
def list_audits():
  return {"audits": [k.replace("fairhire:audit:", "") for k in memory.list_keys("fairhire:audit:*")]}
