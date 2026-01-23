# API route tests
# L20 compliant: real Redis via test containers, pytest-asyncio
import os
import pytest
from httpx import ASGITransport, AsyncClient

# Set test Redis URL before importing app
os.environ["REDIS_URL"] = os.environ.get("REDIS_URL", "redis://localhost:6379")

from fairhire.api.main import app


@pytest.fixture
def sample_csv_bytes() -> bytes:
  """Sample CSV data for testing."""
  return b"gender,experience,hired\n1,5,1\n1,3,1\n0,5,0\n0,3,0\n1,2,1\n0,4,0\n1,6,1\n0,2,0\n"


@pytest.mark.asyncio
async def test_run_audit_success(sample_csv_bytes: bytes) -> None:
  """Test successful audit run."""
  transport = ASGITransport(app=app)
  async with AsyncClient(transport=transport, base_url="http://test") as client:
    response = await client.post(
      "/audit",
      files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
    )
  assert response.status_code == 200
  data = response.json()
  assert "audit_id" in data
  assert data["status"] == "complete"
  assert "# FairHire Audit Report" in data["report"]


@pytest.mark.asyncio
async def test_run_audit_invalid_file_type() -> None:
  """Test audit with invalid file type."""
  transport = ASGITransport(app=app)
  async with AsyncClient(transport=transport, base_url="http://test") as client:
    response = await client.post(
      "/audit",
      files={"file": ("test.json", b"{}", "application/json")},
    )
  assert response.status_code == 400
  assert "CSV" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_audit_not_found() -> None:
  """Test getting non-existent audit."""
  transport = ASGITransport(app=app)
  async with AsyncClient(transport=transport, base_url="http://test") as client:
    response = await client.get("/audit/nonexistent-id")
  assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_audits() -> None:
  """Test listing audits with pagination."""
  transport = ASGITransport(app=app)
  async with AsyncClient(transport=transport, base_url="http://test") as client:
    response = await client.get("/audits?limit=10&offset=0")
  assert response.status_code == 200
  data = response.json()
  assert "audits" in data
  assert "total" in data
  assert data["limit"] == 10
  assert data["offset"] == 0


@pytest.mark.asyncio
async def test_list_audits_invalid_limit() -> None:
  """Test listing audits with invalid limit."""
  transport = ASGITransport(app=app)
  async with AsyncClient(transport=transport, base_url="http://test") as client:
    response = await client.get("/audits?limit=200")
  assert response.status_code == 400


@pytest.mark.asyncio
async def test_health_check() -> None:
  """Test health endpoint."""
  transport = ASGITransport(app=app)
  async with AsyncClient(transport=transport, base_url="http://test") as client:
    response = await client.get("/health")
  assert response.status_code == 200
  assert response.json()["status"] == "healthy"
