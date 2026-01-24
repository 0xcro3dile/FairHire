# Postgres archive tests (Tier 3 memory)
# L20 compliant: strict types, optional by env
import os
import uuid

import pytest

from fairhire.core.archive import PostgresArchive

POSTGRES_URL = os.environ.get("POSTGRES_URL", "")


@pytest.mark.skipif(not POSTGRES_URL, reason="POSTGRES_URL not set")
def test_postgres_archive_roundtrip() -> None:
    archive = PostgresArchive(POSTGRES_URL)
    audit_id = f"test-{uuid.uuid4()}"
    payload = {"status": "complete", "report": "ok"}
    archive.archive_audit(audit_id, payload)
    try:
        fetched = archive.fetch_audit(audit_id)
        assert fetched == payload
    finally:
        archive.delete_audit(audit_id)
