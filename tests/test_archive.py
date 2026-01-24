# Postgres archive tests (Tier 3 memory)
# L20 compliant: strict types, optional by env
import json
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


def test_coerce_payload_dict() -> None:
    payload = {"status": "ok", "report": "done"}
    assert PostgresArchive._coerce_payload(payload) == payload


def test_coerce_payload_bytes_and_str() -> None:
    payload = {"status": "ok"}
    encoded = json.dumps(payload).encode("utf-8")
    assert PostgresArchive._coerce_payload(encoded) == payload
    assert PostgresArchive._coerce_payload(json.dumps(payload)) == payload


def test_coerce_payload_invalid_type() -> None:
    with pytest.raises(TypeError):
        PostgresArchive._coerce_payload(123)
