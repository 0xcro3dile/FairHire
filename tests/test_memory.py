# tests for Memory
import os
import uuid

import pytest

os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

from fairhire.core.memory import Memory
from fairhire.core import memory as memory_module


@pytest.fixture
def memory() -> Memory:
    return Memory()


def test_store_and_recall_audit(memory: Memory) -> None:
    audit_id = f"unit-{uuid.uuid4()}"
    key = f"fairhire:audit:{audit_id}"
    payload = {"status": "complete", "report": "ok"}
    memory.store_audit(audit_id, payload)
    try:
        fetched = memory.recall_audit(audit_id)
        assert fetched == payload
    finally:
        memory.delete(key)


def test_store_audit_duplicate_raises(memory: Memory) -> None:
    audit_id = f"unit-{uuid.uuid4()}"
    key = f"fairhire:audit:{audit_id}"
    payload = {"status": "complete"}
    memory.store_audit(audit_id, payload)
    try:
        with pytest.raises(ValueError):
            memory.store_audit(audit_id, payload)
    finally:
        memory.delete(key)


def test_list_audit_ids_redis(memory: Memory) -> None:
    audit_ids = [f"unit-{uuid.uuid4()}", f"unit-{uuid.uuid4()}"]
    keys = [f"fairhire:audit:{audit_id}" for audit_id in audit_ids]
    for audit_id in audit_ids:
        memory.store_audit(audit_id, {"status": "complete"})
    try:
        listed, total = memory.list_audit_ids(limit=50, offset=0)
        assert total >= len(audit_ids)
        for audit_id in audit_ids:
            assert audit_id in listed
    finally:
        for key in keys:
            memory.delete(key)


def test_store_and_recall_batch(memory: Memory) -> None:
    key_one = f"fairhire:test:batch:{uuid.uuid4()}"
    key_two = f"fairhire:test:batch:{uuid.uuid4()}"
    missing = f"fairhire:test:batch:{uuid.uuid4()}"
    items = [(key_one, {"v": 1}), (key_two, {"v": 2})]
    memory.store_batch(items)
    try:
        results = memory.recall_batch([key_one, key_two, missing])
        assert results[key_one] == {"v": 1}
        assert results[key_two] == {"v": 2}
        assert results[missing] is None
    finally:
        memory.delete(key_one)
        memory.delete(key_two)


def test_batch_empty_noop(memory: Memory) -> None:
    memory.store_batch([])
    assert memory.recall_batch([]) == {}


def test_archive_status_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory_module, "POSTGRES_URL", "")
    assert Memory().archive_status() == "disabled"


def test_ping_redis(memory: Memory) -> None:
    assert memory.ping_redis() is True
