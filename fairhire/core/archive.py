# PostgreSQL audit archive (Tier 3 memory)
# L20 compliant: strict types, structlog, pooled connections
from __future__ import annotations

import json
from typing import Any, cast

import structlog
from psycopg.types.json import Json
from psycopg_pool import ConnectionPool

log = structlog.get_logger(__name__)


class PostgresArchive:
    def __init__(self, dsn: str, min_size: int = 1, max_size: int = 5) -> None:
        self.pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=10,
        )
        self._ensure_schema()
        log.info("postgres_archive_ready")

    def _ensure_schema(self) -> None:
        with self.pool.connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_archive (
                  audit_id TEXT PRIMARY KEY,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                  status TEXT NOT NULL,
                  payload JSONB NOT NULL
                )
                """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS audit_archive_created_at_idx
                ON audit_archive (created_at DESC)
                """)
            conn.commit()

    def archive_audit(self, audit_id: str, payload: dict[str, Any]) -> None:
        status = str(payload.get("status", "unknown"))
        with self.pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_archive (audit_id, status, payload)
                VALUES (%s, %s, %s)
                ON CONFLICT (audit_id) DO UPDATE
                SET status = EXCLUDED.status,
                    payload = EXCLUDED.payload
                """,
                (audit_id, status, Json(payload)),
            )
            conn.commit()

    def fetch_audit(self, audit_id: str) -> dict[str, Any] | None:
        with self.pool.connection() as conn:
            cursor = conn.execute(
                "SELECT payload FROM audit_archive WHERE audit_id = %s",
                (audit_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return self._coerce_payload(row[0])

    def list_audit_ids(self, limit: int, offset: int) -> tuple[list[str], int]:
        with self.pool.connection() as conn:
            total_row = conn.execute("SELECT count(*) FROM audit_archive").fetchone()
            total = int(total_row[0]) if total_row else 0
            cursor = conn.execute(
                """
                SELECT audit_id
                FROM audit_archive
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            rows = cursor.fetchall()
        return [row[0] for row in rows], total

    def delete_audit(self, audit_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("DELETE FROM audit_archive WHERE audit_id = %s", (audit_id,))
            conn.commit()

    def ping(self) -> bool:
        try:
            with self.pool.connection() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            log.warning("postgres_ping_failed", error=str(e))
            return False

    @staticmethod
    def _coerce_payload(payload: object) -> dict[str, Any]:
        if isinstance(payload, dict):
            return cast(dict[str, Any], payload)
        if isinstance(payload, (bytes, bytearray)):
            return cast(dict[str, Any], json.loads(payload.decode("utf-8")))
        if isinstance(payload, str):
            return cast(dict[str, Any], json.loads(payload))
        raise TypeError(f"Unexpected payload type: {type(payload)}")
