# Redis session memory - 90 day TTL for audits
# L20 compliant: orjson, strict types, pipeline support
from typing import Any, cast

import orjson
import redis
import structlog

from fairhire.config import REDIS_URL

log = structlog.get_logger(__name__)


class Memory:
    def __init__(self, redis_url: str = REDIS_URL) -> None:
        self.client: redis.Redis = redis.Redis.from_url(
            redis_url, decode_responses=True
        )
        try:
            self.client.ping()
            log.info("redis_connected", url=redis_url)
        except redis.ConnectionError as e:
            log.error("redis_connection_failed", url=redis_url, error=str(e))
            raise RuntimeError(f"Redis unavailable at {redis_url}: {e}") from e

    def store(self, key: str, data: dict[str, Any], ttl: int = 86400 * 90) -> None:
        self.client.setex(
            key, ttl, orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")
        )

    def recall(self, key: str) -> dict[str, Any] | None:
        data = self.client.get(key)
        if data is None:
            return None
        # orjson.loads accepts str | bytes; data is str from decode_responses=True
        return cast(dict[str, Any], orjson.loads(str(data)))

    def delete(self, key: str) -> bool:
        return bool(self.client.delete(key))

    def list_keys(self, pattern: str = "fairhire:*") -> list[str]:
        return list(self.client.scan_iter(pattern))

    # Pipeline support for batch operations (L20: atomic ops)
    def store_batch(self, items: list[tuple[str, dict[str, Any]]], ttl: int = 86400 * 90) -> None:
        """Store multiple items atomically using Redis pipeline."""
        if not items:
            return
        pipe = self.client.pipeline()
        for key, data in items:
            pipe.setex(
                key, ttl, orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")
            )
        pipe.execute()
        log.info("batch_stored", count=len(items))

    def recall_batch(self, keys: list[str]) -> dict[str, dict[str, Any] | None]:
        """Recall multiple items using Redis pipeline."""
        if not keys:
            return {}
        pipe = self.client.pipeline()
        for key in keys:
            pipe.get(key)
        results = pipe.execute()
        return {
            key: orjson.loads(val) if val else None for key, val in zip(keys, results, strict=True)
        }

    # Audit-specific helpers
    def store_audit(self, audit_id: str, results: dict[str, Any]) -> None:
        key = f"fairhire:audit:{audit_id}"
        if self.client.exists(key):
            log.warning("audit_exists", audit_id=audit_id)
            raise ValueError(f"Audit {audit_id} already exists")
        self.store(key, results)
        log.info("audit_stored", audit_id=audit_id)

    def recall_audit(self, audit_id: str) -> dict[str, Any] | None:
        return self.recall(f"fairhire:audit:{audit_id}")
