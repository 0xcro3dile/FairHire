# Redis session memory - 90 day TTL for audits
import json
import redis
from fairhire.config import REDIS_URL

class Memory:
  def __init__(self, redis_url: str = REDIS_URL):
    self.client = redis.from_url(redis_url, decode_responses=True)

  def store(self, key: str, data: dict, ttl: int = 86400 * 90) -> None:
    self.client.setex(key, ttl, json.dumps(data))

  def recall(self, key: str) -> dict | None:
    data = self.client.get(key)
    return json.loads(data) if data else None

  def delete(self, key: str) -> bool: return bool(self.client.delete(key))
  def list_keys(self, pattern: str = "fairhire:*") -> list[str]: return list(self.client.scan_iter(pattern))

  # audit-specific helpers
  def store_audit(self, audit_id: str, results: dict) -> None: self.store(f"fairhire:audit:{audit_id}", results)
  def recall_audit(self, audit_id: str) -> dict | None: return self.recall(f"fairhire:audit:{audit_id}")
