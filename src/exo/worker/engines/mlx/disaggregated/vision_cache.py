import threading
import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from loguru import logger


@dataclass
class _CacheEntry:
    embeddings: mx.array
    image_token_id: int
    created_at: float
    last_accessed: float
    approx_size_bytes: int


class VisionRegistryCache:
    """Thread-safe Content-Addressable Vision Embeddings Cache with LRU & TTL eviction."""

    def __init__(
        self,
        max_entries: int = 64,
        max_memory_mb: int = 2048,
        ttl_seconds: float = 1800.0,
    ) -> None:
        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._cache: dict[str, _CacheEntry] = {}
        self._hits = 0
        self._misses = 0

    def get(self, image_hash: str) -> tuple[mx.array, int] | None:
        """Fetch cached vision embeddings by image hash."""
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(image_hash)
            if entry is None:
                self._misses += 1
                return None

            # Check TTL expiration
            if now - entry.last_accessed > self.ttl_seconds:
                del self._cache[image_hash]
                self._misses += 1
                logger.debug(f"[VisionCache] TTL expired for hash {image_hash[:12]}")
                return None

            # Update access time (LRU)
            entry.last_accessed = now
            self._hits += 1
            logger.info(
                f"[VisionCache] Cache HIT for hash {image_hash[:12]} (hits={self._hits}, misses={self._misses})"
            )
            return entry.embeddings, entry.image_token_id

    def put(
        self,
        image_hash: str,
        embeddings: mx.array,
        image_token_id: int,
    ) -> None:
        """Store vision embeddings by image hash and evict stale/oldest entries."""
        now = time.monotonic()
        approx_bytes = int(embeddings.size * embeddings.itemsize)

        with self._lock:
            self._sweep_expired_locked(now)

            # Evict LRU if full
            while len(self._cache) >= self.max_entries:
                self._evict_oldest_locked()

            self._cache[image_hash] = _CacheEntry(
                embeddings=embeddings,
                image_token_id=image_token_id,
                created_at=now,
                last_accessed=now,
                approx_size_bytes=approx_bytes,
            )
            logger.info(
                f"[VisionCache] Cached embeddings for hash {image_hash[:12]} ({approx_bytes / 1024 / 1024:.2f} MB, total_entries={len(self._cache)})"
            )

    def _sweep_expired_locked(self, now: float) -> None:
        expired_keys = [
            k
            for k, entry in self._cache.items()
            if (now - entry.last_accessed) > self.ttl_seconds
        ]
        for k in expired_keys:
            del self._cache[k]
            logger.debug(f"[VisionCache] Evicted expired hash {k[:12]}")

    def _evict_oldest_locked(self) -> None:
        if not self._cache:
            return
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
        del self._cache[oldest_key]
        logger.info(f"[VisionCache] LRU evicted oldest hash {oldest_key[:12]}")

    def clear(self) -> None:
        """Explicitly clear the entire vision cache."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"[VisionCache] Cleared all {count} cached entries")

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total_bytes = sum(e.approx_size_bytes for e in self._cache.values())
            return {
                "entries": len(self._cache),
                "max_entries": self.max_entries,
                "memory_mb": round(total_bytes / 1024 / 1024, 2),
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": (
                    round(self._hits / (self._hits + self._misses), 3)
                    if (self._hits + self._misses) > 0
                    else 0.0
                ),
            }


# Global singleton instance for the worker prefill engine
global_vision_cache = VisionRegistryCache()
