from __future__ import annotations

from typing import Protocol

from kvcache_sim.model import PrefillNode


class BlockEvictor(Protocol):
    def evict(self, node: PrefillNode, need: int, exclude: set[int] | None = None) -> list[int]:
        """Return hash_ids to evict to free `need` slots."""
        ...
    def on_access(self, node: PrefillNode, hash_id: int) -> None:
        ...


class FIFOEvictor:
    """Evict blocks with the oldest insert_time."""

    def evict(self, node: PrefillNode, need: int, exclude: set[int] | None = None) -> list[int]:
        evicted = []
        # the earliest entry is in front of cache.keys
        for hid in list(node.cache.keys()):
            if len(evicted) >= need:
                break
            if exclude and hid in exclude:
                continue
            del node.cache[hid]
        return evicted
    
    def on_access(self, node: PrefillNode, hash_id: int) -> None:
        pass    # FIFO：访问不影响顺序


class LRUEvictor:
    """Evict blocks with the oldest last_access_time."""

    def evict(self, node: PrefillNode, need: int, exclude: set[int] | None = None) -> list[int]:
        # pop the least recently accessed item (move_to_end when hit)
        evicted = []
        for hid in list(node.cache.keys()):
            if len(evicted) >= need:
                break
            if exclude and hid in exclude:
                continue
            del node.cache[hid]
        return evicted

    def on_access(self, node: PrefillNode, hash_id: int) -> None:
        node.cache.move_to_end(hash_id) # LRU：访问后的entry移到末尾（最近）
