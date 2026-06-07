from __future__ import annotations

from typing import Protocol

from flux_router.model import PrefillNode


class BlockEvictor(Protocol):
    def evict(self, node: PrefillNode, need: int) -> list[int]:
        """Return hash_ids to evict to free `need` slots."""
        ...
    def on_access(self, node: PrefillNode, hash_id: int) -> None:
        ...


class FIFOEvictor:
    """Evict blocks with the oldest insert_time."""

    def evict(self, node: PrefillNode, need: int) -> list[int]:
        evicted = []
        for _ in range(need):
            # pop the earliest item
            hash_id, _ = node.cache.popitem(last=False)
            evicted.append(hash_id)
        return evicted
    
    def on_access(self, node: PrefillNode, hash_id: int) -> None:
        pass    # FIFO：访问不影响顺序


class LRUEvictor:
    """Evict blocks with the oldest last_access_time."""

    def evict(self, node: PrefillNode, need: int) -> list[int]:
        evicted = []
        for _ in range(need):
            # pop the least recently accessed item (move_to_end when hit)
            hash_id, _ = node.cache.popitem(last=False)
            evicted.append(hash_id)
        return evicted
    
    def on_access(self, node: PrefillNode, hash_id: int) -> None:
        node.cache.move_to_end(hash_id) # LRU：访问后的entry移到末尾（最近）
