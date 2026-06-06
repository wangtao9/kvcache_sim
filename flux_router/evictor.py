from __future__ import annotations

from typing import Protocol

from flux_router.model import PrefillNode


class BlockEvictor(Protocol):
    def evict(self, node: PrefillNode, need: int, now: int) -> list[int]:
        """Return hash_ids to evict to free `need` slots."""
        ...


class FIFOEvictor:
    """Evict blocks with the oldest insert_time."""

    def evict(self, node: PrefillNode, need: int, now: int) -> list[int]:
        sorted_items = sorted(node.cache.items(), key=lambda x: x[1].insert_time)
        return [hid for hid, _ in sorted_items[:need]]


class LRUEvictor:
    """Evict blocks with the oldest last_access_time."""

    def evict(self, node: PrefillNode, need: int, now: int) -> list[int]:
        sorted_items = sorted(node.cache.items(), key=lambda x: x[1].last_access_time)
        return [hid for hid, _ in sorted_items[:need]]
