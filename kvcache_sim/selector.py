from __future__ import annotations

import random
from typing import Protocol

from kvcache_sim.model import PrefillNode, Request


class PrefillSelector(Protocol):
    def select(self, request: Request, nodes: list[PrefillNode]) -> int:
        """Return node_id of the selected prefill node."""
        ...


def prefix_match_len(hash_ids: list[int], cache: dict[int, object]) -> int:
    """Return the number of contiguous prefix blocks in hash_ids that exist in cache."""
    count = 0
    for hid in hash_ids:
        if hid in cache:
            count += 1
        else:
            break
    return count


class RandomSelector:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select(self, request: Request, nodes: list[PrefillNode]) -> int:
        return self._rng.choice(nodes).node_id


class CacheAwareSelector:
    def select(self, request: Request, nodes: list[PrefillNode]) -> int:
        best_node_id = -1
        best_hit = -1
        best_used = float("inf")
        for node in nodes:
            hit = prefix_match_len(request.hash_ids, node.cache)
            if hit > best_hit or (hit == best_hit and node.used < best_used):
                best_node_id = node.node_id
                best_hit = hit
                best_used = node.used
        return best_node_id
