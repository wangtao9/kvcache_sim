from __future__ import annotations

from dataclasses import dataclass, field

from flux_router import LRUEvictor
from flux_router.evictor import BlockEvictor
from flux_router.model import PrefillNode, Request
from flux_router.selector import PrefillSelector, prefix_match_len


@dataclass
class SimResult:
    total_requests: int
    total_blocks_needed: int
    total_blocks_hit: int
    cache_hit_rate: float
    per_request_hits: list[int] = field(default_factory=list)


class PrefillSimulator:
    def __init__(
        self,
        nodes: list[PrefillNode],
        selector: PrefillSelector,
        evictor: BlockEvictor,
    ) -> None:
        self._nodes = nodes
        self._selector = selector
        self._evictor = evictor
        self._node_by_id = {n.node_id: n for n in nodes}

    def process_request(self, request: Request, now: int) -> int:
        node_id = self._selector.select(request, self._nodes)
        node = self._node_by_id[node_id]
        hit_len = prefix_match_len(request.hash_ids, node.cache)

        for hid in request.hash_ids:
            if hid in node.cache:
                self._evictor.on_access(node, hid)
            else:
                if node.used >= node.capacity:
                    self._evictor.evict(node, need=1)
            node.cache[hid] = now

        return hit_len

    def run(self, requests: list[Request]) -> SimResult:
        sorted_requests = sorted(requests, key=lambda r: r.timestamp)
        total_blocks_needed = 0
        total_blocks_hit = 0
        per_request_hits: list[int] = []

        for req in sorted_requests:
            blocks_needed = len(req.hash_ids)
            total_blocks_needed += blocks_needed
            hit = self.process_request(req, now=req.timestamp)
            total_blocks_hit += hit
            per_request_hits.append(hit)

        cache_hit_rate = total_blocks_hit / total_blocks_needed if total_blocks_needed > 0 else 0.0

        return SimResult(
            total_requests=len(requests),
            total_blocks_needed=total_blocks_needed,
            total_blocks_hit=total_blocks_hit,
            cache_hit_rate=cache_hit_rate,
            per_request_hits=per_request_hits,
        )
