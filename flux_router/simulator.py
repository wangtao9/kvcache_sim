from __future__ import annotations

from dataclasses import dataclass, field

from flux_router.evictor import BlockEvictor
from flux_router.model import PrefillNode, Request
from flux_router.selector import PrefillSelector, prefix_match_len


@dataclass
class PerNodeStats:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class SimResult:
    total_requests: int
    total_blocks_needed: int
    total_blocks_hit: int
    total_evictions: int
    cache_hit_rate: float
    per_request_hits: list[int] = field(default_factory=list)
    per_node: dict[int, PerNodeStats] = field(default_factory=dict)


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

    def process_request(self, request: Request, now: int) -> tuple[int, int, int, int]:
        # process one request. return <node_id, hit_len, miss_count, eviction_count>
        node_id = self._selector.select(request, self._nodes)
        node = self._node_by_id[node_id]
        hit_len = prefix_match_len(request.hash_ids, node.cache)
        miss_count = 0
        eviction_count = 0

        for hid in request.hash_ids:
            if hid in node.cache:
                self._evictor.on_access(node, hid)
            else:
                if node.used >= node.capacity:
                    evicted = self._evictor.evict(node, need=1)
                    eviction_count += len(evicted)
                node.cache[hid] = now
                miss_count += 1

        return node_id, hit_len, miss_count, eviction_count

    def run(self, requests: list[Request]) -> SimResult:
        sorted_requests = sorted(requests, key=lambda r: r.timestamp)
        total_blocks_needed = 0
        total_blocks_hit = 0
        total_evictions = 0
        per_request_hits: list[int] = []
        per_node: dict[int, PerNodeStats] = {n.node_id: PerNodeStats() for n in self._nodes}

        for req in sorted_requests:
            total_blocks_needed += len(req.hash_ids)
            node_id, hit, miss, evict = self.process_request(req, now=req.timestamp)
            total_blocks_hit += hit
            total_evictions += evict
            per_request_hits.append(hit)
            per_node[node_id].requests += 1
            per_node[node_id].hits += hit
            per_node[node_id].misses += miss
            per_node[node_id].evictions += evict

        cache_hit_rate = total_blocks_hit / total_blocks_needed if total_blocks_needed > 0 else 0.0

        return SimResult(
            total_requests=len(requests),
            total_blocks_needed=total_blocks_needed,
            total_blocks_hit=total_blocks_hit,
            total_evictions=total_evictions,
            cache_hit_rate=cache_hit_rate,
            per_request_hits=per_request_hits,
            per_node=per_node,
        )
