from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flux_router.evictor import FIFOEvictor, LRUEvictor
from flux_router.model import PrefillNode, Request
from flux_router.selector import CacheAwareSelector, RandomSelector
from flux_router.simulator import PrefillSimulator


def load_requests(path: str) -> list[Request]:
    requests: list[Request] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            requests.append(Request(
                timestamp=obj["timestamp"],
                input_length=obj["input_length"],
                output_length=obj["output_length"],
                hash_ids=obj["hash_ids"],
            ))
    return requests


def print_result(result, selector_name: str, evictor_name: str,
                 num_nodes: int, capacity: int, data_path: str) -> None:
    print("=== Flux Router Simulation ===")
    print(f"Data:        {data_path}")
    print(f"Requests:    {result.total_requests}")
    print(f"Nodes:       {num_nodes}  (capacity: {capacity} blocks each)")
    print(f"Selector:    {selector_name}")
    print(f"Evictor:     {evictor_name}")
    print()
    print("--- Results ---")
    print(f"Total blocks needed:  {result.total_blocks_needed}")
    print(f"Total blocks hit:     {result.total_blocks_hit}")
    print(f"Cache hit rate:       {result.cache_hit_rate:.2%}")
    print()

    hits = result.per_request_hits
    if not hits:
        return
    zero = sum(1 for h in hits if h == 0)
    low = sum(1 for h in hits if 1 <= h <= 5)
    mid = sum(1 for h in hits if 6 <= h <= 10)
    high = sum(1 for h in hits if h >= 11)
    total = len(hits)
    print("Per-request hit distribution:")
    print(f"  0 blocks hit:    {zero:>4} requests ({zero / total:.1%})")
    print(f"  1-5 blocks hit:  {low:>4} requests ({low / total:.1%})")
    print(f"  6-10 blocks hit: {mid:>4} requests ({mid / total:.1%})")
    print(f"  11+ blocks hit:  {high:>4} requests ({high / total:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Flux Router: prefill routing simulation")
    parser.add_argument("--data", required=True, help="Path to JSONL trace file")
    parser.add_argument("--nodes", type=int, default=8, help="Number of prefill nodes (default: 8)")
    parser.add_argument("--capacity", type=int, default=10000, help="Cache capacity per node in blocks (default: 10000)")
    parser.add_argument("--selector", choices=["random", "cache_aware"], default="random",
                        help="Node selection strategy (default: random)")
    parser.add_argument("--evictor", choices=["fifo", "lru"], default="fifo",
                        help="Block eviction strategy (default: fifo)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for RandomSelector (default: 42)")
    args = parser.parse_args()

    requests = load_requests(args.data)

    nodes = [PrefillNode(node_id=i, capacity=args.capacity) for i in range(args.nodes)]

    if args.selector == "random":
        selector = RandomSelector(seed=args.seed)
    else:
        selector = CacheAwareSelector()

    if args.evictor == "fifo":
        evictor = FIFOEvictor()
    else:
        evictor = LRUEvictor()

    sim = PrefillSimulator(nodes, selector, evictor)
    result = sim.run(requests)
    print_result(result, args.selector, args.evictor, args.nodes, args.capacity, args.data)


if __name__ == "__main__":
    main()
