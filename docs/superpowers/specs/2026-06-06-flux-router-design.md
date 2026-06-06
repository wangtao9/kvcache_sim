# Flux Router v1 Design: Offline Prefill Routing Simulation

## Goal

Build an offline simulation that routes LLM prefill requests to N GPU nodes based on KVCache block hits, measures overall cache hit rate, and provides a pluggable abstraction for node selection and block eviction strategies.

## Scope (v1)

- Pure Python offline simulation, no network/GPU/real inference
- Input: Mooncake trace JSONL (`timestamp`, `input_length`, `output_length`, `hash_ids`)
- Output: overall cache hit rate + per-request hit distribution
- Two selector implementations: `RandomSelector`, `CacheAwareSelector`
- Two evictor implementations: `FIFOEvictor`, `LRUEvictor`
- Configurable node count and per-node capacity

Out of scope: TTFT estimation, decoding simulation, production HTTP service, optimization insights (v2), concurrent router architecture (v3).

## Project Structure

```
flux_router/
├── __init__.py
├── model.py          # Request, BlockEntry, PrefillNode dataclasses
├── selector.py       # PrefillSelector protocol + RandomSelector, CacheAwareSelector
├── evictor.py        # BlockEvictor protocol + FIFOEvictor, LRUEvictor
├── simulator.py      # PrefillSimulator + SimResult
└── __main__.py       # CLI entry point
```

## Data Model (`model.py`)

```python
@dataclass
class Request:
    timestamp: int          # arrival time in ms
    input_length: int       # input token count
    output_length: int      # output token count
    hash_ids: list[int]     # prefix block hash IDs (ordered, chained)

@dataclass
class BlockEntry:
    insert_time: int        # timestamp when block was first cached
    last_access_time: int   # timestamp of most recent access (hit or insert)

@dataclass
class PrefillNode:
    node_id: int
    capacity: int                        # max cached blocks
    cache: dict[int, BlockEntry]         # hash_id → BlockEntry

    @property
    def used(self) -> int:
        return len(self.cache)

    @property
    def available(self) -> int:
        return self.capacity - self.used
```

Key: `BlockEntry` stores both `insert_time` (for FIFO) and `last_access_time` (for LRU), so the simulator code has no branching per eviction strategy.

## Node Selector (`selector.py`)

```python
class PrefillSelector(Protocol):
    def select(self, request: Request, nodes: list[PrefillNode]) -> int:
        """Return node_id of the selected prefill node."""
        ...
```

### RandomSelector

Picks a random node. Baseline for comparison.

### CacheAwareSelector

Picks the node with the longest prefix match. Ties broken by lowest `node.used` (least loaded).

Prefix match is **contiguous from the start**: walk `request.hash_ids` left to right, stop at the first `hash_id` not in `node.cache`. This is correct because chained hashing guarantees that hash_id[i] matching implies hash_id[0..i-1] also match.

## Block Evictor (`evictor.py`)

```python
class BlockEvictor(Protocol):
    def evict(self, node: PrefillNode, need: int, now: int) -> list[int]:
        """Return hash_ids to evict to free `need` slots."""
        ...
```

### FIFOEvictor

Evicts blocks with the oldest `insert_time`.

### LRUEvictor

Evicts blocks with the oldest `last_access_time`.

## Simulator (`simulator.py`)

```python
@dataclass
class SimResult:
    total_requests: int
    total_blocks_needed: int       # sum of len(hash_ids) across all requests
    total_blocks_hit: int          # sum of prefix match lengths
    cache_hit_rate: float          # total_blocks_hit / total_blocks_needed
    per_request_hits: list[int]    # hit count per request (for distribution analysis)

class PrefillSimulator:
    def __init__(self, nodes: list[PrefillNode], selector: PrefillSelector, evictor: BlockEvictor): ...

    def process_request(self, request: Request, now: int) -> int:
        """Process one request. Returns prefix hit count."""
        # 1. node_id = selector.select(request, nodes)
        # 2. hit_len = prefix_match_len(request.hash_ids, node.cache)
        # 3. For each hid in request.hash_ids:
        #      if hid in node.cache:
        #          node.cache[hid].last_access_time = now   # update access
        #      else:
        #          if node.used == node.capacity:
        #              evict_ids = evictor.evict(node, need=1, now=now)
        #              del node.cache[evict_ids[0]]
        #          node.cache[hid] = BlockEntry(insert_time=now, last_access_time=now)
        # 4. return hit_len

    def run(self, requests: list[Request]) -> SimResult:
        """Process all requests in timestamp order, return aggregate result."""
```

Key decisions:
- Every request's full `hash_ids` are inserted into the selected node's cache (not just the missed blocks), because freshly prefilled blocks are likely to be reused.
- Eviction is per-block (evict 1 at a time as needed), not batch. This is because some hash_ids may already be in cache, so the actual space needed is less than `len(hash_ids) - hit_len`.
- Requests are processed in `timestamp` order (sorted ascending).

## CLI (`__main__.py`)

```
python -m flux_router --data <path> [--nodes 8] [--capacity 10000] [--selector random|cache_aware] [--evictor fifo|lru]
```

| Flag | Default | Description |
|---|---|---|
| `--data` | required | Path to JSONL trace file |
| `--nodes` | 8 | Number of prefill nodes |
| `--capacity` | 10000 | Max cached blocks per node |
| `--selector` | `random` | Node selection strategy |
| `--evictor` | `fifo` | Block eviction strategy |

Output: cache hit rate as percentage, plus per-request hit distribution in text table.

## Cache Hit Rate Definition

```
cache_hit_rate = total_blocks_hit / total_blocks_needed

where:
  total_blocks_needed = sum(len(r.hash_ids) for r in requests)
  total_blocks_hit    = sum(prefix_match_len for each request)
```

This matches the paper's block-level hit rate: what fraction of input blocks were already cached and could skip prefill computation.
