# Flux Router v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline simulation that routes LLM prefill requests to N GPU nodes based on KVCache block hits, measures overall cache hit rate, with pluggable selector/evictor abstractions.

**Architecture:** Protocol-based abstractions for node selection (`PrefillSelector`) and block eviction (`BlockEvictor`), composed by `PrefillSimulator` which processes requests sequentially. Data model uses dataclasses. CLI via `__main__.py` with argparse.

**Tech Stack:** Python 3.10+, pytest, argparse (stdlib), dataclasses (stdlib), json (stdlib), random (stdlib). No external dependencies.

---

## File Structure

| File | Responsibility |
|---|---|
| `flux_router/__init__.py` | Package init, re-exports key types |
| `flux_router/model.py` | `Request`, `BlockEntry`, `PrefillNode` dataclasses |
| `flux_router/selector.py` | `PrefillSelector` protocol, `RandomSelector`, `CacheAwareSelector`, `prefix_match_len` |
| `flux_router/evictor.py` | `BlockEvictor` protocol, `FIFOEvictor`, `LRUEvictor` |
| `flux_router/simulator.py` | `PrefillSimulator`, `SimResult` |
| `flux_router/__main__.py` | CLI entry point: argparse, load JSONL, run sim, print results |
| `tests/test_model.py` | Tests for data model |
| `tests/test_selector.py` | Tests for selectors and `prefix_match_len` |
| `tests/test_evictor.py` | Tests for evictors |
| `tests/test_simulator.py` | Integration tests for simulator |
| `tests/test_main.py` | CLI smoke test |
| `tests/conftest.py` | Shared fixtures |

---

### Task 1: Project scaffold and data model

**Files:**
- Create: `flux_router/__init__.py`
- Create: `flux_router/model.py`
- Create: `tests/conftest.py`
- Create: `tests/test_model.py`
- Create: `pyproject.toml`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "flux-router"
version = "0.1.0"
requires-python = ">=3.10"

[project.optional-dependencies]
dev = ["pytest>=8.0"]
```

- [ ] **Step 2: Create flux_router/__init__.py**

```python
from flux_router.model import BlockEntry, PrefillNode, Request
from flux_router.selector import CacheAwareSelector, PrefillSelector, RandomSelector
from flux_router.evictor import BlockEvictor, FIFOEvictor, LRUEvictor
from flux_router.simulator import PrefillSimulator, SimResult
```

- [ ] **Step 3: Create flux_router/model.py**

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Request:
    timestamp: int
    input_length: int
    output_length: int
    hash_ids: list[int]


@dataclass
class BlockEntry:
    insert_time: int
    last_access_time: int


@dataclass
class PrefillNode:
    node_id: int
    capacity: int
    cache: dict[int, BlockEntry] = field(default_factory=dict)

    @property
    def used(self) -> int:
        return len(self.cache)

    @property
    def available(self) -> int:
        return self.capacity - self.used
```

- [ ] **Step 4: Create tests/conftest.py**

```python
from flux_router.model import BlockEntry, PrefillNode, Request


def make_request(timestamp: int = 0, input_length: int = 1000,
                 output_length: int = 100, hash_ids: list[int] | None = None) -> Request:
    if hash_ids is None:
        hash_ids = []
    return Request(timestamp=timestamp, input_length=input_length,
                   output_length=output_length, hash_ids=hash_ids)


def make_node(node_id: int = 0, capacity: int = 100,
              cached_ids: list[int] | None = None, now: int = 0) -> PrefillNode:
    node = PrefillNode(node_id=node_id, capacity=capacity)
    if cached_ids:
        for hid in cached_ids:
            node.cache[hid] = BlockEntry(insert_time=now, last_access_time=now)
    return node
```

- [ ] **Step 5: Write tests for model**

Create `tests/test_model.py`:

```python
from flux_router.model import BlockEntry, PrefillNode, Request


def test_request_fields():
    r = Request(timestamp=100, input_length=500, output_length=50, hash_ids=[1, 2, 3])
    assert r.timestamp == 100
    assert r.input_length == 500
    assert r.output_length == 50
    assert r.hash_ids == [1, 2, 3]


def test_block_entry_fields():
    e = BlockEntry(insert_time=10, last_access_time=20)
    assert e.insert_time == 10
    assert e.last_access_time == 20


def test_prefill_node_empty():
    node = PrefillNode(node_id=0, capacity=100)
    assert node.used == 0
    assert node.available == 100


def test_prefill_node_with_cache():
    node = PrefillNode(node_id=1, capacity=100)
    node.cache[5] = BlockEntry(insert_time=0, last_access_time=0)
    node.cache[10] = BlockEntry(insert_time=0, last_access_time=0)
    assert node.used == 2
    assert node.available == 98
```

- [ ] **Step 6: Run tests and verify they pass**

Run: `cd /Users/wt/share/python/flux_router && python -m pytest tests/test_model.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml flux_router/__init__.py flux_router/model.py tests/conftest.py tests/test_model.py
git commit -m "feat: add project scaffold and data model (Request, BlockEntry, PrefillNode)"
```

---

### Task 2: Node selectors

**Files:**
- Create: `flux_router/selector.py`
- Create: `tests/test_selector.py`

- [ ] **Step 1: Write failing tests for prefix_match_len and selectors**

Create `tests/test_selector.py`:

```python
import random

from flux_router.model import BlockEntry, PrefillNode
from flux_router.selector import CacheAwareSelector, RandomSelector, prefix_match_len


class TestPrefixMatchLen:
    def test_empty_request(self):
        cache: dict[int, BlockEntry] = {}
        assert prefix_match_len([], cache) == 0

    def test_empty_cache(self):
        cache: dict[int, BlockEntry] = {}
        assert prefix_match_len([1, 2, 3], cache) == 0

    def test_full_match(self):
        cache = {1: BlockEntry(0, 0), 2: BlockEntry(0, 0), 3: BlockEntry(0, 0)}
        assert prefix_match_len([1, 2, 3], cache) == 3

    def test_partial_match(self):
        cache = {1: BlockEntry(0, 0), 2: BlockEntry(0, 0)}
        assert prefix_match_len([1, 2, 3], cache) == 2

    def test_non_contiguous_match_stops_early(self):
        cache = {1: BlockEntry(0, 0), 3: BlockEntry(0, 0)}
        assert prefix_match_len([1, 2, 3], cache) == 1

    def test_no_match(self):
        cache = {10: BlockEntry(0, 0)}
        assert prefix_match_len([1, 2, 3], cache) == 0


class TestRandomSelector:
    def test_returns_valid_node_id(self):
        nodes = [PrefillNode(node_id=i, capacity=100) for i in range(4)]
        selector = RandomSelector(seed=42)
        from tests.conftest import make_request
        req = make_request(hash_ids=[1, 2, 3])
        chosen = selector.select(req, nodes)
        assert chosen in {0, 1, 2, 3}

    def test_deterministic_with_seed(self):
        nodes = [PrefillNode(node_id=i, capacity=100) for i in range(4)]
        from tests.conftest import make_request
        req = make_request(hash_ids=[1])
        s1 = RandomSelector(seed=123)
        s2 = RandomSelector(seed=123)
        assert s1.select(req, nodes) == s2.select(req, nodes)


class TestCacheAwareSelector:
    def test_picks_node_with_longest_prefix(self):
        from tests.conftest import make_request
        node_a = PrefillNode(node_id=0, capacity=100)
        node_a.cache[1] = BlockEntry(0, 0)
        node_a.cache[2] = BlockEntry(0, 0)
        node_b = PrefillNode(node_id=1, capacity=100)
        node_b.cache[1] = BlockEntry(0, 0)
        node_b.cache[2] = BlockEntry(0, 0)
        node_b.cache[3] = BlockEntry(0, 0)
        selector = CacheAwareSelector()
        req = make_request(hash_ids=[1, 2, 3, 4])
        assert selector.select(req, [node_a, node_b]) == 1

    def test_tie_breaks_by_least_loaded(self):
        from tests.conftest import make_request
        node_a = PrefillNode(node_id=0, capacity=100)
        node_a.cache[1] = BlockEntry(0, 0)
        node_a.cache[99] = BlockEntry(0, 0)
        node_b = PrefillNode(node_id=1, capacity=100)
        node_b.cache[1] = BlockEntry(0, 0)
        selector = CacheAwareSelector()
        req = make_request(hash_ids=[1, 2])
        assert selector.select(req, [node_a, node_b]) == 1

    def test_no_match_picks_least_loaded(self):
        from tests.conftest import make_request
        node_a = PrefillNode(node_id=0, capacity=100)
        node_a.cache[10] = BlockEntry(0, 0)
        node_a.cache[11] = BlockEntry(0, 0)
        node_b = PrefillNode(node_id=1, capacity=100)
        selector = CacheAwareSelector()
        req = make_request(hash_ids=[1, 2])
        assert selector.select(req, [node_a, node_b]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wt/share/python/flux_router && python -m pytest tests/test_selector.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement selector.py**

Create `flux_router/selector.py`:

```python
from __future__ import annotations

import random
from typing import Protocol

from flux_router.model import PrefillNode, Request


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wt/share/python/flux_router && python -m pytest tests/test_selector.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add flux_router/selector.py tests/test_selector.py
git commit -m "feat: add PrefillSelector protocol, RandomSelector, CacheAwareSelector"
```

---

### Task 3: Block evictors

**Files:**
- Create: `flux_router/evictor.py`
- Create: `tests/test_evictor.py`

- [ ] **Step 1: Write failing tests for evictors**

Create `tests/test_evictor.py`:

```python
from flux_router.model import BlockEntry, PrefillNode
from flux_router.evictor import FIFOEvictor, LRUEvictor


def make_filled_node(capacity: int = 5, now: int = 0) -> PrefillNode:
    node = PrefillNode(node_id=0, capacity=capacity)
    for i in range(capacity):
        node.cache[i] = BlockEntry(insert_time=now + i, last_access_time=now + i)
    return node


class TestFIFOEvictor:
    def test_evicts_oldest_insert(self):
        node = make_filled_node(capacity=3, now=0)
        evictor = FIFOEvictor()
        evicted = evictor.evict(node, need=2, now=10)
        assert evicted == [0, 1]

    def test_evict_respects_insert_time_not_access_time(self):
        node = PrefillNode(node_id=0, capacity=3)
        node.cache[0] = BlockEntry(insert_time=0, last_access_time=100)
        node.cache[1] = BlockEntry(insert_time=1, last_access_time=1)
        node.cache[2] = BlockEntry(insert_time=2, last_access_time=2)
        evictor = FIFOEvictor()
        evicted = evictor.evict(node, need=1, now=200)
        assert evicted == [0]


class TestLRUEvictor:
    def test_evicts_oldest_access(self):
        node = PrefillNode(node_id=0, capacity=3)
        node.cache[0] = BlockEntry(insert_time=0, last_access_time=50)
        node.cache[1] = BlockEntry(insert_time=1, last_access_time=1)
        node.cache[2] = BlockEntry(insert_time=2, last_access_time=2)
        evictor = LRUEvictor()
        evicted = evictor.evict(node, need=1, now=100)
        assert evicted == [1]

    def test_evicts_multiple(self):
        node = PrefillNode(node_id=0, capacity=4)
        node.cache[0] = BlockEntry(insert_time=0, last_access_time=10)
        node.cache[1] = BlockEntry(insert_time=1, last_access_time=30)
        node.cache[2] = BlockEntry(insert_time=2, last_access_time=5)
        node.cache[3] = BlockEntry(insert_time=3, last_access_time=20)
        evictor = LRUEvictor()
        evicted = evictor.evict(node, need=3, now=100)
        assert evicted == [2, 0, 3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wt/share/python/flux_router && python -m pytest tests/test_evictor.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement evictor.py**

Create `flux_router/evictor.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wt/share/python/flux_router && python -m pytest tests/test_evictor.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add flux_router/evictor.py tests/test_evictor.py
git commit -m "feat: add BlockEvictor protocol, FIFOEvictor, LRUEvictor"
```

---

### Task 4: Simulator

**Files:**
- Create: `flux_router/simulator.py`
- Create: `tests/test_simulator.py`

- [ ] **Step 1: Write failing tests for simulator**

Create `tests/test_simulator.py`:

```python
from flux_router.model import BlockEntry, PrefillNode, Request
from flux_router.evictor import FIFOEvictor, LRUEvictor
from flux_router.selector import CacheAwareSelector, RandomSelector
from flux_router.simulator import PrefillSimulator


class TestProcessRequest:
    def test_cache_miss_on_empty_node(self):
        node = PrefillNode(node_id=0, capacity=10)
        sim = PrefillSimulator([node], RandomSelector(seed=0), FIFOEvictor())
        req = Request(timestamp=0, input_length=1000, output_length=100, hash_ids=[1, 2, 3])
        hit = sim.process_request(req, now=0)
        assert hit == 0
        assert node.used == 3

    def test_full_prefix_hit(self):
        node = PrefillNode(node_id=0, capacity=10)
        node.cache[1] = BlockEntry(0, 0)
        node.cache[2] = BlockEntry(0, 0)
        node.cache[3] = BlockEntry(0, 0)
        sim = PrefillSimulator([node], CacheAwareSelector(), FIFOEvictor())
        req = Request(timestamp=0, input_length=1000, output_length=100, hash_ids=[1, 2, 3])
        hit = sim.process_request(req, now=10)
        assert hit == 3
        assert node.used == 3
        assert node.cache[1].last_access_time == 10

    def test_partial_prefix_hit(self):
        node = PrefillNode(node_id=0, capacity=10)
        node.cache[1] = BlockEntry(0, 0)
        node.cache[2] = BlockEntry(0, 0)
        sim = PrefillSimulator([node], CacheAwareSelector(), FIFOEvictor())
        req = Request(timestamp=0, input_length=1000, output_length=100, hash_ids=[1, 2, 3, 4])
        hit = sim.process_request(req, now=10)
        assert hit == 2
        assert node.used == 4
        assert 3 in node.cache
        assert 4 in node.cache

    def test_eviction_when_full(self):
        node = PrefillNode(node_id=0, capacity=3)
        node.cache[10] = BlockEntry(0, 0)
        node.cache[11] = BlockEntry(0, 0)
        node.cache[12] = BlockEntry(0, 0)
        sim = PrefillSimulator([node], CacheAwareSelector(), FIFOEvictor())
        req = Request(timestamp=0, input_length=1000, output_length=100, hash_ids=[1, 2])
        hit = sim.process_request(req, now=10)
        assert hit == 0
        assert node.used == 3
        assert 10 not in node.cache
        assert 1 in node.cache
        assert 2 in node.cache

    def test_lru_eviction_updates_access_time(self):
        node = PrefillNode(node_id=0, capacity=3)
        node.cache[1] = BlockEntry(insert_time=0, last_access_time=0)
        node.cache[2] = BlockEntry(insert_time=1, last_access_time=1)
        node.cache[3] = BlockEntry(insert_time=2, last_access_time=2)
        sim = PrefillSimulator([node], CacheAwareSelector(), LRUEvictor())
        req = Request(timestamp=0, input_length=1000, output_length=100, hash_ids=[1, 2, 3, 4])
        hit = sim.process_request(req, now=10)
        assert hit == 3
        assert node.cache[1].last_access_time == 10
        assert node.cache[2].last_access_time == 10
        assert node.cache[3].last_access_time == 10
        assert 4 in node.cache
        assert node.used == 3

    def test_cache_aware_selects_best_node(self):
        node_a = PrefillNode(node_id=0, capacity=10)
        node_a.cache[1] = BlockEntry(0, 0)
        node_a.cache[2] = BlockEntry(0, 0)
        node_b = PrefillNode(node_id=1, capacity=10)
        node_b.cache[1] = BlockEntry(0, 0)
        node_b.cache[2] = BlockEntry(0, 0)
        node_b.cache[3] = BlockEntry(0, 0)
        sim = PrefillSimulator([node_a, node_b], CacheAwareSelector(), FIFOEvictor())
        req = Request(timestamp=0, input_length=1000, output_length=100, hash_ids=[1, 2, 3, 4])
        hit = sim.process_request(req, now=10)
        assert hit == 3
        assert 4 in node_b.cache
        assert 4 not in node_a.cache


class TestRun:
    def test_run_two_requests(self):
        node = PrefillNode(node_id=0, capacity=10)
        sim = PrefillSimulator([node], CacheAwareSelector(), FIFOEvictor())
        requests = [
            Request(timestamp=0, input_length=1000, output_length=100, hash_ids=[1, 2, 3]),
            Request(timestamp=100, input_length=1000, output_length=100, hash_ids=[1, 2, 3, 4, 5]),
        ]
        result = sim.run(requests)
        assert result.total_requests == 2
        assert result.total_blocks_needed == 8
        assert result.total_blocks_hit == 3
        assert result.cache_hit_rate == 3 / 8
        assert result.per_request_hits == [0, 3]

    def test_run_with_zero_blocks_needed(self):
        node = PrefillNode(node_id=0, capacity=10)
        sim = PrefillSimulator([node], CacheAwareSelector(), FIFOEvictor())
        requests = [
            Request(timestamp=0, input_length=1000, output_length=100, hash_ids=[]),
        ]
        result = sim.run(requests)
        assert result.total_requests == 1
        assert result.total_blocks_needed == 0
        assert result.total_blocks_hit == 0
        assert result.cache_hit_rate == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wt/share/python/flux_router && python -m pytest tests/test_simulator.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement simulator.py**

Create `flux_router/simulator.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

from flux_router.evictor import BlockEvictor
from flux_router.model import BlockEntry, PrefillNode, Request
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
                node.cache[hid].last_access_time = now
            else:
                if node.used >= node.capacity:
                    evict_ids = self._evictor.evict(node, need=1, now=now)
                    for eid in evict_ids:
                        del node.cache[eid]
                node.cache[hid] = BlockEntry(insert_time=now, last_access_time=now)

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wt/share/python/flux_router && python -m pytest tests/test_simulator.py -v`
Expected: all passed

- [ ] **Step 5: Run all tests together**

Run: `cd /Users/wt/share/python/flux_router && python -m pytest tests/ -v`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add flux_router/simulator.py tests/test_simulator.py
git commit -m "feat: add PrefillSimulator with process_request and run"
```

---

### Task 5: CLI entry point

**Files:**
- Create: `flux_router/__main__.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write failing test for CLI**

Create `tests/test_main.py`:

```python
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _make_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "flux_router"] + args,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )


def test_cli_basic_run():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        records = [
            {"timestamp": 0, "input_length": 100, "output_length": 10, "hash_ids": [1, 2, 3]},
            {"timestamp": 100, "input_length": 200, "output_length": 20, "hash_ids": [1, 2, 3, 4, 5]},
        ]
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.flush()
        path = f.name

    try:
        result = _run_cli(["--data", path, "--nodes", "2", "--capacity", "10",
                           "--selector", "cache_aware", "--evictor", "fifo"])
        assert result.returncode == 0
        assert "Cache hit rate" in result.stdout
        assert "Requests:    2" in result.stdout
    finally:
        Path(path).unlink()


def test_cli_random_selector():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        records = [
            {"timestamp": 0, "input_length": 100, "output_length": 10, "hash_ids": [1]},
        ]
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.flush()
        path = f.name

    try:
        result = _run_cli(["--data", path, "--nodes", "1", "--capacity", "10",
                           "--selector", "random", "--evictor", "lru"])
        assert result.returncode == 0
        assert "random" in result.stdout
    finally:
        Path(path).unlink()


def test_cli_missing_data_file():
    result = _run_cli(["--data", "/nonexistent/file.jsonl"])
    assert result.returncode != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/wt/share/python/flux_router && python -m pytest tests/test_main.py -v`
Expected: FAIL (module not found or error)

- [ ] **Step 3: Implement __main__.py**

Create `flux_router/__main__.py`:

```python
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
    zero = sum(1 for h in hits if h == 0)
    low = sum(1 for h in hits if 1 <= h <= 5)
    mid = sum(1 for h in hits if 6 <= h <= 10)
    high = sum(1 for h in hits if h >= 11)
    total = len(hits)
    print("Per-request hit distribution:")
    print(f"  0 blocks hit:    {zero:>4} requests ({zero / total:.1%})" if total else "  (no requests)")
    print(f"  1-5 blocks hit:  {low:>4} requests ({low / total:.1%})" if total else "")
    print(f"  6-10 blocks hit: {mid:>4} requests ({mid / total:.1%})" if total else "")
    print(f"  11+ blocks hit:  {high:>4} requests ({high / total:.1%})" if total else "")


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/wt/share/python/flux_router && python -m pytest tests/test_main.py -v`
Expected: all passed

- [ ] **Step 5: Run all tests together**

Run: `cd /Users/wt/share/python/flux_router && python -m pytest tests/ -v`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add flux_router/__main__.py tests/test_main.py
git commit -m "feat: add CLI entry point with JSONL loading and result output"
```

---

### Task 6: End-to-end validation with real data

**Files:**
- No new files (validation only)

- [ ] **Step 1: Download the Mooncake trace dataset**

Run: `curl -sL -o /tmp/mooncake_trace.jsonl "https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl"`

- [ ] **Step 2: Run simulation with random selector + FIFO**

Run: `cd /Users/wt/share/python/flux_router && python -m flux_router --data /tmp/mooncake_trace.jsonl --nodes 8 --capacity 10000 --selector random --evictor fifo`

Expected: Outputs cache hit rate. With random routing and 8 nodes, hit rate should be low (blocks scattered across nodes).

- [ ] **Step 3: Run simulation with cache_aware selector + LRU**

Run: `cd /Users/wt/share/python/flux_router && python -m flux_router --data /tmp/mooncake_trace.jsonl --nodes 8 --capacity 10000 --selector cache_aware --evictor lru`

Expected: Cache hit rate should be significantly higher than random, because cache_aware routes requests to nodes with matching prefix blocks.

- [ ] **Step 4: Run simulation with cache_aware selector + LRU, larger capacity**

Run: `cd /Users/wt/share/python/flux_router && python -m flux_router --data /tmp/mooncake_trace.jsonl --nodes 8 --capacity 50000 --selector cache_aware --evictor lru`

Expected: Hit rate should increase with larger capacity (fewer evictions). The paper reports ~50% hit rate at 50K blocks for LRU.

- [ ] **Step 5: Verify results are reasonable**

Sanity checks:
- `cache_aware` > `random` hit rate (should always hold)
- `lru` >= `fifo` hit rate for this dataset (paper shows LRU slightly better)
- Higher capacity >= lower capacity hit rate
- Total blocks needed should equal sum of `len(hash_ids)` across all requests (218 records)

- [ ] **Step 6: Commit (if any fixes were needed)**

Only commit if code changes were needed. If all tests passed and results look reasonable, no commit needed for this task.

---

### Task 7: Update __init__.py and final cleanup

**Files:**
- Modify: `flux_router/__init__.py`
- Modify: `README.md`

- [ ] **Step 1: Verify __init__.py re-exports are correct**

The `__init__.py` from Task 1 should already re-export all public types. Verify by running:

Run: `cd /Users/wt/share/python/flux_router && python -c "from flux_router import Request, PrefillNode, CacheAwareSelector, RandomSelector, FIFOEvictor, LRUEvictor, PrefillSimulator, SimResult; print('OK')"`

Expected: OK

- [ ] **Step 2: Update README.md with usage instructions**

```markdown
# flux_router

Offline simulation of KVCache-centric prefill request routing for LLM serving.

## Usage

```bash
python -m flux_router --data <trace.jsonl> [--nodes 8] [--capacity 10000] [--selector random|cache_aware] [--evictor fifo|lru] [--seed 42]
```

## Data Format

Expects a JSONL file with fields: `timestamp`, `input_length`, `output_length`, `hash_ids`.

Compatible with the [Mooncake trace format](https://github.com/kvcache-ai/Mooncake).

## Architecture

- `model.py` — `Request`, `BlockEntry`, `PrefillNode` dataclasses
- `selector.py` — `PrefillSelector` protocol, `RandomSelector`, `CacheAwareSelector`
- `evictor.py` — `BlockEvictor` protocol, `FIFOEvictor`, `LRUEvictor`
- `simulator.py` — `PrefillSimulator`, `SimResult`
- `__main__.py` — CLI entry point
```

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/wt/share/python/flux_router && python -m pytest tests/ -v`
Expected: all passed

- [ ] **Step 4: Commit**

```bash
git add flux_router/__init__.py README.md
git commit -m "docs: update README with usage and architecture overview"
```
