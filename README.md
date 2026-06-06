# flux_router

Offline simulation of KVCache-centric prefill request routing for LLM serving.

Based on the [Mooncake](https://arxiv.org/pdf/2407.00079) architecture — routes incoming prefill requests to GPU nodes based on KVCache block prefix matching to maximize cache hit rate.

## Usage

```bash
python -m flux_router --data <trace.jsonl> [--nodes 8] [--capacity 10000] [--selector random|cache_aware] [--evictor fifo|lru] [--seed 42]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--data` | required | Path to JSONL trace file |
| `--nodes` | 8 | Number of prefill nodes |
| `--capacity` | 10000 | Max cached blocks per node |
| `--selector` | `random` | Node selection strategy (`random` or `cache_aware`) |
| `--evictor` | `fifo` | Block eviction strategy (`fifo` or `lru`) |
| `--seed` | 42 | Random seed for `random` selector |

### Example

```bash
# Download the Mooncake trace
curl -sL -o trace.jsonl "https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl"

# Run with cache-aware routing and LRU eviction
python -m flux_router --data trace.jsonl --nodes 8 --capacity 10000 --selector cache_aware --evictor lru
```

## Data Format

Expects a JSONL file where each line has:

```json
{"timestamp": 0, "input_length": 6755, "output_length": 500, "hash_ids": [0, 1, 2, 3]}
```

- `timestamp`: Request arrival time in milliseconds
- `input_length`: Number of input tokens
- `output_length`: Number of output tokens
- `hash_ids`: Ordered list of prefix cache block hash IDs (chained hashing, 512 tokens per block)

Compatible with the [Mooncake trace format](https://github.com/kvcache-ai/Mooncake).

## Architecture

```
flux_router/
├── model.py          # Request, BlockEntry, PrefillNode dataclasses
├── selector.py       # PrefillSelector protocol, RandomSelector, CacheAwareSelector
├── evictor.py        # BlockEvictor protocol, FIFOEvictor, LRUEvictor
├── simulator.py      # PrefillSimulator, SimResult
└── __main__.py       # CLI entry point
```

### Key Concepts

- **PrefillSelector**: Chooses which node handles a request. `CacheAwareSelector` picks the node with the longest KVCache prefix match, breaking ties by least-loaded node.
- **BlockEvictor**: Decides which cached blocks to evict when a node is full. `FIFOEvictor` evicts the oldest inserted block; `LRUEvictor` evicts the least recently accessed.
- **Cache hit rate**: Fraction of input blocks that were already cached on the selected node, avoiding redundant prefill computation.

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```
