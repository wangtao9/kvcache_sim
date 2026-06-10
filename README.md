# KV-Cache Simulator

A simulator for Prefill node selection and KV-Cache block eviction, based on the  [Mooncake](https://arxiv.org/pdf/2407.00079) architecture.
It simulates multi-node prefill scheduling using real-world trace data to evaluate the overall cache hit rate.

## Usage

```bash
python -m kvcache_sim --data <trace.jsonl> [--nodes 8] [--capacity 10000] [--selector random|cache_aware] [--evictor fifo|lru] [--seed 42]
```

### Options

| Flag | Default            | Description |
|---|--------------------|---|
| `--data` | `data/trace.jsonl` | Path to JSONL trace file |
| `--nodes` | 8                  | Number of prefill nodes |
| `--capacity` | 10000              | Max cached blocks per node |
| `--selector` | `random`           | Node selection strategy (`random` or `cache_aware`) |
| `--evictor` | `fifo`             | Block eviction strategy (`fifo` or `lru`) |
| `--seed` | 42                 | Random seed for `random` selector |

### Example

```bash
# Download the Mooncake trace if needed
curl -sL -o data/trace.jsonl "https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl"

# Run with cache-aware routing and LRU eviction
python -m kvcache_sim --data data/trace.jsonl --nodes 8 --capacity 10000 --selector cache_aware --evictor lru
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
kvcache_sim/
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

## Experiment
下表展示了不同容量 (Capacity) 下，四种组合策略的命中率 (Hit Rate, HR) 与驱逐次数 (Evictions, EV) 对比。

| Cap | RA+FIFO<br>HR | RA+FIFO<br>EV | RA+LRU<br>HR | RA+LRU<br>EV | CA+FIFO<br>HR | CA+FIFO<br>EV | CA+LRU<br>HR | CA+LRU<br>EV |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **500** | 32.07% | 274,009 | 34.14% | 265,603 | 31.79% | 275,226 | 34.00% | 268,666 |
| **1,000** | 33.97% | 262,248 | 35.07% | 257,796 | 32.94% | 266,500 | 34.25% | 266,122 |
| **2,000** | 35.77% | 246,790 | 36.37% | 244,455 | 34.14% | 253,595 | 35.03% | 259,940 |
| **4,000** | 37.42% | 223,984 | 37.96% | 221,944 | 36.35% | 228,410 | 37.95% | 241,975 |
| **8,000** | 38.95% | 185,717 | 39.38% | 184,160 | 41.18% | 175,966 | 44.77% | 202,881 |
| **16,000**| 39.79% | 118,303 | 40.03% | 117,485 | 45.42% | 94,568 | 50.63% | 162,903 |
| **32,000**| 40.14% | 4,956 | 40.15% | 4,920 | 49.53% | 105 | 54.82% | 129,605 |
| **64,000**| 40.15% | 0 | 40.15% | 0 | 52.68% | 94 | 55.19% | 96,087 |
| **128,000**| 40.15% | 0 | 40.15% | 0 | 53.93% | 221 | 55.25% | 31,851 |
| **inf** | 40.15% | 0 | 40.15% | 0 | **55.26%** | 0 | **55.26%** | 0 |

> **注**：表格中括号内的数字代表 **驱逐次数 (Evictions)**，百分数为 **命中率 (Hit Rate)**。
> *   **RA**: Random Assignment (随机路由)
> *   **CA**: Cache-Aware (感知缓存路由)
> *   **FIFO/LRU**: 驱逐策略
