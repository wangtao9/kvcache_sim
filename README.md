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
# Download the Mooncake trace if needed
curl -sL -o data/trace.jsonl "https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl"

# Run with cache-aware routing and LRU eviction
python -m flux_router --data data/trace.jsonl --nodes 8 --capacity 10000 --selector cache_aware --evictor lru
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

## Experiment
下表展示了不同容量 (Capacity) 下，四种组合策略的命中率 (Hit Rate, hr) 与驱逐次数 (Evictions, ev) 对比。

| Capacity | **RA + FIFO**<br>(hr / ev) | **RA + LRU**<br>(hr / ev) | **CA + FIFO**<br>(hr / ev) | **CA + LRU**<br>(hr / ev) |
| :---: | :---: | :---: | :---: | :---: |
| **500** | 32.07% <br> *(274,009)* | 34.14% <br> *(265,603)* | 31.79% <br> *(275,226)* | 34.00% <br> *(268,666)* |
| **1,000** | 33.97% <br> *(262,248)* | 35.07% <br> *(257,796)* | 32.94% <br> *(266,500)* | 34.25% <br> *(266,122)* |
| **2,000** | 35.77% <br> *(246,790)* | 36.37% <br> *(244,455)* | 34.14% <br> *(253,595)* | 35.03% <br> *(259,940)* |
| **4,000** | 37.42% <br> *(223,984)* | 37.96% <br> *(221,944)* | 36.35% <br> *(228,410)* | 37.95% <br> *(241,975)* |
| **8,000** | 38.95% <br> *(185,717)* | 39.38% <br> *(184,160)* | 41.18% <br> *(175,966)* | 44.77% <br> *(202,881)* |
| **16,000** | 39.79% <br> *(118,303)* | 40.03% <br> *(117,485)* | 45.42% <br> *(94,568)* | 50.63% <br> *(162,903)* |
| **32,000** | 40.14% <br> *(4,956)* | 40.15% <br> *(4,920)* | 49.53% <br> *(105)* | 54.82% <br> *(129,605)* |
| **64,000** | 40.15% <br> *(0)* | 40.15% <br> *(0)* | 52.68% <br> *(94)* | 55.19% <br> *(96,087)* |
| **128,000**| 40.15% <br> *(0)* | 40.15% <br> *(0)* | 53.93% <br> *(221)* | 55.25% <br> *(31,851)* |
| **inf** | 40.15% <br> *(0)* | 40.15% <br> *(0)* | **55.26%** <br> *(0)* | **55.26%** <br> *(0)* |

> **注**：表格中括号内的数字代表 **驱逐次数 (Evictions)**，百分数为 **命中率 (Hit Rate)**。
> *   **RA**: Random Assignment (随机路由)
> *   **CA**: Cache-Aware (感知缓存路由)
> *   **FIFO/LRU**: 驱逐策略
