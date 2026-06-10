#!/usr/bin/env python3
# capacity sweep: selector × evictor × capacity
import subprocess, sys, os

WORK_PATH = os.path.dirname(os.getcwd())
TRACE = os.path.join(WORK_PATH, "data", "trace.jsonl")
os.environ["PYTHONPATH"] = WORK_PATH

NODES = 8
CAPS = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 250000]
COMBOS = [("random", "fifo"), ("random", "lru"), ("cache_aware", "fifo"), ("cache_aware", "lru")]
LABELS = ["ra+fifo", "ra+lru", "ca+fifo", "ca+lru"]

def run(cap, sel, ev, log):
    cmd = [sys.executable, "-m", "kvcache_sim", "--data", TRACE,
           "--nodes", str(NODES), "--capacity", str(cap), "--selector", sel, "--evictor", ev]
    r = subprocess.run(cmd, capture_output=True, text=True)
    log.write(f"\n{'='*60}\n$ {' '.join(cmd)}\n{r.stdout}\n{r.stderr}\n")
    hr = ev = None
    for line in r.stdout.splitlines():
        line = line.strip()
        if "cache hit rate" in line.lower():
            hr = float(line.split(":")[-1].strip().rstrip("%")) / 100
        elif "total evictions" in line.lower():
            ev = int(line.split(":")[-1].strip())
    if hr is None or ev is None:
        print(f"\nPARSE ERROR for {' '.join(cmd)}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}", file=sys.stderr)
        sys.exit(1)
    return hr, ev

with open("capacity_sweep.log", "w") as log:
    print(f"{'capacity':>10s}", end="")
    for lb in LABELS:
        print(f"  {'hr_'+lb:>10s}  {'ev_'+lb:>10s}", end="")
    print()
    print("-" * (10 + 22 * len(LABELS)))
    for cap in CAPS:
        tag = "inf" if cap == 250000 else f"{cap:,}"
        print(f"{tag:>10s}", end="", flush=True)
        for sel, ev in COMBOS:
            hr, ev = run(cap, sel, ev, log)
            print(f"  {hr:>9.2%}  {ev:>10,}", end="", flush=True)
        print()

