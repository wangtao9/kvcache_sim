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
