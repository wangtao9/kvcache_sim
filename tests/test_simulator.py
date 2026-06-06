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
        node.cache[10] = BlockEntry(insert_time=0, last_access_time=0)
        node.cache[1] = BlockEntry(insert_time=1, last_access_time=1)
        node.cache[2] = BlockEntry(insert_time=2, last_access_time=2)
        sim = PrefillSimulator([node], CacheAwareSelector(), LRUEvictor())
        req = Request(timestamp=0, input_length=1000, output_length=100, hash_ids=[1, 2, 3])
        hit = sim.process_request(req, now=10)
        assert hit == 2
        assert 10 not in node.cache  # evicted: oldest access_time, not part of request
        assert node.cache[1].last_access_time == 10
        assert node.cache[2].last_access_time == 10
        assert 3 in node.cache
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
