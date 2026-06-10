from kvcache_sim.model import PrefillNode
from kvcache_sim.selector import CacheAwareSelector, RandomSelector, prefix_match_len


class TestPrefixMatchLen:
    def test_empty_request(self):
        cache: dict[int, int] = {}
        assert prefix_match_len([], cache) == 0

    def test_empty_cache(self):
        cache: dict[int, int] = {}
        assert prefix_match_len([1, 2, 3], cache) == 0

    def test_full_match(self):
        cache = {1: 0, 2: 0, 3: 0}
        assert prefix_match_len([1, 2, 3], cache) == 3

    def test_partial_match(self):
        cache = {1: 0, 2: 0}
        assert prefix_match_len([1, 2, 3], cache) == 2

    def test_non_contiguous_match_stops_early(self):
        cache = {1: 0, 3: 0}
        assert prefix_match_len([1, 2, 3], cache) == 1

    def test_no_match(self):
        cache = {10: 0}
        assert prefix_match_len([1, 2, 3], cache) == 0


class TestRandomSelector:
    def test_returns_valid_node_id(self):
        nodes = [PrefillNode(node_id=i, capacity=100) for i in range(4)]
        selector = RandomSelector(seed=42)
        from kvcache_sim.model import Request
        req = Request(timestamp=0, input_length=1000, output_length=100, hash_ids=[1, 2, 3])
        chosen = selector.select(req, nodes)
        assert chosen in {0, 1, 2, 3}

    def test_deterministic_with_seed(self):
        nodes = [PrefillNode(node_id=i, capacity=100) for i in range(4)]
        from kvcache_sim.model import Request
        req = Request(timestamp=0, input_length=1000, output_length=100, hash_ids=[1])
        s1 = RandomSelector(seed=123)
        s2 = RandomSelector(seed=123)
        assert s1.select(req, nodes) == s2.select(req, nodes)


class TestCacheAwareSelector:
    def test_picks_node_with_longest_prefix(self):
        from kvcache_sim.model import Request
        node_a = PrefillNode(node_id=0, capacity=100)
        node_a.cache[1] = 0
        node_a.cache[2] = 0
        node_b = PrefillNode(node_id=1, capacity=100)
        node_b.cache[1] = 0
        node_b.cache[2] = 0
        node_b.cache[3] = 0
        selector = CacheAwareSelector()
        req = Request(timestamp=0, input_length=1000, output_length=100, hash_ids=[1, 2, 3, 4])
        assert selector.select(req, [node_a, node_b]) == 1

    def test_tie_breaks_by_least_loaded(self):
        from kvcache_sim.model import Request
        node_a = PrefillNode(node_id=0, capacity=100)
        node_a.cache[1] = 0
        node_a.cache[99] = 0
        node_b = PrefillNode(node_id=1, capacity=100)
        node_b.cache[1] = 0
        selector = CacheAwareSelector()
        req = Request(timestamp=0, input_length=1000, output_length=100, hash_ids=[1, 2])
        assert selector.select(req, [node_a, node_b]) == 1

    def test_no_match_picks_least_loaded(self):
        from kvcache_sim.model import Request
        node_a = PrefillNode(node_id=0, capacity=100)
        node_a.cache[10] = 0
        node_a.cache[11] = 0
        node_b = PrefillNode(node_id=1, capacity=100)
        selector = CacheAwareSelector()
        req = Request(timestamp=0, input_length=1000, output_length=100, hash_ids=[1, 2])
        assert selector.select(req, [node_a, node_b]) == 1
