from kvcache_sim.model import PrefillNode
from kvcache_sim.evictor import FIFOEvictor, LRUEvictor


def make_filled_node(capacity: int = 5, now: int = 0) -> PrefillNode:
    node = PrefillNode(node_id=0, capacity=capacity)
    for i in range(capacity):
        node.cache[i] = now + i
    return node


class TestFIFOEvictor:
    def test_evicts_oldest_insert(self):
        node = make_filled_node(capacity=3, now=0)
        evictor = FIFOEvictor()
        evicted = evictor.evict(node, need=2)
        assert evicted == [0, 1]

    def test_evict_respects_insert_time_not_access_time(self):
        node = PrefillNode(node_id=0, capacity=3)
        node.cache[0] = 0   # timestamp
        node.cache[1] = 1
        node.cache[2] = 2
        evictor = FIFOEvictor()
        evicted = evictor.evict(node, need=1)
        assert evicted == [0]


class TestLRUEvictor:
    def test_evicts_oldest_access(self):
        node = PrefillNode(node_id=0, capacity=3)
        node.cache[0] = 0   # timestamp
        node.cache[1] = 1
        node.cache[2] = 2
        evictor = LRUEvictor()
        evictor.on_access(node, 0)  # 模拟命中block 0: [1, 2, 0]
        evicted = evictor.evict(node, need=1)
        assert evicted == [1]

    def test_evicts_multiple(self):
        node = PrefillNode(node_id=0, capacity=4)
        node.cache[0] = 0
        node.cache[1] = 1
        node.cache[2] = 2
        node.cache[3] = 3
        evictor = LRUEvictor()
        evictor.on_access(node, 0) 
        evictor.on_access(node, 3) 
        evictor.on_access(node, 1) 
        evicted = evictor.evict(node, need=3)
        assert evicted == [2, 0, 3]
