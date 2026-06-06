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
