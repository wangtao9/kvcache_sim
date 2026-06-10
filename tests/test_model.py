from kvcache_sim.model import PrefillNode, Request


def test_request_fields():
    r = Request(timestamp=100, input_length=500, output_length=50, hash_ids=[1, 2, 3])
    assert r.timestamp == 100
    assert r.input_length == 500
    assert r.output_length == 50
    assert r.hash_ids == [1, 2, 3]


def test_prefill_node_empty():
    node = PrefillNode(node_id=0, capacity=100)
    assert node.used == 0
    assert node.available == 100


def test_prefill_node_with_cache():
    node = PrefillNode(node_id=1, capacity=100)
    node.cache[5] = 0
    node.cache[10] = 0
    assert node.used == 2
    assert node.available == 98


def test_prefill_node_available_can_go_negative():
    """available = capacity - used; if cache exceeds capacity, available is negative."""
    node = PrefillNode(node_id=0, capacity=2)
    node.cache[1] = 0
    node.cache[2] = 0
    node.cache[3] = 0
    assert node.used == 3
    assert node.available == -1
