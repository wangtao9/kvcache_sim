from flux_router.model import BlockEntry, PrefillNode, Request


def test_request_fields():
    r = Request(timestamp=100, input_length=500, output_length=50, hash_ids=[1, 2, 3])
    assert r.timestamp == 100
    assert r.input_length == 500
    assert r.output_length == 50
    assert r.hash_ids == [1, 2, 3]


def test_block_entry_fields():
    e = BlockEntry(insert_time=10, last_access_time=20)
    assert e.insert_time == 10
    assert e.last_access_time == 20


def test_prefill_node_empty():
    node = PrefillNode(node_id=0, capacity=100)
    assert node.used == 0
    assert node.available == 100


def test_prefill_node_with_cache():
    node = PrefillNode(node_id=1, capacity=100)
    node.cache[5] = BlockEntry(insert_time=0, last_access_time=0)
    node.cache[10] = BlockEntry(insert_time=0, last_access_time=0)
    assert node.used == 2
    assert node.available == 98


def test_prefill_node_available_can_go_negative():
    """available = capacity - used; if cache exceeds capacity, available is negative."""
    node = PrefillNode(node_id=0, capacity=2)
    node.cache[1] = BlockEntry(insert_time=0, last_access_time=0)
    node.cache[2] = BlockEntry(insert_time=0, last_access_time=0)
    node.cache[3] = BlockEntry(insert_time=0, last_access_time=0)
    assert node.used == 3
    assert node.available == -1
