from flux_router.model import PrefillNode, Request


def make_request(timestamp: int = 0, input_length: int = 1000,
                 output_length: int = 100, hash_ids: list[int] | None = None) -> Request:
    if hash_ids is None:
        hash_ids = []
    return Request(timestamp=timestamp, input_length=input_length,
                   output_length=output_length, hash_ids=hash_ids)


def make_node(node_id: int = 0, capacity: int = 100,
              cached_ids: list[int] | None = None, now: int = 0) -> PrefillNode:
    node = PrefillNode(node_id=node_id, capacity=capacity)
    if cached_ids:
        for hid in cached_ids:
            node.cache[hid] = now	# timestamp 
    return node
