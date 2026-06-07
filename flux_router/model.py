from __future__ import annotations

from dataclasses import dataclass, field
from collections import OrderedDict


@dataclass
class Request:
    timestamp: int
    input_length: int
    output_length: int
    hash_ids: list[int]

@dataclass
class PrefillNode:
    node_id: int
    capacity: int
    # <hash_id, timestamp>
    cache: OrderedDict[int, int] = field(default_factory=OrderedDict)

    @property
    def used(self) -> int:
        return len(self.cache)

    @property
    def available(self) -> int:
        return self.capacity - self.used
