from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Request:
    timestamp: int
    input_length: int
    output_length: int
    hash_ids: list[int]


@dataclass
class BlockEntry:
    insert_time: int
    last_access_time: int


@dataclass
class PrefillNode:
    node_id: int
    capacity: int
    cache: dict[int, BlockEntry] = field(default_factory=dict)

    @property
    def used(self) -> int:
        return len(self.cache)

    @property
    def available(self) -> int:
        return self.capacity - self.used
