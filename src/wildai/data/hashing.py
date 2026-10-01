"""Keyed hashing for order-independent, reproducible sampling.

Every random choice of documents in the pipeline (pool order, monthly samples, audit samples, evaluation sets) ranks
documents by a keyed BLAKE2b hash of their id. The ranking does not depend on the order documents are read in, so any
machine that sees the same ids makes the same choice, and a prefix of the ranking is a uniform random sample.
"""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Callable, Iterable
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def keyed_hash(value: str, key: str) -> int:
    """Unsigned 64-bit rank of ``value``: the first 8 bytes (big-endian) of its 128-bit BLAKE2b digest keyed by ``key``."""

    digest = hashlib.blake2b(value.encode("utf-8"), key=key.encode("utf-8")[:64], digest_size=16).digest()
    return int.from_bytes(digest[:8], "big")


def keyed_hashes(values: Iterable[str], key: str) -> np.ndarray:
    """:func:`keyed_hash` of many values, as a ``uint64`` array."""

    return np.fromiter((keyed_hash(v, key) for v in values), dtype=np.uint64)


def seeded_key(name: str, seed: int) -> str:
    """The hash key of one seeded selection, e.g. ``seeded_key("monthly/2026-06", 20260925)``."""

    return f"{name}:{seed}"


def bottom_k(items: Iterable[T], k: int, key: str, id_of: Callable[[T], str]) -> list[T]:
    """The ``k`` items with the smallest keyed hash of their id, in ascending hash order (a streaming uniform sample).

    Ties (identical ids) keep the first item seen.
    """

    if k <= 0:
        return []
    heap: list[tuple[int, int, T]] = []  # (-hash, -position, item): a max-heap on hash
    for position, item in enumerate(items):
        rank = keyed_hash(id_of(item), key)
        entry = (-rank, -position, item)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    return [item for _neg_rank, _neg_pos, item in sorted(heap, key=lambda e: (-e[0], -e[1]))]
