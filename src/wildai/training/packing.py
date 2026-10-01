# Adapted from nanochat (https://github.com/karpathy/nanochat), Copyright (c) 2025 Andrej Karpathy, MIT License; see NOTICE.
"""BOS-aligned best-fit packing of documents into fixed-length rows (nanochat's data loader algorithm).

Documents (each starting with `<|bos|>`) are drawn into a buffer of at least `buffer_size` documents, refilled a
whole source batch at a time. Each row of `row_len` tokens is filled by repeatedly taking the longest buffered
document that fits the remaining space (ties: the one buffered first). When none fits, the shortest buffered document
is cropped to fill the row exactly. nanochat discards the cropped remainder, which is how evaluation streams are
packed; training uses `keep_remainder=True`, which returns the remainder to the buffer so that every token of every
selected document is trained on exactly once.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Piece:
    """A document or part of one. `tag` is caller metadata (e.g. the source pool or the pass over an eval set)."""

    tokens: np.ndarray
    tag: int = 0

    def __len__(self) -> int:
        return int(self.tokens.shape[0])


class _Buffer:
    """Buffered pieces ordered by (length, arrival) so best-fit and shortest lookups are logarithmic."""

    def __init__(self) -> None:
        self._keys: list[tuple[int, int]] = []
        self._pieces: dict[int, Piece] = {}
        self._arrivals = 0

    def __len__(self) -> int:
        return len(self._keys)

    def push(self, piece: Piece) -> None:
        key = (len(piece), self._arrivals)
        self._arrivals += 1
        bisect.insort(self._keys, key)
        self._pieces[key[1]] = piece

    def pop_best_fit(self, space: int) -> Piece | None:
        """Longest piece of length <= space, earliest arrival among equals."""
        i = bisect.bisect_right(self._keys, (space, self._arrivals)) - 1
        if i < 0:
            return None
        i = bisect.bisect_left(self._keys, (self._keys[i][0], -1))
        return self._pieces.pop(self._keys.pop(i)[1])

    def pop_shortest(self) -> Piece:
        return self._pieces.pop(self._keys.pop(0)[1])


class BestFitPacker:
    """Turns a stream of document batches into rows; iterate to get each row as its list of pieces."""

    def __init__(self, batches: Iterable[Sequence[Piece]], row_len: int, buffer_size: int = 1000, keep_remainder: bool = False) -> None:
        self._batches = iter(batches)
        self._exhausted = False
        self.row_len = row_len
        self.buffer_size = buffer_size
        self.keep_remainder = keep_remainder
        self._buffer = _Buffer()

    def _refill(self) -> None:
        while len(self._buffer) < self.buffer_size and not self._exhausted:
            batch = next(self._batches, None)
            if batch is None:
                self._exhausted = True
                return
            for piece in batch:
                self._buffer.push(piece)

    def __iter__(self) -> Iterator[list[Piece]]:
        while (row := self._next_row()) is not None:
            yield row

    def _next_row(self) -> list[Piece] | None:
        """The next full row, or None once the source cannot fill another one."""
        row: list[Piece] = []
        pos = 0
        while pos < self.row_len:
            self._refill()
            if not len(self._buffer):
                return None
            space = self.row_len - pos
            piece = self._buffer.pop_best_fit(space)
            if piece is None:
                piece = self._buffer.pop_shortest()
                if self.keep_remainder:
                    self._buffer.push(Piece(piece.tokens[space:], piece.tag))
                piece = Piece(piece.tokens[:space], piece.tag)
            row.append(piece)
            pos += len(piece)
        return row


def row_tokens(row: Sequence[Piece]) -> np.ndarray:
    return np.concatenate([piece.tokens for piece in row])
