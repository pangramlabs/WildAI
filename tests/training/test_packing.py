"""Best-fit packing: nanochat's semantics when discarding, exact token conservation when keeping remainders."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from wildai.training.packing import BestFitPacker, Piece, row_tokens

ROW = 33


def documents(count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.concatenate([[0], rng.integers(1, 100, n)]) for n in rng.integers(1, 80, count)]


def batches(docs: list[np.ndarray], size: int = 7) -> Iterator[list[Piece]]:
    for start in range(0, len(docs), size):
        yield [Piece(d, i) for i, d in enumerate(docs[start : start + size], start)]


def reference_rows(docs: list[np.ndarray], row_len: int, buffer_size: int, size: int = 7) -> list[np.ndarray]:
    """nanochat's list-based loop, run until the source cannot fill another row."""
    source = iter([list(b) for b in (docs[i : i + size] for i in range(0, len(docs), size))])
    buffer: list[np.ndarray] = []
    rows = []
    while True:
        row, pos = [], 0
        while pos < row_len:
            while len(buffer) < buffer_size:
                batch = next(source, None)
                if batch is None:
                    break
                buffer.extend(batch)
            if not buffer:
                return rows
            remaining = row_len - pos
            best, best_len = -1, 0
            for i, doc in enumerate(buffer):
                if best_len < len(doc) <= remaining:
                    best, best_len = i, len(doc)
            if best >= 0:
                doc = buffer.pop(best)
            else:
                doc = buffer.pop(min(range(len(buffer)), key=lambda i: len(buffer[i])))[:remaining]
            row.append(doc)
            pos += len(doc)
        rows.append(np.concatenate(row))


def test_discarding_packer_matches_nanochat() -> None:
    docs = documents(500, seed=0)
    ours = [row_tokens(r) for r in BestFitPacker(batches(docs), ROW, buffer_size=20)]
    reference = reference_rows(docs, ROW, buffer_size=20)
    assert len(ours) == len(reference) > 100
    assert all(np.array_equal(a, b) for a, b in zip(ours, reference))


def test_keeping_remainders_trains_every_token_once() -> None:
    docs = documents(400, seed=1)
    missing = -sum(len(d) for d in docs) % ROW
    if missing:
        docs.append(np.zeros(missing, dtype=np.int64))  # make the total a whole number of rows
    rows = list(BestFitPacker(batches(docs), ROW, buffer_size=20, keep_remainder=True))
    assert all(len(row_tokens(r)) == ROW for r in rows)
    used: dict[int, int] = {}
    for row in rows:
        for piece in row:
            used[piece.tag] = used.get(piece.tag, 0) + len(piece)
    assert used == {i: len(d) for i, d in enumerate(docs)}


def test_rows_start_with_a_document_when_documents_fit_a_row() -> None:
    docs = [d[:ROW] for d in documents(400, seed=2)]
    rows = list(BestFitPacker(batches(docs), ROW, buffer_size=20, keep_remainder=True))
    assert all(r[0].tokens[0] == 0 for r in rows[:-5])
