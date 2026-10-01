"""Tokenized document pools: build once per pool, then every training run reads token spans from them.

A raw pool is one or more parquet files with at least an `id` (string) and a `text` (string) column. Training uses
two: the human-labelled pool and the AI-labelled pool (every run, including the filtering pairs, mixes these two).

A pool store is a directory with
- `tokens.bin`: every document as `[<|bos|>, *tokens]`, concatenated, uint16;
- `index.parquet`: one row per document in *selection order* with `id`, `key`, `offset`, `length`. `key` is the
  keyed 64-bit BLAKE2b hash of the document id that also orders the released pools (`wildai.data.hashing`, key
  `wildai-pool-v1`); selection order is ascending `key`, so any budget of a pool is a prefix of the pool's order no
  matter how the raw files are sharded;
- `store.json`: provenance (tokenizer fingerprint, counts, the kept key fraction).

`--fraction f` keeps only documents with `key < f * 2**64`, i.e. the first ~f of the selection order, which is enough
for small models and identical to the corresponding prefix of a full store.

    python -m wildai.training.pools --pool human --output data/human     # the released pools (pangram/WildAI)
    python -m wildai.training.pools --pool ai --output data/ai
    python -m wildai.training.pools --input my_pool/ --output data/mine --tokenizer tokenizer.json   # local files

The tokenizer defaults to the released one (``pangram/WildAI-models``).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from wildai.data.hashing import keyed_hashes
from wildai.training.tokenizer import Tokenizer

TOKENS_FILE = "tokens.bin"
INDEX_FILE = "index.parquet"
INFO_FILE = "store.json"
SELECTION_KEY = "wildai-pool-v1"  # the hash key that orders the released pools (configs/data/pools.yaml)
TOKEN_DTYPE = np.uint16


class StoreInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    tokenizer_sha256: str
    vocab_size: int
    documents: int
    tokens: int
    fraction: float


@dataclass(frozen=True)
class PoolStore:
    """A tokenized pool opened read-only. Index arrays are in selection order."""

    directory: Path

    @cached_property
    def info(self) -> StoreInfo:
        return StoreInfo.model_validate_json((self.directory / INFO_FILE).read_text(encoding="utf-8"))

    @cached_property
    def _index(self) -> pa.Table:
        return pq.read_table(self.directory / INDEX_FILE, columns=["key", "offset", "length"])

    @cached_property
    def keys(self) -> np.ndarray:
        return self._index["key"].to_numpy().astype(np.uint64)

    @cached_property
    def offsets(self) -> np.ndarray:
        return self._index["offset"].to_numpy().astype(np.int64)

    @cached_property
    def lengths(self) -> np.ndarray:
        return self._index["length"].to_numpy().astype(np.int64)

    @cached_property
    def tokens(self) -> np.memmap:
        return np.memmap(self.directory / TOKENS_FILE, dtype=TOKEN_DTYPE, mode="r")

    def document(self, row: int, length: int | None = None) -> np.ndarray:
        """Tokens of the document at `row` of the selection order (optionally only its first `length` tokens)."""
        start = int(self.offsets[row])
        return self.tokens[start : start + int(self.lengths[row] if length is None else length)]


def _raw_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        files.extend(sorted(path.rglob("*.parquet")) if path.is_dir() else [path])
    if not files:
        raise FileNotFoundError(f"no parquet files in {inputs}")
    return files


def _batches(files: list[Path], batch_rows: int) -> Iterator[tuple[list[str], list[str]]]:
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=["id", "text"]):
            yield batch.column("id").to_pylist(), batch.column("text").to_pylist()


def build_store(
    inputs: list[Path],
    output: Path,
    tokenizer: Tokenizer,
    fraction: float = 1.0,
    num_threads: int = os.cpu_count() or 8,
    batch_rows: int = 8192,
) -> StoreInfo:
    """Tokenize the raw pool into a store (documents stay in file order in `tokens.bin`; the index is key-sorted)."""
    assert 0 < fraction <= 1
    assert tokenizer.vocab_size <= np.iinfo(TOKEN_DTYPE).max + 1
    limit = np.uint64(min(int(fraction * 2**64), 2**64 - 1))
    output.mkdir(parents=True, exist_ok=True)
    id_chunks: list[pa.Array] = []
    key_chunks: list[np.ndarray] = []
    length_chunks: list[np.ndarray] = []
    with (output / TOKENS_FILE).open("wb") as sink:
        for batch_ids, texts in _batches(_raw_files(inputs), batch_rows):
            batch_keys = keyed_hashes(batch_ids, SELECTION_KEY)
            keep = np.arange(len(batch_ids)) if fraction == 1 else np.flatnonzero(batch_keys < limit)
            if not len(keep):
                continue
            encoded = tokenizer.encode_documents([texts[i] for i in keep], num_threads=num_threads)
            sink.write(np.concatenate([np.asarray(t, dtype=TOKEN_DTYPE) for t in encoded]).tobytes())
            id_chunks.append(pa.array([batch_ids[i] for i in keep], type=pa.large_string()))
            key_chunks.append(batch_keys[keep])
            length_chunks.append(np.fromiter((len(t) for t in encoded), dtype=np.int64, count=len(encoded)))
    if not key_chunks:
        raise ValueError("no documents kept")
    keys = np.concatenate(key_chunks)
    lengths = np.concatenate(length_chunks)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    if len(np.unique(keys)) != len(keys):
        raise ValueError("duplicate document ids (or a 64-bit key collision) in the pool")
    order = np.argsort(keys, kind="stable")
    table = {
        "id": pa.concat_arrays(id_chunks).take(pa.array(order)),
        "key": keys[order],
        "offset": offsets[order],
        "length": lengths[order].astype(np.int32),
    }
    pq.write_table(pa.table(table), output / INDEX_FILE)
    info = StoreInfo(
        tokenizer_sha256=tokenizer.fingerprint,
        vocab_size=tokenizer.vocab_size,
        documents=len(lengths),
        tokens=int(lengths.sum()),
        fraction=fraction,
    )
    (output / INFO_FILE).write_text(info.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize a raw document pool into a WildAI pool store.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pool", choices=("human", "ai", "mixed"), help="a pool of the released dataset, downloaded from the Hub")
    source.add_argument("--input", type=Path, nargs="+", help="local parquet files or directories (columns id, text)")
    parser.add_argument("--output", type=Path, required=True, help="store directory to create")
    parser.add_argument("--tokenizer", type=Path, help="tokenizer.json or tokenizer.pkl (default: the released tokenizer)")
    parser.add_argument("--fraction", type=float, default=1.0, help="keep documents whose key is in the first fraction of the key space")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    args = parser.parse_args()
    if args.pool:
        from wildai.hub.download import dataset_dir

        inputs = [dataset_dir(args.pool)]
    else:
        inputs = args.input
    info = build_store(inputs, args.output, Tokenizer.load(args.tokenizer), args.fraction, args.threads)
    print(info.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
