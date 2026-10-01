"""Small synthetic pools, a byte-level tokenizer and a small recipe, so the tests need no downloaded data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken

from wildai.training.recipe import Recipe
from wildai.training.tokenizer import BOS, Tokenizer

SMALL_RECIPE = Recipe(sequence_len=64, tokens_per_step=512, device_batch_size=2, vocab_size=257, fp8=False, compile=False)


def byte_tokenizer() -> Tokenizer:
    """One token per byte plus <|bos|> (id 256)."""
    encoding = tiktoken.Encoding(name="bytes", pat_str=r".", mergeable_ranks={bytes([i]): i for i in range(256)}, special_tokens={BOS: 256})
    return Tokenizer(encoding)


def write_raw_pool(directory: Path, prefix: str, count: int, seed: int) -> None:
    """Documents of 5 to 300 letters, in two parquet files."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(5, 300, count)
    columns = {
        "id": [f"{prefix}-{i}" for i in range(count)],
        "text": ["".join(chr(97 + (i + j) % 26) for j in range(n)) for i, n in enumerate(lengths)],
    }
    directory.mkdir(parents=True, exist_ok=True)
    half = count // 2  # two files, to check that selection does not depend on sharding
    table = pa.table(columns)
    pq.write_table(table.slice(0, half), directory / "part-0.parquet", row_group_size=97)
    pq.write_table(table.slice(half), directory / "part-1.parquet", row_group_size=97)
