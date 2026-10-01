"""Run a labeler over a directory of document shards, writing one sidecar shard per input shard.

A sidecar has the same file name and row order as its input shard and holds ``id`` plus the labeler's columns, so labels
join to documents by position (and are checked by id). Finished sidecars are skipped on rerun, which makes every labeler
resumable at shard granularity; ``--shard-index/--num-shards`` split the shards across processes or GPUs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from wildai.data.parquet_io import list_shards, write_table_atomic

LabelFn = Callable[[pa.Table, Path], pa.Table]
"""(input columns of one shard, the shard's path) -> sidecar table with ``id`` first, one row per input row."""


def my_shards(input_dir: Path, shard_index: int, num_shards: int) -> list[Path]:
    if not 0 <= shard_index < num_shards:
        raise ValueError("need 0 <= shard_index < num_shards")
    return list_shards(input_dir)[shard_index::num_shards]


def label_directory(input_dir: Path, output_dir: Path, columns: list[str], label: LabelFn, *,
                    shard_index: int = 0, num_shards: int = 1) -> list[Path]:
    """Label this process's share of the input shards; returns the sidecars written."""

    written = []
    for shard in my_shards(input_dir, shard_index, num_shards):
        out = output_dir / shard.name
        if out.exists():
            continue
        table = pq.read_table(shard, columns=columns)
        sidecar = label(table, shard)
        if sidecar.column_names[0] != "id" or sidecar["id"].to_pylist() != table["id"].to_pylist():
            raise RuntimeError(f"sidecar for {shard.name} is not aligned with its input")
        written.append(write_table_atomic(sidecar, out))
        print(f"labeled {shard.name}: {len(table):,} documents", flush=True)
    return written
