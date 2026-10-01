"""Sharded Parquet reading and writing shared by every stage.

A dataset is a directory of ``part-00000.parquet``, ``part-00001.parquet``, ... files with one schema. Shards are written to
a temporary name and renamed when complete, so a crashed run never leaves a truncated shard that looks finished.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import TracebackType

import pyarrow as pa
import pyarrow.parquet as pq
from typing_extensions import Self

SHARD_GLOB = "*.parquet"


def list_shards(directory: Path) -> list[Path]:
    """The Parquet shards of a dataset directory, in name order (the dataset's row order)."""

    shards = sorted(p for p in directory.glob(SHARD_GLOB) if p.is_file())
    if not shards:
        raise FileNotFoundError(f"no Parquet shards in {directory}")
    return shards


def write_table_atomic(table: pa.Table, path: Path, row_group_size: int = 1024) -> Path:
    """Write one Parquet file (zstd) via a temporary name, so readers never see a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp, compression="zstd", row_group_size=row_group_size)
    os.replace(tmp, path)
    return path


def read_dataset(directory: Path, columns: Sequence[str] | None = None) -> pa.Table:
    """All shards of a dataset directory as one table, in row order."""

    tables = [pq.read_table(p, columns=list(columns) if columns else None) for p in list_shards(directory)]
    return pa.concat_tables(tables) if len(tables) > 1 else tables[0]


def iter_batches(directory: Path, columns: Sequence[str] | None = None, batch_size: int = 8192) -> Iterator[pa.RecordBatch]:
    """Stream a dataset directory in row order without loading it whole."""

    for path in list_shards(directory):
        yield from pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=list(columns) if columns else None)


class ShardedWriter:
    """Append tables to a dataset directory, cutting a new shard every ``rows_per_shard`` rows.

    >>> with ShardedWriter(out, schema, rows_per_shard=100_000) as writer:
    ...     writer.write(table)
    """

    def __init__(self, directory: Path, schema: pa.Schema, rows_per_shard: int = 100_000, prefix: str = "part",
                 row_group_size: int = 1024) -> None:
        if rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        self.directory = directory
        self.schema = schema
        self.rows_per_shard = rows_per_shard
        self.prefix = prefix
        self.row_group_size = row_group_size
        self.paths: list[Path] = []
        self._pending: list[pa.Table] = []
        self._pending_rows = 0

    def __enter__(self) -> Self:
        self.directory.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        if exc_type is None:
            self.close()

    def write(self, table: pa.Table) -> None:
        table = table.select(self.schema.names).cast(self.schema)
        while len(table):
            take = min(len(table), self.rows_per_shard - self._pending_rows)
            self._pending.append(table.slice(0, take))
            self._pending_rows += take
            table = table.slice(take)
            if self._pending_rows == self.rows_per_shard:
                self._flush()

    def write_rows(self, rows: Sequence[dict[str, object]]) -> None:
        if rows:
            self.write(pa.Table.from_pylist(list(rows), schema=self.schema))

    def close(self) -> list[Path]:
        self._flush()
        return self.paths

    def _flush(self) -> None:
        if not self._pending_rows:
            return
        table = pa.concat_tables(self._pending)
        path = self.directory / f"{self.prefix}-{len(self.paths):05d}.parquet"
        self.paths.append(write_table_atomic(table, path, self.row_group_size))
        self._pending, self._pending_rows = [], 0


def dataset_dirs(root: Path) -> list[Path]:
    """Every directory under ``root`` (``root`` included) that holds Parquet shards, in path order; hidden paths such as
    labeler caches are skipped."""

    found = sorted({p.parent for p in root.rglob(SHARD_GLOB)
                    if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)})
    if not found:
        raise FileNotFoundError(f"no Parquet shards under {root}")
    return found


def mirror(directory: Path, root: Path, other_root: Path) -> Path:
    """The directory at the same relative position under ``other_root`` (where a labeler wrote its sidecars)."""

    return other_root / directory.relative_to(root)


def read_joined(documents: Path, sidecars: dict[str, Path], columns: list[str] | None = None) -> pa.Table:
    """One document shard with the columns of its sidecars (name -> sidecar directory), checked row by row.

    A sidecar has the same file name and row order as the shard it labels and starts with an ``id`` column.
    """

    table = pq.read_table(documents, columns=columns)
    for directory in sidecars.values():
        side = pq.read_table(directory / documents.name)
        if side["id"].to_pylist() != table["id"].to_pylist():
            raise RuntimeError(f"{directory / documents.name} is not aligned with {documents}")
        for name in side.column_names[1:]:
            table = table.append_column(name, side[name])
    return table
