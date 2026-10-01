"""Split Pangram-labeled candidates into training pools, sorted by a keyed hash so any prefix is a uniform sample.

Each candidate goes to the pool of its Pangram label (``human``, ``ai``, ``mixed``; the mixed pool is kept for analysis
and not used for training); documents Pangram could not label go to ``_unlabeled``. Duplicate ids keep their first
occurrence in input order. With ``--natural`` the candidates are a label-blind draw and form one ``natural`` pool of its
human and AI documents (Mixed documents are left out, as in the paper's FW26), used for the FW26 evaluation sets.

Sorting works one hash range at a time: rows are spread over ``bins`` range files, then each range is sorted and
deduplicated (equal ids have equal hashes, so duplicates share a range) and appended to its pool.

    python -m wildai.data.pools --candidates-root data/candidates/editlens --pangram-root data/labels/pangram/editlens \
        --output-dir data/pools
    python -m wildai.data.pools --natural --candidates-root data/candidates/natural \
        --pangram-root data/labels/pangram/natural --output-dir data/pools

Output: ``<output-dir>/<pool>/part-*.parquet`` with the :class:`wildai.data.schema.PoolDocument` schema and
``<output-dir>/<pool>/_pool.json`` (documents and GPT-2 tokens).
"""

from __future__ import annotations

import argparse
import json
import shutil
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from wildai.data.arrow_schema import arrow_schema
from wildai.data.config import default_config, load_config
from wildai.data.hashing import keyed_hashes
from wildai.data.parquet_io import ShardedWriter, dataset_dirs, list_shards, mirror, read_joined, write_table_atomic
from wildai.data.pools_config import PoolsConfig, PoolSettings
from wildai.data.schema import PoolDocument

POOL_SCHEMA = arrow_schema(PoolDocument)
PANGRAM_RENAMES = {"pangram_label": "label"}


def pool_names(natural: bool) -> tuple[str, ...]:
    """Every pool a build writes; the last one holds documents Pangram could not label."""

    return ("natural", "_unlabeled_natural") if natural else ("human", "ai", "mixed", "_unlabeled")


def pool_name(label: str | None, natural: bool) -> str | None:
    """The pool a Pangram label goes to (``None``: left out)."""

    if label is None:
        return pool_names(natural)[-1]
    if natural:
        return "natural" if label in ("Human", "AI") else None
    return label.lower()


def stage(candidates_root: Path, pangram_root: Path, staging: Path, settings: PoolSettings) -> None:
    """Spread every labeled candidate over hash-range files, with its input position for first-occurrence dedup."""

    shift = np.uint64(64 - int(np.log2(settings.bins)))
    position = chunks = 0
    for directory in dataset_dirs(candidates_root):
        sidecars = {"pangram": mirror(directory, candidates_root, pangram_root)}
        for shard in list_shards(directory):
            table = read_joined(shard, sidecars)
            hashes = keyed_hashes(table["id"].to_pylist(), settings.key)
            table = (table.rename_columns([PANGRAM_RENAMES.get(c, c) for c in table.column_names])
                     .append_column("sampling_hash", pa.array(hashes, pa.uint64()))
                     .append_column("order", pa.array(np.arange(position, position + len(table)), pa.int64())))
            position += len(table)
            ranges = hashes >> shift
            for bin_index in np.unique(ranges):
                part = table.filter(pa.array(ranges == bin_index))
                write_table_atomic(part, staging / f"{int(bin_index):05d}" / f"{chunks:08d}.parquet")
                chunks += 1


def build_pools(candidates_root: Path, pangram_root: Path, output: Path, settings: PoolSettings, natural: bool) -> dict:
    if settings.bins & (settings.bins - 1):
        raise ValueError("bins must be a power of two")
    staging = output / ".staging"
    for name in (".staging", *pool_names(natural)):  # a rebuild replaces the pools it writes
        shutil.rmtree(output / name, ignore_errors=True)
    stage(candidates_root, pangram_root, staging, settings)
    writers: dict[str, ShardedWriter] = {}
    stats: dict[str, dict[str, int]] = {}
    duplicates = left_out = 0
    with ExitStack() as stack:
        for bin_dir in sorted(staging.iterdir()):
            table = pa.concat_tables([pq.read_table(p) for p in sorted(bin_dir.glob("*.parquet"))])
            table = table.sort_by([("sampling_hash", "ascending"), ("id", "ascending"), ("order", "ascending")])
            ids = table["id"].to_pylist()
            first = np.array([i == 0 or ids[i] != ids[i - 1] for i in range(len(ids))], dtype=bool)
            duplicates += int((~first).sum())
            table = table.filter(pa.array(first)).drop_columns(["order"])
            names = np.array([pool_name(label, natural) for label in table["label"].to_pylist()], dtype=object)
            left_out += int(sum(n is None for n in names))
            for name in sorted({n for n in names if n is not None}):
                part = table.filter(pa.array(names == name))
                unlabeled = name == pool_names(natural)[-1]
                if not unlabeled:
                    part = part.set_column(part.schema.get_field_index("label"), "label", pc.utf8_lower(part["label"]))
                if name not in writers:
                    schema = part.schema if unlabeled else POOL_SCHEMA  # unlabeled rows keep every column, all nullable
                    writers[name] = stack.enter_context(ShardedWriter(output / name, schema, settings.rows_per_shard))
                    stats[name] = {"documents": 0, "tokens": 0}
                writers[name].write(part)
                stats[name]["documents"] += len(part)
                stats[name]["tokens"] += int(pc.sum(part["token_count"]).as_py() or 0)
    for name in writers:
        (output / name / "_pool.json").write_text(json.dumps({"pool": name, **stats[name], "key": settings.key}, indent=1))
    shutil.rmtree(staging)
    return {"pools": stats, "duplicate_ids_dropped": duplicates, "left_out": left_out}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=default_config("pools.yaml"))
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--pangram-root", type=Path, required=True, help="Pangram sidecars, mirroring --candidates-root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--natural", action="store_true", help="build the natural pool from a label-blind draw")
    args = parser.parse_args(argv)
    settings = load_config(args.config, PoolsConfig).pools
    summary = build_pools(args.candidates_root, args.pangram_root, args.output_dir, settings, args.natural)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
