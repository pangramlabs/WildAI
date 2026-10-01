"""Export release tables: pools, the monthly sample and the filter audit, joined with their labels.

    python -m wildai.data.release.export pools --pools-dir data/pools \
        --weborganizer-dir data/labels/weborganizer/pools --release-dir release
    python -m wildai.data.release.export monthly_sample --sample-dir data/monthly_sample \
        --pangram-dir data/labels/pangram/monthly_sample --weborganizer-dir data/labels/weborganizer/monthly_sample \
        --release-dir release
    python -m wildai.data.release.export filter_audit --audit-dir data/filter_audit \
        --pangram-dir data/labels/pangram/filter_audit --weborganizer-dir data/labels/weborganizer/filter_audit \
        --release-dir release

Writes ``<release-dir>/<config>/part-*.parquet`` (``filter_audit/<split>/``); see :mod:`wildai.data.release`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from multiprocessing import get_context
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from wildai.data.config import default_config, load_config
from wildai.data.parquet_io import ShardedWriter, list_shards, read_joined
from wildai.data.pii import anonymize
from wildai.data.release.config import ReleaseConfig
from wildai.data.release.schema import FilterAuditRecord, MonthlyRecord, PoolRecord, release_schema
from wildai.data.tokens import Gpt2TokenCounter, TokenCounter

POOL_LABELS = {"human": "Human", "ai": "AI", "mixed": "Mixed"}


def anonymize_rows(table: pa.Table, needs: list[bool], counter: TokenCounter) -> pa.Table:
    """Anonymize the rows flagged in ``needs``; changed rows get a new GPT-2 count and the after-labeling flag."""

    texts, counts = table["text"].to_pylist(), table["token_count"].to_pylist()
    changed = [False] * len(texts)
    for i, need in enumerate(needs):
        if need:
            result = anonymize(texts[i])
            if result.changed:
                texts[i], changed[i] = result.text, True
    recount = [i for i, c in enumerate(changed) if c]
    for i, count in zip(recount, counter.count_batch([texts[i] for i in recount])):
        counts[i] = count
    table = table.set_column(table.schema.get_field_index("text"), "text", pa.array(texts, pa.string()))
    table = table.set_column(table.schema.get_field_index("token_count"), "token_count", pa.array(counts, pa.int64()))
    return table.append_column("pii_anonymized_after_labeling", pa.array(changed, pa.bool_()))


def with_gpt2_counts(table: pa.Table, counter: TokenCounter) -> pa.Table:
    """Set ``token_count`` to the GPT-2 count of each text (replacing any existing count)."""

    counts = pa.array(counter.count_batch(table["text"].to_pylist()), pa.int64())
    if "token_count" in table.column_names:
        return table.set_column(table.schema.get_field_index("token_count"), "token_count", counts)
    return table.append_column("token_count", counts)


PoolJob = tuple[Path, Path, str, TokenCounter]  # pool shard, its WebOrganizer sidecar directory, label, token counter


def release_pool_shard(job: PoolJob) -> pa.Table:
    """One pool shard joined with its WebOrganizer sidecar, anonymized where FineWeb's PII step never ran, and labeled."""

    shard, weborganizer_dir, label, counter = job
    table = read_joined(shard, {"weborganizer": weborganizer_dir})
    table = anonymize_rows(table, [not v for v in table["pii_anonymized"].to_pylist()], counter)
    return table.append_column("pangram_label", pa.array([label] * len(table), pa.string()))


class Exporter:
    def __init__(self, release_dir: Path, config: ReleaseConfig, counter: TokenCounter | None = None, workers: int = 1) -> None:
        self.release_dir = release_dir
        self.config = config
        self.counter = counter or Gpt2TokenCounter()
        self.workers = workers

    def writer(self, stack: ExitStack, name: str, schema: pa.Schema) -> ShardedWriter:
        return stack.enter_context(ShardedWriter(self.release_dir / name, schema, self.config.rows_per_shard))

    def pools(self, pools_dir: Path, weborganizer_dir: Path) -> None:
        windows = self.config.include_pangram_windows
        labels_schema = release_schema(PoolRecord, with_text=False, with_windows=windows)
        with ExitStack() as stack:
            labels = self.writer(stack, "labels", labels_schema)
            for pool, label in POOL_LABELS.items():
                out = self.writer(stack, pool, release_schema(PoolRecord, with_windows=windows))
                jobs = [(shard, weborganizer_dir / pool, label, self.counter) for shard in list_shards(pools_dir / pool)]
                for table in self.map(release_pool_shard, jobs, stack):
                    out.write(table)
                    labels.write(table)

    def map(self, function: Callable[[PoolJob], pa.Table], jobs: list[PoolJob], stack: ExitStack) -> Iterator[pa.Table]:
        """``function`` over ``jobs`` in order, in ``workers`` processes (the writers stay in this one)."""
        if self.workers == 1:
            return map(function, jobs)
        return stack.enter_context(get_context("spawn").Pool(self.workers)).imap(function, jobs, chunksize=4)

    def monthly_sample(self, sample_dir: Path, pangram_dir: Path, weborganizer_dir: Path) -> None:
        schema = release_schema(MonthlyRecord, with_windows=self.config.include_pangram_windows)
        with ExitStack() as stack:
            out = self.writer(stack, "monthly_sample", schema)
            for path in sorted(sample_dir.glob("*.parquet")):  # month files, rows kept in file order
                table = read_joined(path, {"pangram": pangram_dir, "weborganizer": weborganizer_dir})
                out.write(anonymize_rows(table, [not v for v in table["pii_anonymized"].to_pylist()], self.counter))

    def filter_audit(self, audit_dir: Path, pangram_dir: Path, weborganizer_dir: Path) -> None:
        sample = audit_dir / "sample" / "part-00000.parquet"
        table = read_joined(sample, {"pangram": pangram_dir / "sample", "weborganizer": weborganizer_dir / "sample"})
        stages = pq.read_table(audit_dir / "fineweb_stages.parquet")
        if stages["id"].to_pylist() != table["id"].to_pylist():
            raise RuntimeError("fineweb_stages.parquet is not aligned with the audit sample")
        for name in stages.column_names[1:]:
            table = table.append_column(name, stages[name])
        table = with_gpt2_counts(table, self.counter)
        table = table.append_column("source", pa.array(["common_crawl"] * len(table), pa.string()))
        table = table.append_column("truncated", pa.array([False] * len(table), pa.bool_()))
        table = anonymize_rows(table, [True] * len(table), self.counter)  # extracted text never went through PII
        schema = release_schema(FilterAuditRecord, with_windows=self.config.include_pangram_windows)
        with ExitStack() as stack:
            self.writer(stack, "filter_audit/fineweb", schema).write(table)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", choices=["pools", "monthly_sample", "filter_audit"])
    parser.add_argument("--config", type=Path, default=default_config("release.yaml"))
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--pools-dir", type=Path)
    parser.add_argument("--sample-dir", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--pangram-dir", type=Path)
    parser.add_argument("--weborganizer-dir", type=Path, required=True)
    parser.add_argument("--include-pangram-windows", action="store_true", help="override the config and export windows")
    parser.add_argument("--workers", type=int, default=1, help="processes that read and anonymize pool shards")
    args = parser.parse_args(argv)
    config = load_config(args.config, ReleaseConfig)
    if args.include_pangram_windows:
        config = config.model_copy(update={"include_pangram_windows": True})
    exporter = Exporter(args.release_dir, config, workers=args.workers)
    needed = {"pools": ["pools_dir"], "monthly_sample": ["sample_dir", "pangram_dir"], "filter_audit": ["audit_dir", "pangram_dir"]}
    missing = [f"--{n.replace('_', '-')}" for n in needed[args.table] if getattr(args, n) is None]
    if missing:
        parser.error(f"{args.table} needs {' '.join(missing)}")
    if args.table == "pools":
        exporter.pools(args.pools_dir, args.weborganizer_dir)
    elif args.table == "monthly_sample":
        exporter.monthly_sample(args.sample_dir, args.pangram_dir, args.weborganizer_dir)
    else:
        exporter.filter_audit(args.audit_dir, args.pangram_dir, args.weborganizer_dir)
    print(f"exported {args.table} to {args.release_dir}")


if __name__ == "__main__":
    main()
