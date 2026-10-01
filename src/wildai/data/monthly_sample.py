"""The monthly measurement sample: ``documents_per_month`` documents per crawl month, each month covering its whole crawl.

* Months FineWeb has released come from Hugging Face FineWeb: row groups of the month's dump(s) are read in a seeded
  random order until ``fineweb_pool_factor`` x N documents captured in that month are gathered.
* Later months come from ``warc_files_per_crawl`` WARC files drawn uniformly from the crawl and run through the FineWeb
  recipe (:mod:`wildai.data.collect.common_crawl`; needs the collection environment).

In both cases the month's N documents are the ones with the smallest keyed hash of their id, keyed by the month's seed
(frame seed + ``YYYYMM``), which also seeds the month's bootstrap interval in :mod:`wildai.data.measure.ai_share`. A
document's month is its capture month (``date[:7]``).

    python -m wildai.data.monthly_sample --output-dir data/monthly_sample --work-dir work/monthly
    python -m wildai.data.monthly_sample --output-dir data/monthly_sample --work-dir work/monthly --months 2026-07 2026-08

Output: ``<output-dir>/<YYYY-MM>.parquet`` with the :class:`wildai.data.schema.MonthlyDocument` schema, in hash order.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.compute as pc
from pydantic import Field

from wildai.data.arrow_schema import arrow_schema
from wildai.data.collect.common_crawl import load_or_plan_frame, run_crawl
from wildai.data.collect.config import RecipeSettings
from wildai.data.collect.fineweb import to_documents
from wildai.data.config import StrictModel, default_config, load_config
from wildai.data.fineweb_hf import FINEWEB_REVISION, ParquetDump, iter_row_groups
from wildai.data.hashing import bottom_k, seeded_key
from wildai.data.parquet_io import read_dataset, write_table_atomic
from wildai.data.schema import MonthlyDocument

MonthlySource = Literal["fineweb", "common_crawl_random_warc"]
MONTHLY_SCHEMA = arrow_schema(MonthlyDocument)


class MonthlyFrame(StrictModel):
    """Months drawn the same way, with one base seed."""

    source: MonthlySource
    seed: int
    months: dict[str, list[str]]
    """``YYYY-MM`` -> the crawl(s) whose captures fall in that month."""


class MonthlySampleConfig(StrictModel):
    documents_per_month: int = Field(5000, gt=0)
    fineweb_revision: str = FINEWEB_REVISION
    fineweb_pool_factor: int = Field(4, gt=0)
    warc_files_per_crawl: int = Field(32, gt=0)
    recipe: RecipeSettings = RecipeSettings()
    frames: list[MonthlyFrame]

    def month(self, month: str) -> tuple[MonthlyFrame, list[str]]:
        for frame in self.frames:
            if month in frame.months:
                return frame, frame.months[month]
        raise KeyError(f"{month} is not a configured month")

    def all_months(self) -> list[str]:
        return sorted(m for frame in self.frames for m in frame.months)


def month_seed(frame_seed: int, month: str) -> int:
    """The month's seed: its frame's seed plus ``YYYYMM``."""

    return frame_seed + int(month.replace("-", ""))


def in_month(table: pa.Table, month: str) -> pa.Table:
    return table.filter(pc.equal(pc.utf8_slice_codeunits(table["date"], 0, 7), month))


def select_month(tables: Iterator[pa.Table], month: str, seed: int, n: int) -> pa.Table:
    """The ``n`` documents with the smallest month-keyed hash, as MonthlyDocument rows in hash order."""

    rows = (row for table in tables for row in in_month(table, month).to_pylist())
    chosen = bottom_k(rows, n, seeded_key(f"monthly/{month}", seed), id_of=lambda r: r["id"])
    if len(chosen) < n:
        raise RuntimeError(f"{month}: only {len(chosen)} eligible documents for a sample of {n}")
    return pa.Table.from_pylist([{**r, "month": month} for r in chosen], schema=MONTHLY_SCHEMA)


def fineweb_candidates(dumps: list[str], month: str, seed: int, config: MonthlySampleConfig) -> Iterator[pa.Table]:
    """Row groups of the month's dump(s) in seeded order, until ``fineweb_pool_factor`` x N in-month documents are read."""

    sources = {name: ParquetDump.fineweb(name, config.fineweb_revision) for name in dumps}
    needed, found = config.documents_per_month * config.fineweb_pool_factor, 0
    for table in iter_row_groups(sources, seed):
        table = to_documents(table, config.recipe.min_chars)
        found += len(in_month(table, month))
        yield table
        if found >= needed:
            return


def build_month(month: str, config: MonthlySampleConfig, work: Path, workers: int) -> pa.Table:
    frame, dumps = config.month(month)
    seed = month_seed(frame.seed, month)
    if frame.source == "fineweb":
        tables = fineweb_candidates(dumps, month, seed, config)
    else:
        (dump,) = dumps
        crawl_work = work / dump
        warcs = load_or_plan_frame(crawl_work, dump, config.warc_files_per_crawl, frame.seed)
        documents = work / "documents" / dump
        run_crawl(warcs, config.recipe, crawl_work, documents, workers)
        tables = iter([read_dataset(documents)])
    return select_month(tables, month, seed, config.documents_per_month)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=default_config("monthly_sample.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="scratch space for the WARC-file months")
    parser.add_argument("--months", nargs="+", help="subset of the configured months (YYYY-MM)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    config = load_config(args.config, MonthlySampleConfig)
    for month in args.months or config.all_months():
        path = args.output_dir / f"{month}.parquet"
        if path.exists():
            print(f"{month}: exists, skipped")
            continue
        table = build_month(month, config, args.work_dir, args.workers)
        write_table_atomic(table, path)
        print(f"{month}: {len(table):,} documents")


if __name__ == "__main__":
    main()
