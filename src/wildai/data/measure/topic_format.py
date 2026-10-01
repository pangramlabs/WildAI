"""Documents and GPT-2 tokens by WebOrganizer topic, format and Pangram label.

``monthly`` counts the monthly measurement sample by capture month (``monthly_topic_format.csv``); ``pool`` counts the
WildAI pool (``human``, ``ai`` and ``mixed``) by crawl year (``pool_topic_format.csv``). Documents without a Pangram
label are left out.

    python -m wildai.data.measure.topic_format monthly --sample-dir data/monthly_sample \
        --pangram-dir data/labels/pangram/monthly_sample --weborganizer-dir data/labels/weborganizer/monthly_sample
    python -m wildai.data.measure.topic_format pool --pools-dir data/pools --weborganizer-dir data/labels/weborganizer/pools
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from wildai.data.measure.ai_share import DEFAULT_OUTPUT, write_csv
from wildai.data.parquet_io import list_shards, read_joined

POOL_LABELS = {"human": "Human", "ai": "AI", "mixed": "Mixed"}


class TopicFormatCount(BaseModel):
    model_config = ConfigDict(frozen=True)

    period: str
    topic: str
    format: str
    label: str
    documents: int
    tokens: int


def count(tables: Iterable[pa.Table]) -> list[TopicFormatCount]:
    """Group rows of (period, topic, format, label, token_count) tables; rows with no label are skipped."""

    totals: dict[tuple[str, str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for table in tables:
        columns = [table[c].to_pylist() for c in ("period", "topic", "format", "label", "token_count")]
        for period, topic, fmt, label, tokens in zip(*columns):
            if label is None:
                continue
            cell = totals[(period, topic, fmt, label)]
            cell[0] += 1
            cell[1] += tokens
    return [TopicFormatCount(period=p, topic=t, format=f, label=lab, documents=d, tokens=n)
            for (p, t, f, lab), (d, n) in sorted(totals.items())]


def monthly_tables(sample_dir: Path, pangram_dir: Path, weborganizer_dir: Path) -> Iterator[pa.Table]:
    for path in sorted(sample_dir.glob("*.parquet")):
        table = read_joined(path, {"pangram": pangram_dir, "weborganizer": weborganizer_dir},
                            columns=["id", "month", "token_count"])
        yield pa.table({"period": table["month"], "topic": table["topic"], "format": table["format"],
                        "label": table["pangram_label"], "token_count": table["token_count"]})


def pool_tables(pools_dir: Path, weborganizer_dir: Path) -> Iterator[pa.Table]:
    for pool, label in POOL_LABELS.items():
        for shard in list_shards(pools_dir / pool):
            table = read_joined(shard, {"weborganizer": weborganizer_dir / pool}, columns=["id", "dump", "token_count"])
            years = [dump.split("-")[2] for dump in table["dump"].to_pylist()]  # CC-MAIN-YYYY-WW
            yield pa.table({"period": years, "topic": table["topic"], "format": table["format"],
                            "label": [label] * len(table), "token_count": table["token_count"]})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scope", choices=["monthly", "pool"])
    parser.add_argument("--sample-dir", type=Path, help="monthly: the monthly sample")
    parser.add_argument("--pangram-dir", type=Path, help="monthly: its Pangram sidecars")
    parser.add_argument("--pools-dir", type=Path, help="pool: directory holding the human, ai and mixed pools")
    parser.add_argument("--weborganizer-dir", type=Path, required=True, help="WebOrganizer sidecars (pool: one per pool)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.scope == "monthly":
        if args.sample_dir is None or args.pangram_dir is None:
            parser.error("monthly needs --sample-dir and --pangram-dir")
        rows, name = count(monthly_tables(args.sample_dir, args.pangram_dir, args.weborganizer_dir)), "monthly_topic_format.csv"
    else:
        if args.pools_dir is None:
            parser.error("pool needs --pools-dir")
        rows, name = count(pool_tables(args.pools_dir, args.weborganizer_dir)), "pool_topic_format.csv"
    write_csv(rows, args.output_dir / name)
    print(f"wrote {len(rows):,} cells to {args.output_dir / name}")


if __name__ == "__main__":
    main()
