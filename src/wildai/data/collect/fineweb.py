"""Sample documents from Hugging Face FineWeb dumps.

For each dump, Parquet row groups are read in a seeded random order (see :mod:`wildai.data.fineweb_hf`) and every document
of at least ``min_chars`` characters is kept, until the dump's GPT-2 token target is reached. Text is never truncated.
FineWeb has already applied its language, quality, MinHash and PII steps.

    python -m wildai.data.collect.fineweb --output-dir data/documents/fineweb
    python -m wildai.data.collect.fineweb --output-dir data/documents/fineweb --dumps CC-MAIN-2025-26

Output: ``<output-dir>/<dump>/part-*.parquet`` with the :class:`wildai.data.schema.WebDocument` schema, and a
``_done`` marker per dump (a completed dump is skipped when the command is rerun).
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from wildai.data.arrow_schema import arrow_schema
from wildai.data.collect.config import CollectionConfig, FineWebCollection
from wildai.data.config import default_config, load_config
from wildai.data.fineweb_hf import ParquetDump, iter_row_groups, warc_path
from wildai.data.parquet_io import ShardedWriter
from wildai.data.schema import WebDocument

DOCUMENT_SCHEMA = arrow_schema(WebDocument)
DONE = "_done"


def to_documents(table: pa.Table, min_chars: int) -> pa.Table:
    """FineWeb rows as :class:`WebDocument` rows, dropping texts shorter than ``min_chars``."""

    table = table.filter(pc.greater_equal(pc.utf8_length(table["text"]), min_chars))
    n = len(table)
    columns = {
        "id": table["id"], "text": table["text"], "url": table["url"], "date": table["date"], "dump": table["dump"],
        "source": pa.array(["fineweb"] * n, pa.string()),
        "warc_path": pa.array([warc_path(p) for p in table["file_path"].to_pylist()], pa.string()),
        "language_score": table["language_score"], "token_count": table["token_count"],
        "pii_anonymized": pa.array([True] * n, pa.bool_()), "truncated": pa.array([False] * n, pa.bool_()),
    }
    return pa.table(columns).cast(DOCUMENT_SCHEMA)


def take_tokens(tables: Iterable[pa.Table], target_tokens: int) -> Iterator[pa.Table]:
    """Yield tables until their summed ``token_count`` reaches ``target_tokens``; the last table is cut at the target."""

    total = 0
    for table in tables:
        counts = table["token_count"].to_pylist()
        for i, count in enumerate(counts):
            total += count
            if total >= target_tokens:
                yield table.slice(0, i + 1)
                return
        yield table


def collect_dump(dump: ParquetDump, name: str, settings: FineWebCollection, output: Path) -> dict[str, int]:
    """Collect one dump into ``output``; returns document and token counts."""

    tables = (to_documents(t, settings.min_chars) for t in iter_row_groups({name: dump}, settings.seed))
    documents = tokens = 0
    with ShardedWriter(output, DOCUMENT_SCHEMA) as writer:
        for table in take_tokens(tables, settings.target_tokens_per_dump):
            writer.write(table)
            documents += len(table)
            tokens += int(pc.sum(table["token_count"]).as_py() or 0)
    return {"documents": documents, "tokens": tokens}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=default_config("collection.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dumps", nargs="+", help="subset of the configured dumps")
    args = parser.parse_args(argv)
    settings = load_config(args.config, CollectionConfig).fineweb
    for name in args.dumps or settings.dumps:
        if name not in settings.dumps:
            parser.error(f"{name} is not a configured FineWeb dump")
        output = args.output_dir / name
        if (output / DONE).exists():
            print(f"{name}: complete, skipped")
            continue
        shutil.rmtree(output, ignore_errors=True)  # an interrupted dump restarts from scratch (the order is seeded)
        counts = collect_dump(ParquetDump.fineweb(name, settings.revision), name, settings, output)
        (output / DONE).write_text(json.dumps({"dump": name, **counts, "settings": settings.model_dump()}, indent=1))
        print(f"{name}: {counts['documents']:,} documents, {counts['tokens']:,} GPT-2 tokens")


if __name__ == "__main__":
    main()
