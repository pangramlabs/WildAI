"""Run the FineWeb recipe on Common Crawl WARC files of crawls after FineWeb's last release.

For each crawl: choose the WARC frame (the whole crawl, or ``warc_files_per_crawl`` files drawn uniformly), then per
batch of WARC files download, extract and filter (raw files are deleted once the batch is done), then deduplicate the
crawl with MinHash and anonymize PII, and finally write :class:`wildai.data.schema.WebDocument` Parquet with pinned GPT-2
token counts. Every step leaves a marker, so an interrupted run resumes where it stopped.

Run in the collection environment (see :mod:`wildai.data.collect.recipe`):

    python -m wildai.data.collect.common_crawl --work-dir work/cc --output-dir data/documents/common_crawl
    python -m wildai.data.collect.common_crawl --work-dir work/cc --output-dir ... --dumps CC-MAIN-2026-25 --workers 32
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa

from wildai.data.arrow_schema import arrow_schema
from wildai.data.collect import recipe
from wildai.data.collect.config import CollectionConfig, RecipeSettings
from wildai.data.commoncrawl import WarcFrame, crawl_warc_paths, download, select_frame
from wildai.data.config import default_config, load_config
from wildai.data.parquet_io import ShardedWriter
from wildai.data.schema import WebDocument
from wildai.data.tokens import Gpt2TokenCounter, TokenCounter

DOCUMENT_SCHEMA = arrow_schema(WebDocument)
DONE = "_done"


def load_or_plan_frame(work: Path, dump: str, files: int | None, seed: int) -> WarcFrame:
    """The crawl's WARC frame, planned once and then read back, so a resumed run never redraws it."""

    path = work / "frame.json"
    if path.exists():
        frame = WarcFrame.model_validate_json(path.read_text())
        if (frame.dump, frame.seed) != (dump, seed):
            raise ValueError(f"{path} was planned for {frame.dump} with seed {frame.seed}")
        return frame
    frame = select_frame(dump, crawl_warc_paths(dump), files, seed)
    work.mkdir(parents=True, exist_ok=True)
    path.write_text(frame.model_dump_json(indent=1))
    return frame


def download_all(paths: list[str], directory: Path, workers: int) -> None:
    """Download WARC files in parallel (at most 8 at a time, to stay polite to Common Crawl's servers)."""

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
        list(pool.map(lambda path: download(path, directory / Path(path).name), paths))


def filter_batches(frame: WarcFrame, settings: RecipeSettings, work: Path, workers: int) -> Path:
    """Download, extract and filter every batch of the frame; returns the directory of filtered JSONL."""

    filtered = work / "filtered"
    size = settings.batch_files
    for number, start in enumerate(range(0, len(frame.paths), size)):
        batch = work / "batches" / f"{number:05d}"
        if (batch / DONE).exists():
            continue
        paths = frame.paths[start:start + size]
        raw = batch / "raw"
        download_all(paths, raw, workers)
        pipeline = recipe.filter_pipeline(raw, filtered, frame.dump, settings.extraction_timeout,
                                          output_filename=f"b{number:05d}_${{rank}}.jsonl.gz")
        recipe.run_local(pipeline, len(paths), workers, batch / "logs")
        shutil.rmtree(raw)
        (batch / DONE).write_text(json.dumps({"warc_paths": paths}))
    return filtered


def iter_jsonl(directory: Path) -> Iterator[dict]:
    for path in sorted(directory.glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield from (json.loads(line) for line in handle if line.strip())


def materialize(final: Path, frame: WarcFrame, settings: RecipeSettings, output: Path, batch_rows: int = 4096,
                counter: TokenCounter | None = None) -> int:
    """DataTrove JSONL -> WebDocument Parquet with pinned GPT-2 counts; returns the number of documents written."""

    crawl_path = {Path(p).name: p for p in frame.paths}
    counter = counter or Gpt2TokenCounter()
    written = 0

    def rows(records: list[dict]) -> pa.Table:
        texts = [r["text"] for r in records]
        counts = counter.count_batch(texts)
        return pa.Table.from_pylist([
            {"id": r["id"], "text": r["text"], "url": r["metadata"].get("url", ""), "date": r["metadata"].get("date", ""),
             "dump": frame.dump, "source": "common_crawl", "warc_path": crawl_path[Path(r["metadata"]["file_path"]).name],
             "language_score": float(r["metadata"].get("language_score", 0.0)), "token_count": count,
             "pii_anonymized": settings.pii, "truncated": False}
            for r, count in zip(records, counts)
        ], schema=DOCUMENT_SCHEMA)

    with ShardedWriter(output, DOCUMENT_SCHEMA) as writer:
        pending: list[dict] = []
        for record in iter_jsonl(final):
            if len(record["text"]) < settings.min_chars:
                continue
            pending.append(record)
            if len(pending) == batch_rows:
                writer.write(rows(pending))
                written, pending = written + len(pending), []
        if pending:
            writer.write(rows(pending))
            written += len(pending)
    return written


def run_crawl(frame: WarcFrame, settings: RecipeSettings, work: Path, output: Path, workers: int,
              counter: TokenCounter | None = None) -> int:
    """The whole recipe for one frame, resumable; returns the number of documents materialized."""

    if (output / DONE).exists():
        return json.loads((output / DONE).read_text())["documents"]
    recipe.check_environment()
    filtered = filter_batches(frame, settings, work, workers)
    final = work / "final"
    if not (final / DONE).exists():
        shutil.rmtree(final, ignore_errors=True)
        recipe.deduplicate_and_anonymize(filtered, work, final, minhash=settings.minhash, pii=settings.pii, workers=workers)
        (final / DONE).touch()
    documents = materialize(final, frame, settings, output, counter=counter)
    (output / DONE).write_text(json.dumps({"documents": documents, "frame": frame.model_dump(exclude={"paths"}),
                                           "recipe": settings.model_dump()}))
    return documents


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=default_config("collection.yaml"))
    parser.add_argument("--work-dir", type=Path, required=True, help="scratch space for WARC batches and DataTrove state")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dumps", nargs="+", help="subset of the configured crawls")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    settings = load_config(args.config, CollectionConfig).common_crawl
    for dump in args.dumps or settings.dumps:
        if dump not in settings.dumps:
            parser.error(f"{dump} is not a configured crawl")
        work = args.work_dir / dump
        frame = load_or_plan_frame(work, dump, settings.warc_files_per_crawl, settings.seed)
        documents = run_crawl(frame, settings.recipe, work, args.output_dir / dump, args.workers)
        print(f"{dump}: {documents:,} documents from {len(frame.paths):,} of {frame.crawl_files:,} WARC files")


if __name__ == "__main__":
    main()
