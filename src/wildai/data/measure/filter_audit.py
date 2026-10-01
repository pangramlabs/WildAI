"""The filter-survival audit: which documents of one crawl survive each FineWeb filter stage.

1. ``warc_files`` WARC files are drawn uniformly from the crawl;
2. every page passes FineWeb's URL filter and Trafilatura extraction (the audit population), and ``documents`` of them
   are drawn by the smallest keyed hash of their id (``sample/part-00000.parquet``);
3. FineWeb's filters run in pipeline order on each sampled document (C4's rules also remove lines, so later stages see
   the edited text), recording whether the document is still alive after each stage (``fineweb_stages.parquet``).

Pangram (and WebOrganizer) then label the ``sample`` directory; :mod:`wildai.data.measure.filter_survival` combines the two.
MinHash is not audited: duplicates in WARC files outside the sample cannot be seen.

Run in the collection environment (see :mod:`wildai.data.collect.recipe`):

    python -m wildai.data.measure.filter_audit --work-dir work/audit --output-dir data/filter_audit
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from wildai.data.arrow_schema import arrow_schema, table_from_models
from wildai.data.collect import recipe
from wildai.data.collect.common_crawl import DONE, download_all, iter_jsonl, load_or_plan_frame
from wildai.data.config import StrictModel, default_config, load_config
from wildai.data.hashing import bottom_k, seeded_key
from wildai.data.parquet_io import write_table_atomic


class FilterAuditConfig(StrictModel):
    dump: str
    warc_files: int = Field(gt=0)
    documents: int = Field(gt=0)
    seed: int
    extraction_timeout: float = 0.1


class AuditDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    url: str
    date: str
    dump: str
    warc_path: str


class FineWebStages(BaseModel):
    """Whether a document is still alive after each FineWeb stage (cumulative), and where it was removed."""

    model_config = ConfigDict(frozen=True)

    id: str
    language: str
    language_score: float
    english: bool
    gopher_repetition: bool
    gopher_quality: bool
    c4: bool
    fineweb_quality: bool
    removed_at: str | None
    removal_reason: str | None


def extract_population(config: FilterAuditConfig, work: Path, workers: int) -> Path:
    """URL filter + Trafilatura on the audit's WARC files; returns the directory of extracted JSONL."""

    frame = load_or_plan_frame(work, config.dump, config.warc_files, config.seed)
    extracted = work / "extracted"
    if not (extracted / DONE).exists():
        from datatrove.pipeline.writers.jsonl import JsonlWriter

        raw = work / "raw"
        download_all(frame.paths, raw, workers)
        steps = [*recipe.extraction_steps(raw, config.dump, config.extraction_timeout), JsonlWriter(str(extracted))]
        recipe.run_local(steps, len(frame.paths), workers, work / "logs")
        shutil.rmtree(raw)
        (extracted / DONE).touch()
    return extracted


def sample_documents(config: FilterAuditConfig, extracted: Path, frame_paths: list[str]) -> pa.Table:
    crawl_path = {Path(p).name: p for p in frame_paths}
    rows = ({"id": r["id"], "text": r["text"], "url": r["metadata"].get("url", ""), "date": r["metadata"].get("date", ""),
             "dump": config.dump, "warc_path": crawl_path[Path(r["metadata"]["file_path"]).name]}
            for r in iter_jsonl(extracted))
    chosen = bottom_k(rows, config.documents, seeded_key("filter-audit", config.seed), id_of=lambda r: r["id"])
    return pa.Table.from_pylist(chosen, schema=arrow_schema(AuditDocument))


def score_stages(sample: pa.Table) -> pa.Table:
    """Run FineWeb's filters in order on every sampled document."""

    from datatrove.data import Document

    filters = recipe.quality_filters()
    rows = []
    for doc_id, text, url in zip(sample["id"].to_pylist(), sample["text"].to_pylist(), sample["url"].to_pylist()):
        document = Document(text=text, id=doc_id, metadata={"url": url})
        alive, removed_at, reason = True, None, None
        passed: dict[str, bool] = {}
        for stage, fineweb_filter in filters:
            if alive:
                verdict = fineweb_filter.filter(document)
                keep, why = verdict if isinstance(verdict, tuple) else (bool(verdict), None)
                if not keep:
                    alive, removed_at, reason = False, stage, why
            passed[stage] = alive
        rows.append(FineWebStages(id=doc_id, language=str(document.metadata.get("language", "")),
                                  language_score=float(document.metadata.get("language_score", 0.0)),
                                  english=passed["language"], gopher_repetition=passed["gopher_repetition"],
                                  gopher_quality=passed["gopher_quality"], c4=passed["c4"],
                                  fineweb_quality=passed["fineweb_quality"], removed_at=removed_at, removal_reason=reason))
    return table_from_models(rows, FineWebStages)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=default_config("filter_audit.yaml"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    config = load_config(args.config, FilterAuditConfig)
    recipe.check_environment()
    extracted = extract_population(config, args.work_dir, args.workers)
    sample_path = args.output_dir / "sample" / "part-00000.parquet"
    if not sample_path.exists():
        frame = load_or_plan_frame(args.work_dir, config.dump, config.warc_files, config.seed)
        write_table_atomic(sample_documents(config, extracted, frame.paths), sample_path)
    stages = score_stages(pq.read_table(sample_path))
    write_table_atomic(stages, args.output_dir / "fineweb_stages.parquet")
    print(f"{len(stages):,} documents; {sum(stages['fineweb_quality'].to_pylist()):,} survive every FineWeb stage")


if __name__ == "__main__":
    main()
