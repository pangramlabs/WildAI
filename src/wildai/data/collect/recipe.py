"""FineWeb's processing recipe, built from DataTrove v0.2.0 components (the version FineWeb was produced with).

The order and parameters follow DataTrove's ``examples/fineweb.py``:

1. per batch of WARC files: WarcReader -> URLFilter -> Trafilatura(favour_precision) -> LanguageFilter (English > 0.65)
   -> GopherRepetitionFilter -> GopherQualityFilter -> C4QualityFilter(filter_no_terminal_punct=False)
   -> FineWebQualityFilter;
2. per crawl: MinHash near-duplicate removal (5-grams, 14 buckets x 8 hashes, 64-bit), then PIIFormatter.

Token counts are computed later with the pinned GPT-2 tokenizer (:mod:`wildai.data.tokens`) instead of DataTrove's
unpinned ``TokensCounter``. DataTrove is imported inside the functions: this module needs the collection environment
(Python 3.10, ``numpy==1.26.4``, ``trafilatura==1.8.1``, ``lxml==5.1.1``, ``datatrove[io,processing]==0.2.0``).
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

DATATROVE_VERSION = "0.2.0"
TRAFILATURA_VERSION = "1.8.1"
MINHASH = {"n_grams": 5, "num_buckets": 14, "hashes_per_bucket": 8, "use_64bit_hashes": True}

# FineWeb's filters after extraction, in pipeline order; the filter audit reports survival after each.
FILTER_STAGES = ("language", "gopher_repetition", "gopher_quality", "c4", "fineweb_quality")


def check_environment() -> None:
    """Fail fast if the pinned recipe versions are not installed; other versions change which documents survive."""

    for package, expected in (("datatrove", DATATROVE_VERSION), ("trafilatura", TRAFILATURA_VERSION)):
        found = metadata.version(package)
        if found != expected:
            raise RuntimeError(f"the FineWeb recipe needs {package}=={expected}, found {found}")


def quality_filters() -> list[tuple[str, object]]:
    """FineWeb's filters after extraction, as (stage name, DataTrove filter) pairs in pipeline order."""

    from datatrove.pipeline.filters import (
        C4QualityFilter,
        FineWebQualityFilter,
        GopherQualityFilter,
        GopherRepetitionFilter,
        LanguageFilter,
    )

    filters = [LanguageFilter(), GopherRepetitionFilter(), GopherQualityFilter(),
               C4QualityFilter(filter_no_terminal_punct=False), FineWebQualityFilter(exclusion_writer=None)]
    return list(zip(FILTER_STAGES, filters))


def extraction_steps(raw_dir: Path, dump: str, timeout: float) -> list[object]:
    """Read WARC files, drop blocklisted URLs and extract main text with Trafilatura (FineWeb's settings)."""

    from datatrove.pipeline.extractors import Trafilatura
    from datatrove.pipeline.filters import URLFilter
    from datatrove.pipeline.readers import WarcReader

    return [
        WarcReader(str(raw_dir), glob_pattern="*.warc.gz", default_metadata={"dump": dump}),
        URLFilter(),
        Trafilatura(favour_precision=True, timeout=timeout),
    ]


def filter_pipeline(raw_dir: Path, output_dir: Path, dump: str, timeout: float, output_filename: str) -> list[object]:
    """Extraction followed by every FineWeb quality filter, writing survivors as JSONL."""

    from datatrove.pipeline.writers.jsonl import JsonlWriter

    return [*extraction_steps(raw_dir, dump, timeout), *(f for _name, f in quality_filters()),
            JsonlWriter(str(output_dir), output_filename=output_filename)]


def run_local(pipeline: list[object], tasks: int, workers: int, logging_dir: Path) -> None:
    from datatrove.executor.local import LocalPipelineExecutor

    LocalPipelineExecutor(pipeline=pipeline, tasks=tasks, workers=min(workers, tasks), logging_dir=str(logging_dir)).run()


def deduplicate_and_anonymize(filtered_dir: Path, work_dir: Path, output_dir: Path, *, minhash: bool, pii: bool,
                              workers: int) -> None:
    """Per-crawl MinHash (optional) and PII anonymization (optional) of the filtered JSONL files."""

    from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter, MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
    from datatrove.pipeline.formatters import PIIFormatter
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers.jsonl import JsonlWriter

    files = sorted(filtered_dir.glob("*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(f"no filtered documents in {filtered_dir}")
    tasks = len(files)  # signature and filter stages must shard the input identically
    logs = work_dir / "logs"
    final_steps: list[object] = [JsonlReader(str(filtered_dir))]
    if minhash:
        config = MinhashConfig(**MINHASH)
        signatures, buckets, remove_ids = (work_dir / "minhash" / p for p in ("signatures", "buckets", "remove_ids"))
        run_local([JsonlReader(str(filtered_dir)), MinhashDedupSignature(output_folder=str(signatures), config=config)],
                  tasks, workers, logs / "minhash_signatures")
        run_local([MinhashDedupBuckets(input_folder=str(signatures), output_folder=str(buckets), config=config)],
                  config.num_buckets, workers, logs / "minhash_buckets")
        run_local([MinhashDedupCluster(input_folder=str(buckets), output_folder=str(remove_ids), config=config)],
                  1, 1, logs / "minhash_cluster")
        final_steps.append(MinhashDedupFilter(input_folder=str(remove_ids)))
    if pii:
        final_steps.append(PIIFormatter())
    final_steps.append(JsonlWriter(str(output_dir)))
    run_local(final_steps, tasks, workers, logs / "finalize")
