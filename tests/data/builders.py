"""Builders of tiny tables for the data-pipeline tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from wildai.data.arrow_schema import arrow_schema
from wildai.data.schema import CandidateDocument, WebDocument


class WordCounter:
    """Counts whitespace-separated words; stands in for the GPT-2 counter so tests need no download."""

    def count_batch(self, texts: Sequence[str]) -> list[int]:
        return [len(t.split()) for t in texts]


def web_documents(ids: Sequence[str], dump: str = "CC-MAIN-2026-25", *, texts: Sequence[str] | None = None,
                  tokens: Sequence[int] | None = None, pii_anonymized: bool = True) -> pa.Table:
    texts = list(texts) if texts is not None else [f"document {i} " * 60 for i in ids]
    rows = [{"id": i, "text": t, "url": f"https://example.com/{i}", "date": "2026-06-10T12:00:00Z", "dump": dump,
             "source": "common_crawl", "warc_path": f"crawl-data/{dump}/segments/1/warc/{i}.warc.gz",
             "language_score": 0.9, "token_count": (tokens[k] if tokens else len(t.split())),
             "pii_anonymized": pii_anonymized, "truncated": False}
            for k, (i, t) in enumerate(zip(ids, texts))]
    return pa.Table.from_pylist(rows, schema=arrow_schema(WebDocument))


def candidates(ids: Sequence[str], dump: str = "CC-MAIN-2026-25", selection: str = "editlens_ai", **kwargs: object) -> pa.Table:
    table = web_documents(ids, dump, **kwargs)
    n = len(table)
    table = table.append_column("selection", pa.array([selection] * n))
    table = table.append_column("editlens_bucket", pa.array([3] * n, pa.int64()))
    table = table.append_column("editlens_score", pa.array([0.9] * n))
    return table.cast(arrow_schema(CandidateDocument))


def write(table: pa.Table, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def pangram_sidecar(ids: Sequence[str], labels: Sequence[str | None]) -> pa.Table:
    """A Pangram sidecar with the columns of :class:`wildai.labeling.pangram.models.PangramLabels`."""

    from wildai.data.arrow_schema import table_from_models
    from wildai.labeling.pangram.models import PangramLabels, WindowScore

    rows = [PangramLabels(id=i, pangram_label=lab, pangram_fraction_ai=1.0 if lab == "AI" else 0.0,
                          pangram_fraction_ai_assisted=0.0, pangram_fraction_human=0.0 if lab == "AI" else 1.0,
                          pangram_version="test" if lab else None, pangram_model="default",
                          pangram_windows=[WindowScore(start_index=0, end_index=10, label="x", ai_assistance_score=0.5,
                                                       confidence="High")] if lab else [],
                          pangram_sent_chars=10, pangram_text_sha256="0" * 64, pangram_error=None if lab else "failed")
            for i, lab in zip(ids, labels)]
    return table_from_models(rows, PangramLabels)
