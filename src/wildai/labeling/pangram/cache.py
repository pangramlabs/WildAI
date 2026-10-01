"""Resumable cache of one shard's Pangram labels and submitted bulk jobs.

Layout of a shard's cache directory:

* ``results-00000.parquet``, ... -- :class:`PangramLabels` rows, one file per collected job (append-only);
* ``jobs.jsonl`` -- a ledger line when a job is submitted and another when it is closed (results stored or expired).

A rerun reuses stored labels (when the submitted text and model selector match) and collects jobs that were submitted but
not yet closed, instead of paying for them again. The API keeps bulk results for 48 hours after a job finishes.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from wildai.data.arrow_schema import arrow_schema, table_from_models
from wildai.data.parquet_io import write_table_atomic
from wildai.labeling.pangram.models import PangramLabels

LABELS_SCHEMA = arrow_schema(PangramLabels)


class JobRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    bulk_id: str
    ids: list[str]
    closed: bool = False
    """Whether the job needs no more work: its results are stored, or it expired before they could be."""


class ShardCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.ledger = directory / "jobs.jsonl"

    def labels(self) -> dict[str, PangramLabels]:
        stored: dict[str, PangramLabels] = {}
        for path in sorted(self.directory.glob("results-*.parquet")):
            for row in pq.read_table(path, schema=LABELS_SCHEMA).to_pylist():
                stored[row["id"]] = PangramLabels.model_validate(row)
        return stored

    def store(self, rows: list[PangramLabels]) -> None:
        if not rows:
            return
        index = len(list(self.directory.glob("results-*.parquet")))
        write_table_atomic(table_from_models(rows, PangramLabels), self.directory / f"results-{index:05d}.parquet")

    def jobs(self) -> list[JobRecord]:
        """Jobs in submission order, with their latest ledger state."""

        if not self.ledger.exists():
            return []
        latest: dict[str, JobRecord] = {}
        for line in self.ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                job = JobRecord.model_validate_json(line)
                latest[job.bulk_id] = job
        return list(latest.values())

    def record(self, job: JobRecord) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a", encoding="utf-8") as handle:
            handle.write(job.model_dump_json() + "\n")
            handle.flush()
