"""Label one shard of documents with Pangram through the Bulk API, reusing and extending the shard's cache.

A document's label is Pangram's ``prediction_short`` (Human, Mixed or AI) together with the fractions of its text
classified AI, AI-assisted and human. Documents longer than ``max_chars`` are labeled on their first ``max_chars``
characters. A result whose model version differs from ``expected_version`` stops the run, so one table never mixes
detector versions.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, get_args

from wildai.labeling.config import PangramSettings
from wildai.labeling.pangram.batching import pack_jobs
from wildai.labeling.pangram.cache import JobRecord, ShardCache
from wildai.labeling.pangram.client import PangramAPIError
from wildai.labeling.pangram.models import (
    BulkItem,
    BulkResult,
    BulkStatus,
    BulkSubmission,
    PangramLabel,
    PangramLabels,
    WindowScore,
)


class BulkApi(Protocol):
    """The Bulk API calls the labeler makes (implemented by :class:`wildai.labeling.pangram.client.PangramClient`)."""

    sleep: Callable[[float], None]

    def submit_bulk(self, items: Sequence[BulkItem], model: str) -> BulkSubmission: ...

    def bulk_status(self, bulk_id: str) -> BulkStatus: ...

    def bulk_results(self, bulk_id: str) -> list[BulkResult]: ...


@dataclass(frozen=True)
class Submission:
    id: str
    text: str
    sha256: str


def submissions(ids: Sequence[str], texts: Sequence[str], max_chars: int) -> dict[str, Submission]:
    """One submission per distinct id; a repeated id must repeat its text."""

    out: dict[str, Submission] = {}
    for doc_id, text in zip(ids, texts, strict=True):
        sent = text[:max_chars]
        sub = Submission(doc_id, sent, hashlib.sha256(sent.encode("utf-8")).hexdigest())
        if out.setdefault(doc_id, sub) != sub:
            raise ValueError(f"document id {doc_id} appears twice with different text")
    return out


def to_labels(result: BulkResult | None, sub: Submission, settings: PangramSettings) -> PangramLabels:
    common = {"id": sub.id, "pangram_model": settings.model, "pangram_sent_chars": len(sub.text),
              "pangram_text_sha256": sub.sha256}
    prediction = result.result if result else None
    if prediction is None or prediction.stage != "STAGE_SUCCESS":
        error = (result.error if result else None) or (prediction.headline if prediction else None) or "no result returned"
        return PangramLabels(**common, pangram_label=None, pangram_fraction_ai=None, pangram_fraction_ai_assisted=None,
                             pangram_fraction_human=None, pangram_version=None, pangram_windows=[], pangram_error=error)
    if settings.expected_version and prediction.version != settings.expected_version:
        raise RuntimeError(f"Pangram returned version {prediction.version!r}, expected {settings.expected_version!r}")
    if prediction.prediction_short not in get_args(PangramLabel):
        raise RuntimeError(f"unexpected Pangram label {prediction.prediction_short!r}")
    windows = [WindowScore(start_index=w.start_index, end_index=w.end_index, label=w.label,
                           ai_assistance_score=w.ai_assistance_score, confidence=w.confidence)
               for w in prediction.windows] if settings.keep_windows else []
    return PangramLabels(**common, pangram_label=prediction.prediction_short, pangram_fraction_ai=prediction.fraction_ai,
                         pangram_fraction_ai_assisted=prediction.fraction_ai_assisted,
                         pangram_fraction_human=prediction.fraction_human, pangram_version=prediction.version,
                         pangram_windows=windows, pangram_error=None)


class PangramLabeler:
    def __init__(self, client: BulkApi, settings: PangramSettings) -> None:
        self.client = client
        self.settings = settings

    def label(self, ids: Sequence[str], texts: Sequence[str], cache: ShardCache) -> list[PangramLabels]:
        """Labels for ``ids`` in input order; cached labels are reused and missing ones are fetched."""

        subs = submissions(ids, texts, self.settings.max_chars)
        for job in cache.jobs():
            if not job.closed:
                self._collect(job, subs, cache)
        todo = [BulkItem(id=s.id, text=s.text) for s in subs.values() if s.id not in self._reusable(cache, subs)]
        self._run(todo, subs, cache)
        stored = self._reusable(cache, subs)
        missing = [i for i in subs if i not in stored]
        if missing:
            raise RuntimeError(f"{len(missing)} documents are still unlabeled, e.g. {missing[0]}")
        return [stored[i] for i in ids]

    def _reusable(self, cache: ShardCache, subs: dict[str, Submission]) -> dict[str, PangramLabels]:
        return {i: lab for i, lab in cache.labels().items()
                if i in subs and lab.pangram_text_sha256 == subs[i].sha256 and lab.pangram_model == self.settings.model}

    def _run(self, todo: list[BulkItem], subs: dict[str, Submission], cache: ShardCache) -> None:
        queue = deque(pack_jobs(todo, max_units=self.settings.max_units_per_job, words_per_unit=self.settings.words_per_unit,
                                max_items=self.settings.max_items_per_job))
        outstanding: list[JobRecord] = []
        while queue or outstanding:
            while queue and len(outstanding) < self.settings.max_outstanding_jobs:
                items = queue.popleft()
                submission = self.client.submit_bulk(items, self.settings.model)
                job = JobRecord(bulk_id=submission.bulk_id, ids=[item.id for item in items])
                cache.record(job)
                outstanding.append(job)
            self.client.sleep(self.settings.poll_interval)
            for job in list(outstanding):
                if self.client.bulk_status(job.bulk_id).terminal:
                    self._store(job, subs, cache)
                    outstanding.remove(job)

    def _collect(self, job: JobRecord, subs: dict[str, Submission], cache: ShardCache) -> None:
        """Finish a job a previous run submitted; an expired job is closed and its documents are submitted again."""

        try:
            while not self.client.bulk_status(job.bulk_id).terminal:
                self.client.sleep(self.settings.poll_interval)
        except PangramAPIError as error:
            if error.status != 404:
                raise
            cache.record(job.model_copy(update={"closed": True}))
            return
        self._store(job, subs, cache)

    def _store(self, job: JobRecord, subs: dict[str, Submission], cache: ShardCache) -> None:
        results = {r.id: r for r in self.client.bulk_results(job.bulk_id) if r.id is not None}
        cache.store([to_labels(results.get(i), subs[i], self.settings) for i in job.ids if i in subs])
        cache.record(job.model_copy(update={"closed": True}))
