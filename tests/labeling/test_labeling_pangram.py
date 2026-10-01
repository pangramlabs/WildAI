"""The Pangram API client, job packing and the resumable labeler, against a fake API."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from pathlib import Path

import pytest
import requests

from wildai.labeling.config import PangramSettings
from wildai.labeling.pangram.batching import billable_units, pack_jobs
from wildai.labeling.pangram.cache import JobRecord, ShardCache
from wildai.labeling.pangram.client import PangramAPIError, PangramClient, RetryPolicy
from wildai.labeling.pangram.labeler import PangramLabeler
from wildai.labeling.pangram.models import BulkItem, BulkResult, BulkStatus, BulkSubmission, Prediction, Window


class Response:
    def __init__(self, status: int, payload: dict | None = None, headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self.payload = payload or {}
        self.headers = headers or {}
        self.text = str(payload)

    def json(self) -> dict:
        return self.payload


class ScriptedSession(requests.Session):
    """Returns the scripted responses in order and records every request."""

    def __init__(self, responses: list[Response]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.requests: list[tuple[str, str]] = []

    def request(self, method: str, url: str, **kwargs: object) -> Response:  # type: ignore[override]
        self.requests.append((method, url))
        return self.responses.pop(0)


def client_with(responses: list[Response]) -> tuple[PangramClient, ScriptedSession, list[float]]:
    session, sleeps = ScriptedSession(responses), []
    return PangramClient("key", session=session, sleep=sleeps.append, retry=RetryPolicy(attempts=3)), session, sleeps


def test_billable_units_and_packing() -> None:
    assert billable_units("", 100) == 1 and billable_units("w " * 101, 100) == 2
    items = [BulkItem(id=str(i), text="w " * 250) for i in range(7)]  # 3 units each
    jobs = pack_jobs(items, max_units=10, words_per_unit=100)
    assert [len(j) for j in jobs] == [3, 3, 1]
    assert [len(j) for j in pack_jobs(items, max_units=100, max_items=2)] == [2, 2, 2, 1]
    with pytest.raises(ValueError):
        pack_jobs([BulkItem(id="big", text="w " * 5000)], max_units=10)


def test_client_retries_reads_and_rate_limits() -> None:
    client, session, sleeps = client_with([Response(503), Response(429, headers={"Retry-After": "2"}),
                                           Response(200, {"models": ["default"]})])
    assert client.list_models() == ["default"]
    assert sleeps == [1.0, 2.0] and session.headers["x-api-key"] == "key"


def test_client_does_not_retry_ambiguous_submissions() -> None:
    client, session, _ = client_with([Response(500, {"detail": "boom"})])
    with pytest.raises(PangramAPIError) as error:
        client.submit_bulk([BulkItem(id="a", text="t")], "default")
    assert error.value.status == 500 and len(session.requests) == 1
    client, session, _ = client_with([Response(429), Response(202, {"bulk_id": "b", "status": "queued", "total_items": 1})])
    assert client.submit_bulk([BulkItem(id="a", text="t")], "default").bulk_id == "b"


def test_bulk_results_pages_through_successes_and_failures() -> None:
    page = {"bulk_id": "b", "offset": 0, "limit": 1000, "total_items": 2,
            "items": [{"index": 0, "id": "a", "stage": "STAGE_SUCCESS", "result": {"stage": "STAGE_SUCCESS",
                                                                                   "prediction_short": "AI"}}],
            "failed_items": [{"index": 1, "id": "c", "stage": "STAGE_FAILED", "error": "no text"}]}
    client, _, _ = client_with([Response(200, page)])
    results = client.bulk_results("b")
    assert [(r.id, r.error) for r in results] == [("a", None), ("c", "no text")]


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PANGRAM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PANGRAM_API_KEY"):
        PangramClient.from_env()


class FakeBulkApi:
    """In-memory Bulk API: every job finishes at the first status check; labels depend on the text."""

    def __init__(self, version: str = "3.3.2", expired: set[str] | None = None) -> None:
        self.jobs: dict[str, list[BulkItem]] = {}
        self.submitted = 0
        self.version = version
        self.expired = expired or set()
        self.ids = itertools.count()

    def sleep(self, _seconds: float) -> None:
        return None

    def submit_bulk(self, items: Sequence[BulkItem], model: str) -> BulkSubmission:
        self.submitted += len(items)
        bulk_id = f"bulk-{next(self.ids)}"
        self.jobs[bulk_id] = list(items)
        return BulkSubmission(bulk_id=bulk_id, status="queued", total_items=len(items))

    def bulk_status(self, bulk_id: str) -> BulkStatus:
        if bulk_id in self.expired:
            raise PangramAPIError(404, "not found")
        return BulkStatus(bulk_id=bulk_id, status="succeeded", total_items=len(self.jobs[bulk_id]))

    def bulk_results(self, bulk_id: str) -> list[BulkResult]:
        out = []
        for index, item in enumerate(self.jobs[bulk_id]):
            if not item.text.strip():
                out.append(BulkResult(index=index, id=item.id, stage="STAGE_FAILED", error="no valid text"))
                continue
            label = "AI" if "delve" in item.text else "Human"
            window = Window(label="AI-Generated" if label == "AI" else "Human Written", ai_assistance_score=0.9,
                            confidence="High", start_index=0, end_index=len(item.text))
            out.append(BulkResult(index=index, id=item.id, stage="STAGE_SUCCESS", result=Prediction(
                stage="STAGE_SUCCESS", version=self.version, prediction_short=label, fraction_ai=float(label == "AI"),
                fraction_human=float(label == "Human"), windows=[window])))
        return out


def test_labeler_labels_caches_and_resumes(tmp_path: Path) -> None:
    api = FakeBulkApi()
    settings = PangramSettings(expected_version="3.3.2", max_items_per_job=2, max_chars=20)
    labeler = PangramLabeler(api, settings)
    ids = ["a", "b", "c", "a"]
    texts = ["let us delve deeper", "plain human words", "   ", "let us delve deeper"]
    labels = labeler.label(ids, texts, ShardCache(tmp_path))
    assert [lab.pangram_label for lab in labels] == ["AI", "Human", None, "AI"]
    assert labels[2].pangram_error == "no valid text" and labels[0].pangram_windows[0].label == "AI-Generated"
    assert api.submitted == 3  # the repeated id is labeled once
    again = labeler.label(ids, texts, ShardCache(tmp_path))
    assert again == labels and api.submitted == 3  # everything came from the cache
    longer = labeler.label(["b"], ["plain human words, now edited"], ShardCache(tmp_path))
    assert api.submitted == 4 and longer[0].pangram_sent_chars == 20  # changed text is relabeled, on its prefix


def test_labeler_collects_or_resubmits_open_jobs(tmp_path: Path) -> None:
    api = FakeBulkApi(expired={"old"})
    cache = ShardCache(tmp_path)
    api.jobs["open"] = [BulkItem(id="a", text="delve")]
    cache.record(JobRecord(bulk_id="open", ids=["a"]))
    cache.record(JobRecord(bulk_id="old", ids=["b"]))
    labels = PangramLabeler(api, PangramSettings()).label(["a", "b"], ["delve", "human"], cache)
    assert [lab.pangram_label for lab in labels] == ["AI", "Human"]
    assert api.submitted == 1  # only the expired job's document was paid for again
    assert all(job.closed for job in cache.jobs())


def test_labeler_refuses_another_detector_version(tmp_path: Path) -> None:
    labeler = PangramLabeler(FakeBulkApi(version="4.0"), PangramSettings(expected_version="3.3.2"))
    with pytest.raises(RuntimeError, match="version"):
        labeler.label(["a"], ["text"], ShardCache(tmp_path))


def test_labeler_rejects_conflicting_duplicates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="different text"):
        PangramLabeler(FakeBulkApi(), PangramSettings()).label(["a", "a"], ["one", "two"], ShardCache(tmp_path))
