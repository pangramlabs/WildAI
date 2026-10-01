"""HTTP client for the public Pangram API, with retries and exponential backoff.

Endpoints (https://docs.pangram.com/api-reference/introduction), all authenticated with the ``x-api-key`` header:
``GET /models``, ``POST /task`` + ``GET /task/{id}`` (one text), ``POST /bulk`` + ``GET /bulk/{id}`` +
``GET /bulk/{id}/results`` (many texts). The key is read from ``PANGRAM_API_KEY`` and never logged.

Retries: reads (GET) are retried on 429, 5xx and network errors. Submissions (POST) create billable work, so they are
retried only when the request was certainly not processed: 429 (rate limited), 503 (model unavailable) and connection
timeouts.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence

import requests
from pydantic import BaseModel, ConfigDict

from wildai.labeling.config import PANGRAM_PUBLIC_API as PUBLIC_BASE_URL
from wildai.labeling.pangram.models import (
    TERMINAL_TASK_STAGES,
    BulkItem,
    BulkResult,
    BulkResultsPage,
    BulkStatus,
    BulkSubmission,
    Prediction,
)

API_KEY_ENV = "PANGRAM_API_KEY"
RESULTS_PAGE_LIMIT = 1000  # the documented maximum page size


class PangramAPIError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"Pangram API error {status}: {message}")
        self.status = status


class RetryPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempts: int = 6
    base_delay: float = 1.0
    max_delay: float = 60.0

    def delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after and retry_after.replace(".", "", 1).isdigit():
            return min(float(retry_after), self.max_delay)
        return min(self.base_delay * 2**attempt, self.max_delay)


class PangramClient:
    def __init__(self, api_key: str, base_url: str = PUBLIC_BASE_URL, retry: RetryPolicy | None = None,
                 timeout: float = 120.0, session: requests.Session | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not api_key:
            raise ValueError("an API key is required")
        self.base_url = base_url.rstrip("/")
        self.retry = retry or RetryPolicy()
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({"x-api-key": api_key, "Content-Type": "application/json"})
        self.sleep = sleep

    @classmethod
    def from_env(cls, base_url: str = PUBLIC_BASE_URL) -> PangramClient:
        key = os.environ.get(API_KEY_ENV, "")
        if not key:
            raise RuntimeError(f"set {API_KEY_ENV} to a Pangram API key (https://www.pangram.com/solutions/api)")
        return cls(key, base_url)

    def _request(self, method: str, path: str, *, json: object = None, params: dict[str, int] | None = None) -> dict:
        retry_statuses = {429, 500, 502, 503, 504} if method == "GET" else {429, 503}
        network_errors = (requests.ConnectionError, requests.Timeout) if method == "GET" else (requests.ConnectTimeout,)
        for attempt in range(self.retry.attempts):
            last = attempt == self.retry.attempts - 1
            try:
                response = self.session.request(method, self.base_url + path, json=json, params=params, timeout=self.timeout)
            except network_errors:
                if last:
                    raise
                self.sleep(self.retry.delay(attempt, None))
                continue
            if response.status_code in retry_statuses and not last:
                self.sleep(self.retry.delay(attempt, response.headers.get("Retry-After")))
                continue
            if response.status_code >= 400:
                raise PangramAPIError(response.status_code, response.text[:500])
            return response.json()
        raise AssertionError("unreachable")

    def list_models(self) -> list[str]:
        return list(self._request("GET", "/models")["models"])

    def submit_bulk(self, items: Sequence[BulkItem], model: str) -> BulkSubmission:
        body = {"items": [item.model_dump() for item in items], "model": model}
        return BulkSubmission.model_validate(self._request("POST", "/bulk", json=body))

    def bulk_status(self, bulk_id: str) -> BulkStatus:
        return BulkStatus.model_validate(self._request("GET", f"/bulk/{bulk_id}"))

    def bulk_results(self, bulk_id: str) -> list[BulkResult]:
        """Every item of a finished job, successful or failed, across all result pages."""

        results: list[BulkResult] = []
        offset, total = 0, None
        while total is None or offset < total:
            page = BulkResultsPage.model_validate(self._request(
                "GET", f"/bulk/{bulk_id}/results", params={"offset": offset, "limit": RESULTS_PAGE_LIMIT}))
            total = page.total_items
            results.extend(page.items)
            results.extend(BulkResult.model_validate(f.model_dump()) for f in page.failed_items)
            offset += RESULTS_PAGE_LIMIT
        return results

    def predict(self, text: str, model: str, poll_interval: float = 1.0, timeout: float = 300.0) -> Prediction:
        """Label one text with the task API, polling until it finishes."""

        task_id = self._request("POST", "/task", json={"text": text, "model": model})["task_id"]
        deadline = time.monotonic() + timeout
        while True:
            payload = self._request("GET", f"/task/{task_id}")
            if payload.get("stage") in TERMINAL_TASK_STAGES:
                return Prediction.model_validate(payload)
            if time.monotonic() > deadline:
                raise TimeoutError(f"task {task_id} did not finish in {timeout} s")
            self.sleep(poll_interval)
