"""Typed models of the public Pangram REST API (AI detection task and Bulk API), and the label record we store.

Response models ignore fields they do not declare, so additions to the API do not break parsing. Field meanings follow
https://docs.pangram.com/api-reference/ai-detection and https://docs.pangram.com/api-reference/bulk-api.
"""

from __future__ import annotations

from typing import Annotated, Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from wildai.data.arrow_schema import ArrowType

PangramLabel = Literal["Human", "Mixed", "AI"]
"""The document label (``prediction_short``)."""
BulkState = Literal["queued", "running", "succeeded", "failed", "partial"]
TERMINAL_BULK_STATES: frozenset[str] = frozenset({"succeeded", "failed", "partial"})
TERMINAL_TASK_STAGES: frozenset[str] = frozenset({"STAGE_SUCCESS", "STAGE_FAILED"})


class _Response(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class Window(_Response):
    """One independently classified segment of a document."""

    label: str
    ai_assistance_score: float
    confidence: str
    start_index: int
    end_index: int
    word_count: int = 0
    token_length: int = 0


class Prediction(_Response):
    """A completed task (``GET /task/{task_id}``, or ``result`` of a Bulk API item)."""

    stage: str
    version: str = ""
    headline: str = ""
    prediction_short: str = ""
    fraction_ai: float = 0.0
    fraction_ai_assisted: float = 0.0
    fraction_human: float = 0.0
    num_ai_segments: int = 0
    num_ai_assisted_segments: int = 0
    num_human_segments: int = 0
    windows: list[Window] = []


class BulkItem(BaseModel):
    """One input of a bulk job; ``id`` is our document id and comes back with the item's status and result."""

    model_config = ConfigDict(frozen=True)

    id: str
    text: str


class BulkItemStatus(_Response):
    index: int
    id: str | None = None
    task_id: str | None = None
    stage: str | None = None
    error: str | None = None


class BulkSubmission(_Response):
    """``POST /bulk`` response."""

    bulk_id: str
    status: BulkState
    total_items: int
    accepted_items: list[BulkItemStatus] = []
    failed_items: list[BulkItemStatus] = []


class BulkStatus(_Response):
    """``GET /bulk/{bulk_id}`` response."""

    bulk_id: str
    status: BulkState
    total_items: int
    accepted: int = 0
    succeeded: int = 0
    failed: int = 0

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_BULK_STATES


class BulkResult(BulkItemStatus):
    result: Prediction | None = None


class BulkResultsPage(_Response):
    """``GET /bulk/{bulk_id}/results`` response (one page)."""

    bulk_id: str
    offset: int
    limit: int
    total_items: int
    items: list[BulkResult] = []
    failed_items: list[BulkItemStatus] = []


class WindowScore(BaseModel):
    """A window as stored: offsets into the submitted text, the window label and its AI-assistance score."""

    model_config = ConfigDict(frozen=True)

    start_index: int
    end_index: int
    label: str
    ai_assistance_score: Annotated[float, ArrowType(pa.float32())]
    confidence: str


class PangramLabels(BaseModel):
    """The Pangram sidecar row of one document (also the cache record)."""

    model_config = ConfigDict(frozen=True)

    id: str
    pangram_label: PangramLabel | None
    """``None`` when the API could not label the document (see ``pangram_error``)."""
    pangram_fraction_ai: float | None
    pangram_fraction_ai_assisted: float | None
    pangram_fraction_human: float | None
    pangram_version: str | None
    pangram_model: str
    """The model selector the request used."""
    pangram_windows: list[WindowScore]
    pangram_sent_chars: int
    """Characters submitted (documents longer than the configured cap are labeled on their prefix)."""
    pangram_text_sha256: str
    pangram_error: str | None
